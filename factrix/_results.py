"""v0.14 result dataclasses — ``EvaluationResult`` / ``MetricResult`` / ``Warning``.

Lands the result-type group that unification surfaces from the
DAG executor. This module ships the dataclasses + serialisation
methods.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from factrix._axis import DataStructure, FactorDensity, FactorScope
from factrix._codes import WarningCode


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Single-metric result produced by a ``factrix.metrics.*`` primitive.

    Attributes:
        value: Raw metric value.
        p_value: Two-sided p-value for the metric's hypothesis test.
            ``None`` for descriptive metrics that carry no formal test.
        n_obs: Effective sample size the estimator actually used
            (e.g. number of non-overlapping IC periods, number of
            events, number of bootstrap windows). ``None`` where a
            single integer count is not meaningful (e.g. multi-window
            CAAR series).
        stat: Test statistic (t, z, W, chi2, ...), when applicable.
        metadata: Estimator-specific context beyond the top-level fields
            (``stat_type``, ``h0``, ``method`` are the standard keys).
        warning_codes: Per-metric advisory :class:`WarningCode` values
            (as strings) the producer attached to *this* output.
            Empty tuple when the metric raised no advisory.
        name: Metric name stamped by the DAG executor at dispatch time.
            Empty string for outputs constructed outside the registry.
    """

    value: float
    p_value: float | None = None
    n_obs: int | None = None
    stat: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    warning_codes: tuple[str, ...] = ()
    name: str = ""

    def __repr__(self) -> str:
        name = self.name or "?"
        parts = [f"{name}={self.value:.4f}"]
        if self.p_value is not None:
            parts.append(f"p_value={self.p_value:.4g}")
        if self.n_obs is not None:
            parts.append(f"n_obs={self.n_obs}")
        if self.stat is not None:
            parts.append(f"stat={self.stat:.2f}")
        return f"MetricResult({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class Warning:
    """Flat per-evaluation diagnostic record.

    Source convention: a per-metric warning carries
    ``source == <metric label>``; a panel-level / cross-metric warning
    carries ``source is None``.

    Attributes:
        code: The :class:`WarningCode` enum member.
        source: Metric label that emitted the warning, or ``None`` for
            bundle-level diagnostics.
        message: Human-readable detail.
    """

    code: WarningCode
    source: str | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Bundle-level result for one factor.

    Attributes:
        factor: Factor column name from the source panel.
        cell: ``(scope, density, structure)`` tuple derived from the
            panel structure at dispatch time. ``structure`` is
            ``DataStructure.PANEL`` or ``DataStructure.TIMESERIES``
            resolved from the panel's asset count; ``scope`` and
            ``density`` default to INDIVIDUAL / DENSE.
        forward_periods: Forward-return horizon passed to
            :func:`factrix.evaluate`.
        n_periods: Number of unique dates in the factor panel where
            the factor column is non-null. A panel structural property —
            independent of any individual metric's estimator.
        n_pairs: Number of non-null ``(date, asset_id)`` pairs in the
            factor panel. A panel structural property.
        n_assets: Unique assets in the panel (cell-invariant;
            ``1`` is legal for TIMESERIES).
        metrics: Read-only ``label -> MetricResult`` mapping carrying
            per-metric outputs, keyed by the caller-supplied label.
        context: Caller-supplied free-form labels (e.g.
            ``{"region": "US"}``). Read by
            ``bhy(expand_over=...)`` / ``partial_conjunction`` /
            ``bhy_hierarchical`` to partition or aggregate inputs.
        warnings: Flat list of :class:`Warning` records. Per-metric
            entries carry ``source=label``; cross-metric or pre-dispatch
            entries carry ``source=None``.
        plan: Multi-line DAG execution plan (topological order,
            ``[batchable]`` / ``[per-factor]`` annotations).
    """

    factor: str
    cell: tuple[FactorScope, FactorDensity, DataStructure]
    forward_periods: int
    n_periods: int
    n_pairs: int
    n_assets: int
    metrics: Mapping[str, MetricResult]
    plan: str
    context: Mapping[str, Any] = field(default_factory=dict)
    warnings: list[Warning] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        r"""One row per produced metric, prefixed with bundle identity.

        Schema (column order is stable):

        | column | dtype | source |
        |---|---|---|
        | ``factor`` | str | :attr:`factor` |
        | ``n_assets`` | i64 | :attr:`n_assets` |
        | ``metric_name`` | str | ``MetricResult.name`` |
        | ``value`` | f64 \| null | ``MetricResult.value`` |
        | ``p_value`` | f64 \| null | ``MetricResult.p_value`` |
        | ``stat`` | f64 \| null | ``MetricResult.stat`` |
        | ``n_obs`` | i64 \| null | ``MetricResult.n_obs`` — estimator effective sample size |
        | ``warning_codes`` | list[str] | per-metric warning codes |

        Designed for stacking across factors:
        ``pl.concat([r.to_frame() for r in results.values()])``
        """
        by_metric: dict[str, list[str]] = {}
        for w in self.warnings:
            if w.source is None:
                continue
            by_metric.setdefault(w.source, []).append(w.code.value)
        rows = [
            {
                "factor": self.factor,
                "n_assets": self.n_assets,
                **_output_row(key, out, by_metric),
            }
            for key, out in self.metrics.items()
        ]
        return pl.DataFrame(rows, schema=_TO_FRAME_SCHEMA)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict view.

        Layout (top-level keys, stable order):

        - ``factor`` / ``cell`` / ``forward_periods`` / ``n_periods`` /
          ``n_pairs`` / ``n_assets`` / ``context``
        - ``metrics``: ``label -> {value, p_value, stat, n_obs, metadata}``
        - ``warnings``: list of ``{code, source, message}``
        - ``plan``

        Float ``NaN`` / ``Inf`` are emitted as ``None``.
        """
        scope, density, structure = self.cell
        return {
            "factor": self.factor,
            "cell": {
                "scope": scope.value,
                "density": density.value,
                "structure": structure.value,
            },
            "forward_periods": self.forward_periods,
            "n_periods": self.n_periods,
            "n_pairs": self.n_pairs,
            "n_assets": self.n_assets,
            "context": dict(self.context),
            "metrics": {
                name: _metric_output_to_record(out)
                for name, out in self.metrics.items()
            },
            "warnings": [
                {
                    "code": w.code.value,
                    "source": w.source,
                    "message": w.message,
                }
                for w in self.warnings
            ],
            "plan": self.plan,
        }

    def _repr_html_(self) -> str:
        scope, density, structure = self.cell
        header_rows: list[tuple[str, Any]] = [
            ("factor", self.factor),
            ("cell", f"({scope.value}, {density.value}, {structure.value})"),
            ("forward_periods", self.forward_periods),
            ("n_periods", self.n_periods),
            ("n_pairs", self.n_pairs),
            ("n_assets", self.n_assets),
            ("n_metrics", len(self.metrics)),
        ]
        if self.context:
            header_rows.append(("context", dict(self.context)))
        if self.warnings:
            header_rows.append(("n_warnings", len(self.warnings)))
        header_html = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in header_rows
        )

        metric_rows = []
        for name, out in sorted(self.metrics.items()):
            if isinstance(out, MetricResult):
                val_repr = "null" if math.isnan(out.value) else f"{out.value:.4g}"
                p_repr = f"{out.p_value:.4g}" if isinstance(out.p_value, float) else ""
            else:
                val_repr = "object"
                p_repr = ""
            metric_rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td style='text-align:right'>{val_repr}</td>"
                f"<td style='text-align:right'>{p_repr}</td></tr>"
            )
        metric_table = (
            "<table><thead><tr><th>metric</th><th>value</th>"
            "<th>p</th></tr></thead>"
            f"<tbody>{''.join(metric_rows)}</tbody></table>"
        )

        warnings_block = ""
        if self.warnings:
            w_rows = "".join(
                f"<tr><td>{html.escape(w.code.value)}</td>"
                f"<td>{html.escape(w.source or '')}</td>"
                f"<td>{html.escape(w.message)}</td></tr>"
                for w in self.warnings
            )
            warnings_block = (
                "<details open><summary>warnings "
                f"({len(self.warnings)})</summary>"
                "<table><thead><tr><th>code</th><th>source</th>"
                "<th>message</th></tr></thead>"
                f"<tbody>{w_rows}</tbody></table></details>"
            )

        plan_block = (
            "<details><summary>plan</summary>"
            f"<pre>{html.escape(self.plan)}</pre></details>"
            if self.plan
            else ""
        )
        return (
            "<div class='factrix-evaluation-result'>"
            "<table><caption>EvaluationResult</caption>"
            f"<tbody>{header_html}</tbody></table>"
            f"{metric_table}{warnings_block}{plan_block}"
            "</div>"
        )


_TO_FRAME_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "factor": pl.Utf8,
    "n_assets": pl.Int64,
    "metric_name": pl.Utf8,
    "value": pl.Float64,
    "p_value": pl.Float64,
    "stat": pl.Float64,
    "n_obs": pl.Int64,
    "warning_codes": pl.List(pl.Utf8),
}


def _output_row(
    key: str,
    out: Any,
    warnings_by_metric: Mapping[str, list[str]],
) -> dict[str, Any]:
    if not isinstance(out, MetricResult):
        return {
            "metric_name": key,
            "value": None,
            "p_value": None,
            "stat": None,
            "n_obs": None,
            "warning_codes": list(warnings_by_metric.get(key, [])),
        }
    label = out.name or key
    return {
        "metric_name": label,
        "value": _float_or_none(out.value),
        "p_value": _float_or_none(out.p_value),
        "stat": _float_or_none(out.stat),
        "n_obs": out.n_obs,
        "warning_codes": list(warnings_by_metric.get(label, [])),
    }


def _float_or_none(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _metric_output_to_record(out: Any) -> dict[str, Any]:
    if not isinstance(out, MetricResult):
        if hasattr(out, "__dict__"):
            return {
                "value": None,
                "p_value": None,
                "stat": None,
                "n_obs": None,
                "metadata": {"result": repr(out)},
            }
        return {
            "value": None,
            "p_value": None,
            "stat": None,
            "n_obs": None,
            "metadata": {},
        }
    return {
        "value": _float_or_none(out.value),
        "p_value": _float_or_none(out.p_value),
        "stat": _float_or_none(out.stat),
        "n_obs": out.n_obs,
        "metadata": {k: _scrub_nonfinite(v) for k, v in out.metadata.items()},
    }


def _scrub_nonfinite(v: object) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
