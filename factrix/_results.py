"""v0.14 result dataclasses — ``EvaluationResult`` / ``MetricResult`` / ``Warning``.

Lands the result-type group that #438 unification will surface from the
DAG executor (#442). This module ships the dataclasses + serialisation
methods only; the runner wiring (replacing ``FactorProfile`` /
``MetricsBundle`` returns) is a follow-up.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterator, KeysView, Mapping, ValuesView
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import WarningCode
from factrix._metric_index import MetricSpec, emitted_name_of
from factrix._types import MetricOutput

if TYPE_CHECKING:
    from collections.abc import ItemsView


@dataclass(frozen=True, slots=True)
class Warning:
    """Flat per-evaluation diagnostic record.

    Replaces the per-procedure ``frozenset[WarningCode]`` aggregation
    on :class:`factrix.FactorProfile` with an explicit tuple of
    ``(code, source, message)`` so callers can filter on the metric
    that emitted the warning.

    Attributes:
        code: The :class:`WarningCode` enum member.
        source: Metric name that emitted the warning, or ``None`` for
            bundle-level / cross-metric diagnostics (e.g. sample-guard
            failures detected before dispatch).
        message: Human-readable detail. Empty string when the code's
            registered description in
            :data:`factrix._codes._WARNING_DESCRIPTIONS` is sufficient.
    """

    code: WarningCode
    source: str | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Group of metric outputs for one factor at one cell.

    Mirrors the ``MetricGroups`` shape that #443 ``inspect_panel``
    exposes: three ``list[MetricSpec]`` partitions plus dict-like
    access to the produced :class:`MetricOutput` instances.

    Iteration / membership / ``keys`` / ``values`` / ``items`` all key
    by metric name (``MetricSpec.name``). Use :func:`emitted_name_of`
    on the spec when a user-facing label is needed.

    Attributes:
        applicable: Every spec applicable to the dispatched cell
            (superset of ``primary + diagnostic``).
        primary: Specs whose ``MetricOutput`` carries the bundle's
            primary p-value (driver of downstream multi-factor FDR).
        diagnostic: Specs whose output is descriptive / supplementary.
        outputs: ``metric_name -> MetricOutput`` for every spec that
            produced a value (including short-circuit outputs).
    """

    applicable: list[MetricSpec]
    primary: list[MetricSpec]
    diagnostic: list[MetricSpec]
    outputs: Mapping[str, MetricOutput] = field(default_factory=dict)

    def __getitem__(self, key: str) -> MetricOutput:
        return self.outputs[key]

    def __contains__(self, key: object) -> bool:
        return key in self.outputs

    def __iter__(self) -> Iterator[str]:
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

    def keys(self) -> KeysView[str]:
        return self.outputs.keys()

    def values(self) -> ValuesView[MetricOutput]:
        return self.outputs.values()

    def items(self) -> ItemsView[str, MetricOutput]:
        return self.outputs.items()


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Bundle-level result for one factor under one ``AnalysisConfig``.

    Single return type for the unified DAG executor (#438 / #442);
    replaces :class:`factrix.FactorProfile` (inferential) and
    :class:`factrix.MetricsBundle` (descriptive) once #442 lands.

    Attributes:
        factor: Factor column name from the source panel.
        axes: ``(scope, signal, metric)`` tuple of the dispatched cell.
            ``metric`` may be ``None`` when the cell is metric-wildcard.
        mode: ``Mode.PANEL`` or ``Mode.TIMESERIES`` resolved at dispatch.
        n_obs: Final-stage estimator sample size — the n the primary
            metric saw after trimming. Matches the cell's primary
            ``MetricOutput.n_obs`` when one exists; bundle-level field
            so consumers don't have to reach inside ``metrics``.
        n_assets: Unique assets in the panel under the any-non-null
            union (cell-invariant; ``1`` is legal for TIMESERIES).
        metrics: :class:`MetricResult` carrying the per-metric
            outputs and the applicable / primary / diagnostic spec
            partitions.
        warnings: Flat list of :class:`Warning` records. Per-metric
            entries carry ``source=metric_name``; cross-metric or
            pre-dispatch entries carry ``source=None``.
        plan: Multi-line execution plan emitted by the DAG executor —
            numbered topological order of every spec the executor ran,
            each line annotated with ``[batchable]`` / ``[per-factor]``,
            the ``requires=`` upstream list, and any stage-1
            share-hit marker. Required: callers constructing
            :class:`EvaluationResult` outside the DAG (tests, manual
            assembly) must supply an explicit string. There is no
            default to avoid silent ``""`` placeholders that obscure
            whether the DAG actually ran.
    """

    factor: str
    axes: tuple[FactorScope, Signal, Metric | None]
    mode: Mode
    n_obs: int
    n_assets: int
    metrics: MetricResult
    plan: str
    warnings: list[Warning] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        r"""One row per produced metric, prefixed with bundle identity.

        Schema (column order is stable):

        | column | dtype | source |
        |---|---|---|
        | ``factor`` | str | :attr:`factor` |
        | ``n_assets`` | i64 | :attr:`n_assets` |
        | ``metric_name`` | str | :func:`emitted_name_of` of the spec, falls back to mapping key |
        | ``value`` | f64 \\| null | ``MetricOutput.value`` (``NaN`` / ``Inf`` -> null) |
        | ``p`` | f64 \\| null | ``MetricOutput.metadata["p_value"]`` |
        | ``stat`` | f64 \\| null | ``MetricOutput.stat`` |
        | ``n_obs`` | i64 \\| null | ``MetricOutput.n_obs`` |
        | ``significance`` | str \\| null | ``MetricOutput.significance`` |
        | ``warning_codes`` | list[str] | per-metric :class:`Warning` codes (``source == metric_name``); empty list when none |

        Short-circuit rows surface as ``value=null`` / ``p=null``; the
        explanatory message lives on the matching :class:`Warning`
        record. Bundle-level warnings (``source is None``) do not
        produce rows; read them from :attr:`warnings` directly.

        Designed for stacking across factors:
        ``pl.concat([r.to_frame() for r in results])`` is the parquet
        write path.
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
            for key, out in self.metrics.outputs.items()
        ]
        return pl.DataFrame(rows, schema=_TO_FRAME_SCHEMA)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict view.

        Layout (top-level keys, stable order):

        - ``factor`` / ``axes`` / ``mode`` / ``n_obs`` / ``n_assets``
        - ``metrics``: dict ``metric_name -> MetricOutput-as-dict``
          (``value`` / ``p`` / ``stat`` / ``n_obs`` / ``significance``
          / ``metadata``)
        - ``metrics_partition``: ``{"primary": [...], "diagnostic": [...]}``
          listing metric names in each partition
        - ``warnings``: list of ``{code, source, message}`` dicts
        - ``plan``: the DAG execution plan string

        Float ``NaN`` / ``Inf`` are emitted as ``None`` so the dict
        survives ``json.dumps`` without ``allow_nan``.
        """
        scope, signal, metric = self.axes
        return {
            "factor": self.factor,
            "axes": {
                "scope": scope.value,
                "signal": signal.value,
                "metric": metric.value if metric is not None else None,
            },
            "mode": self.mode.value,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "metrics": {
                name: _metric_output_to_record(out)
                for name, out in self.metrics.outputs.items()
            },
            "metrics_partition": {
                "primary": [s.name for s in self.metrics.primary],
                "diagnostic": [s.name for s in self.metrics.diagnostic],
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
        scope, signal, metric = self.axes
        metric_token = metric.value if metric is not None else "*"
        header_rows = [
            ("factor", self.factor),
            ("axes", f"({scope.value}, {signal.value}, {metric_token})"),
            ("mode", self.mode.value),
            ("n_obs", self.n_obs),
            ("n_assets", self.n_assets),
            ("n_metrics", len(self.metrics)),
        ]
        if self.warnings:
            header_rows.append(("n_warnings", len(self.warnings)))
        header_html = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in header_rows
        )

        primary_names = {s.name for s in self.metrics.primary}
        metric_rows = []
        for name, out in sorted(self.metrics.outputs.items()):
            tag = "primary" if name in primary_names else "diagnostic"
            val_repr = "null" if math.isnan(out.value) else f"{out.value:.4g}"
            p_value = out.metadata.get("p_value")
            p_repr = f"{p_value:.4g}" if isinstance(p_value, float) else ""
            metric_rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{tag}</td>"
                f"<td style='text-align:right'>{val_repr}</td>"
                f"<td style='text-align:right'>{p_repr}</td>"
                f"<td>{html.escape(out.significance or '')}</td></tr>"
            )
        metric_table = (
            "<table><thead><tr><th>metric</th><th>role</th><th>value</th>"
            "<th>p</th><th>sig</th></tr></thead>"
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
    "p": pl.Float64,
    "stat": pl.Float64,
    "n_obs": pl.Int64,
    "significance": pl.Utf8,
    "warning_codes": pl.List(pl.Utf8),
}


def _output_row(
    key: str,
    out: MetricOutput,
    warnings_by_metric: Mapping[str, list[str]],
) -> dict[str, Any]:
    label = emitted_name_of(out.spec) if out.spec is not None else key
    return {
        "metric_name": label,
        "value": _float_or_none(out.value),
        "p": _float_or_none(out.metadata.get("p_value")),
        "stat": _float_or_none(out.stat),
        "n_obs": out.n_obs,
        "significance": out.significance,
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


def _metric_output_to_record(out: MetricOutput) -> dict[str, Any]:
    return {
        "value": _float_or_none(out.value),
        "p": _float_or_none(out.metadata.get("p_value")),
        "stat": _float_or_none(out.stat),
        "n_obs": out.n_obs,
        "significance": out.significance,
        "metadata": {k: _scrub_nonfinite(v) for k, v in out.metadata.items()},
    }


def _scrub_nonfinite(v: object) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
