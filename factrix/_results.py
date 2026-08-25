"""v0.14 result dataclasses — ``EvaluationResult`` / ``MetricResult`` / ``Warning``.

Lands the result-type group that unification surfaces from the
DAG executor. This module ships the dataclasses + serialisation
methods.
"""

from __future__ import annotations

import html
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import polars as pl

from factrix._axis import DataStructure, FactorDensity, FactorScope
from factrix._codes import WarningCode
from factrix._types import SampleAxis

PValueAlternative = Literal["two-sided", "greater", "less"]

# Runtime mirror of ``PValueAlternative`` — a Literal is erased at runtime, so
# a typo'd alternative would otherwise flow through to ``to_frame`` / ``to_dict``
# and be read as a direction that was never tested.
_ALTERNATIVES: frozenset[str] = frozenset(get_args(PValueAlternative))


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Single-metric result produced by a ``factrix.metrics.*`` primitive.

    Attributes:
        value: Raw metric value.
        p_value: P-value for the metric's hypothesis test. ``None`` for
            descriptive metrics that carry no formal test.
        alternative: Alternative-hypothesis direction used to construct
            ``p_value``. Present exactly when ``p_value`` is present.
        n_obs: Effective sample size the estimator actually used
            (e.g. number of non-overlapping IC periods, number of
            events, number of bootstrap windows). ``None`` where a
            single integer count is not meaningful (e.g. multi-window
            CAAR series).
        n_obs_axis: Sample dimension ``n_obs`` counts along — one of
            ``"periods"`` / ``"events"`` / ``"pairs"`` / ``"assets"``.
            A bare count is uninterpretable without its axis (a
            Fama-MacBeth ``n_obs`` is periods; a pooled-OLS one is
            ``(date, asset)`` pairs), so producers stamp the axis
            alongside the count. ``None`` exactly when ``n_obs`` is.
        stat: Test statistic (t, z, W, chi2, ...), when applicable. Must be
            finite; ``NaN`` is accepted as the "estimator ran but could not
            form the statistic" marker, ``±Inf`` is rejected (it only ever
            arises from an unguarded division by a zero standard error).
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
    alternative: PValueAlternative | None = None
    n_obs: int | None = None
    n_obs_axis: SampleAxis | None = None
    stat: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    warning_codes: tuple[str, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        if (self.p_value is None) != (self.alternative is None):
            raise ValueError(
                "MetricResult: p_value and alternative must either both be "
                "provided or both be None."
            )
        # WHY: bool is a subclass of float's numeric tower — `True` passes the
        # [0, 1] range check and then serialises as a probability of 1.0.
        if isinstance(self.p_value, bool):
            raise ValueError(
                "MetricResult: p_value must be a float probability, not a bool; "
                f"got {self.p_value!r}."
            )
        if self.p_value is not None and (
            not math.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0
        ):
            raise ValueError(
                "MetricResult: p_value must be finite and lie in [0, 1]; "
                f"got {self.p_value!r}."
            )
        if self.alternative is not None and self.alternative not in _ALTERNATIVES:
            raise ValueError(
                "MetricResult: alternative must be one of "
                f"{sorted(_ALTERNATIVES)}; got {self.alternative!r}."
            )
        if self.n_obs is not None and self.n_obs < 0:
            raise ValueError(
                f"MetricResult: n_obs must be non-negative; got {self.n_obs!r}."
            )
        # WHY: NaN is a legitimate `stat` — an estimator that ran but could not
        # form its statistic reports NaN, the same "computed, undefined" marker
        # `value` uses on a short-circuit. ±Inf is not: it always comes from a
        # division by an unguarded zero standard error, which silently reads
        # out as infinite significance downstream.
        if self.stat is not None and math.isinf(self.stat):
            raise ValueError(
                "MetricResult: stat must be finite (NaN allowed for a statistic "
                f"that could not be formed); got {self.stat!r}."
            )

    def __repr__(self) -> str:
        name = self.name or "?"
        parts = [f"{name}={self.value:.4f}"]
        if self.p_value is not None:
            parts.append(f"p_value={self.p_value:.4g}")
            parts.append(f"alternative={self.alternative}")
        if self.n_obs is not None:
            axis = f" {self.n_obs_axis}" if self.n_obs_axis else ""
            parts.append(f"n_obs={self.n_obs}{axis}")
        if self.stat is not None:
            parts.append(f"stat={self.stat:.2f}")
        return f"MetricResult({', '.join(parts)})"

    @property
    def reason(self) -> str | None:
        """Stable short-circuit reason, when the metric did not run cleanly."""
        reason = self.metadata.get("reason")
        return reason if isinstance(reason, str) else None

    @property
    def is_applicable(self) -> bool:
        """Whether the metric produced a usable result for this input.

        ``strict=False`` short-circuits unsupported metrics into placeholder
        outputs with ``metadata["reason"]``. This flag lets reporting and
        grid-search code filter those rows without relying on metadata shape.
        """
        return self.reason is None


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
        expected: ``True`` when the caller declared this code as the
            study's design via ``evaluate(..., expected_warnings=(...,))``.
            The record is never dropped — the flag says "acknowledged",
            not "absent" — so human-facing channels (stderr echo, repr
            emphasis) can go quiet while the audit trail stays complete.
    """

    code: WarningCode
    source: str | None = None
    message: str = ""
    expected: bool = False


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
        forward_periods: The data's overlap horizon — read from the panel's
            ``compute_forward_return`` stamp (or the declared fallback for a
            self-attached panel). A property of the data, not a per-metric knob.
        n_periods: Number of unique dates in the factor panel where
            the factor column is non-null. A panel structural property —
            independent of any individual metric's estimator.
        n_pairs: Number of non-null ``(date, asset_id)`` pairs in the
            factor panel. A panel structural property.
        n_assets: Unique assets in the panel (cell-invariant;
            ``1`` is legal for TIMESERIES).
        metrics: Read-only ``label -> MetricResult`` mapping carrying
            per-metric outputs, keyed by the caller-supplied label.
            Structural metrics (e.g. ``greedy_forward_selection``) carry
            their non-scalar payload in ``MetricResult.metadata`` — the
            value is still a :class:`MetricResult`.
        params: Caller-supplied hypothesis parameters — the sweep knobs
            that decide *which* hypothesis this result is (e.g.
            ``{"timeframe": "1h", "universe": "tw50"}``). Every value joins
            the hypothesis identifier, so two results that differ only in
            ``params`` are two distinct hypotheses. ``expand_over`` may
            name these keys to partition a family.
        metadata: Caller-supplied bookkeeping labels that do *not* define
            the hypothesis (e.g. ``{"run_id": ..., "vintage": ...}``).
            Never joins the identifier and never partitions a family, so
            two results differing only in ``metadata`` still collide as a
            duplicate hypothesis.
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
    params: Mapping[str, Hashable] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: list[Warning] = field(default_factory=list)

    @property
    def unexpected_warnings(self) -> list[Warning]:
        """Warnings the caller did *not* declare via ``expected_warnings``.

        The alert view for pipelines and humans: :attr:`warnings` is the
        complete record (declared regimes included, flagged
        ``expected=True``); this subset is what still deserves attention.
        """
        return [w for w in self.warnings if not w.expected]

    def metric(self, label: str) -> MetricResult:
        """Return the :class:`MetricResult` for ``label``.

        Convenience over ``result.metrics[label]`` — the same lookup with a
        message that lists the available labels on a miss, so a typo in an
        interactive session fails loudly instead of with a bare ``KeyError``.
        """
        try:
            return self.metrics[label]
        except KeyError:
            available = ", ".join(sorted(self.metrics)) or "(none)"
            raise KeyError(
                f"no metric {label!r} on this result; available: {available}"
            ) from None

    def to_frame(self) -> pl.DataFrame:
        r"""One row per produced metric, prefixed with bundle identity.

        Schema (column order is stable):

        | column | dtype | source |
        |---|---|---|
        | ``factor`` | str | :attr:`factor` |
        | ``forward_periods`` | i64 | :attr:`forward_periods` |
        | *one column per* :attr:`params` *key* | inferred | :attr:`params` value |
        | ``n_assets`` | i64 | :attr:`n_assets` |
        | ``metric_name`` | str | ``MetricResult.name`` |
        | ``value`` | f64 \| null | ``MetricResult.value`` |
        | ``p_value`` | f64 \| null | ``MetricResult.p_value`` |
        | ``alternative`` | str \| null | ``MetricResult.alternative`` |
        | ``stat`` | f64 \| null | ``MetricResult.stat`` |
        | ``n_obs`` | i64 \| null | ``MetricResult.n_obs`` — estimator effective sample size |
        | ``n_obs_axis`` | str \| null | ``MetricResult.n_obs_axis`` — axis ``n_obs`` counts along (``periods`` / ``events`` / ``pairs`` / ``assets``) |
        | ``is_applicable`` | bool | false for ``strict=False`` short-circuits |
        | ``reason`` | str \| null | short-circuit reason when not applicable |
        | ``warning_codes`` | list[str] | per-metric warning codes — bundle records sourced on this metric, unioned (de-duplicated, first-seen order) with ``MetricResult.warning_codes`` |

        The leading ``factor`` / ``forward_periods`` / :attr:`params` block is
        the **hypothesis identity** — the same tuple :meth:`to_dict` and
        :func:`factrix.compare` carry. Without it, stacking the three results of
        an ``evaluate_horizons`` sweep produced three indistinguishable rows.

        Designed for stacking across factors:
        ``pl.concat([r.to_frame() for r in results.values()])``
        (results whose :attr:`params` keys differ need
        ``pl.concat(..., how="diagonal")``, as in :func:`factrix.compare`).

        Raises:
            ValueError: a :attr:`params` key collides with a fixed column name.
        """
        collisions = sorted(set(self.params) & set(_TO_FRAME_SCHEMA))
        if collisions:
            raise ValueError(
                f"EvaluationResult.to_frame(): params key(s) {collisions} collide "
                f"with fixed column name(s); rename the params key(s). Reserved: "
                f"{sorted(_TO_FRAME_SCHEMA)}"
            )
        by_metric: dict[str, list[str]] = {}
        for w in self.warnings:
            if w.source is None:
                continue
            by_metric.setdefault(w.source, []).append(w.code.value)
        rows = [
            {
                "factor": self.factor,
                "forward_periods": self.forward_periods,
                **dict(self.params),
                "n_assets": self.n_assets,
                **_output_row(key, out, by_metric),
            }
            for key, out in self.metrics.items()
        ]
        schema: dict[str, pl.DataType | type[pl.DataType] | None] = {
            "factor": pl.Utf8,
            "forward_periods": pl.Int64,
            # params values are caller-supplied Hashables — let polars infer.
            **dict.fromkeys(self.params),
            **{k: v for k, v in _TO_FRAME_SCHEMA.items() if k != "factor"},
        }
        return pl.DataFrame(rows, schema=schema)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict view.

        Layout (top-level keys, stable order):

        - ``factor`` / ``cell`` / ``forward_periods`` / ``n_periods`` /
          ``n_pairs`` / ``n_assets`` / ``params`` / ``metadata``
        - ``metrics``: ``label -> {value, p_value, alternative, stat, n_obs,
          n_obs_axis, is_applicable, reason, metadata}``
        - ``warnings``: list of ``{code, source, message, expected}``
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
            "params": dict(self.params),
            "metadata": dict(self.metadata),
            "metrics": {
                name: _metric_output_to_record(out)
                for name, out in self.metrics.items()
            },
            "warnings": [
                {
                    "code": w.code.value,
                    "source": w.source,
                    "message": w.message,
                    "expected": w.expected,
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
        if self.params:
            header_rows.append(("params", dict(self.params)))
        if self.metadata:
            header_rows.append(("metadata", dict(self.metadata)))
        if self.warnings:
            n_expected = sum(1 for w in self.warnings if w.expected)
            n_summary = (
                f"{len(self.warnings) - n_expected} (+{n_expected} expected)"
                if n_expected
                else str(len(self.warnings))
            )
            header_rows.append(("n_warnings", n_summary))
        header_html = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in header_rows
        )

        metric_rows = []
        for name, out in sorted(self.metrics.items()):
            val_repr = _html_value(out.value)
            p_repr = f"{out.p_value:.4g}" if isinstance(out.p_value, float) else ""
            alternative = out.alternative or ""
            metric_rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td style='text-align:right'>{val_repr}</td>"
                f"<td style='text-align:right'>{p_repr}</td>"
                f"<td>{html.escape(alternative)}</td></tr>"
            )
        metric_table = (
            "<table><thead><tr><th>metric</th><th>value</th>"
            "<th>p</th><th>alternative</th></tr></thead>"
            f"<tbody>{''.join(metric_rows)}</tbody></table>"
        )

        warnings_block = ""
        if self.warnings:
            w_rows = "".join(
                f"<tr><td>{html.escape(w.code.value)}</td>"
                f"<td>{html.escape(w.source or '')}</td>"
                f"<td>{'yes' if w.expected else ''}</td>"
                f"<td>{html.escape(w.message)}</td></tr>"
                for w in self.warnings
            )
            # A declared study collapses the block by default — the record
            # stays one click away, but only unexpected warnings demand
            # attention on open.
            details_open = " open" if self.unexpected_warnings else ""
            warnings_block = (
                f"<details{details_open}><summary>warnings "
                f"({len(self.warnings)})</summary>"
                "<table><thead><tr><th>code</th><th>source</th>"
                "<th>expected</th><th>message</th></tr></thead>"
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


def _html_value(value: object) -> str:
    """Cell text for a metric ``value`` in ``_repr_html_``.

    ``value`` is annotated ``float``, but structural metrics reach the HTML
    repr with whatever their producer put there (a bare ``None`` on a
    hand-built result, a label). ``math.isnan`` raises ``TypeError`` on those,
    which took down the whole notebook repr; fall back to the escaped ``str``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return html.escape(str(value))
    if math.isnan(value):
        return "null"
    return f"{value:.4g}"


_TO_FRAME_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "factor": pl.Utf8,
    "n_assets": pl.Int64,
    "metric_name": pl.Utf8,
    "value": pl.Float64,
    "p_value": pl.Float64,
    "alternative": pl.Utf8,
    "stat": pl.Float64,
    "n_obs": pl.Int64,
    "n_obs_axis": pl.Utf8,
    "is_applicable": pl.Boolean,
    "reason": pl.Utf8,
    "warning_codes": pl.List(pl.Utf8),
}


def _output_row(
    key: str,
    out: MetricResult,
    warnings_by_metric: Mapping[str, list[str]],
) -> dict[str, Any]:
    label = out.name or key
    # Union of the bundle-level records keyed on this metric and the codes the
    # producer stamped on the output itself, first-seen order, de-duplicated:
    # a short-circuit surfaces METRIC_UNAVAILABLE on both sides and must not
    # appear twice in the row.
    codes = dict.fromkeys((*warnings_by_metric.get(label, []), *out.warning_codes))
    return {
        "metric_name": label,
        "value": _float_or_none(out.value),
        "p_value": _float_or_none(out.p_value),
        "alternative": out.alternative,
        "stat": _float_or_none(out.stat),
        "n_obs": out.n_obs,
        "n_obs_axis": out.n_obs_axis,
        "is_applicable": out.is_applicable,
        "reason": out.reason,
        "warning_codes": list(codes),
    }


def _float_or_none(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _metric_output_to_record(out: MetricResult) -> dict[str, Any]:
    return {
        "value": _float_or_none(out.value),
        "p_value": _float_or_none(out.p_value),
        "alternative": out.alternative,
        "stat": _float_or_none(out.stat),
        "n_obs": out.n_obs,
        "n_obs_axis": out.n_obs_axis,
        "is_applicable": out.is_applicable,
        "reason": out.reason,
        "metadata": {k: _scrub_nonfinite(v) for k, v in out.metadata.items()},
    }


def _scrub_nonfinite(v: object) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
