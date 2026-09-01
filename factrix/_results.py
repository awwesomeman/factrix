"""v0.14 result dataclasses — ``EvaluationResult`` / ``MetricResult`` / ``Warning``.

Lands the result-type group that unification surfaces from the
DAG executor. This module ships the dataclasses + serialisation
methods.
"""

from __future__ import annotations

import contextlib
import html
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args

import polars as pl

from factrix._axis import DataStructure, FactorDensity, FactorScope
from factrix._codes import WarningCode
from factrix._errors import UserInputError
from factrix._types import PValueAlternative, SampleAxis

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
            raise UserInputError(
                func_name="MetricResult",
                field="alternative",
                value=self.alternative,
                expected=(
                    "p_value and alternative supplied together, or both None: "
                    "a p-value without the alternative it was computed under "
                    "is uninterpretable."
                ),
                docs_path="api/evaluation-results#factrix.MetricResult",
            )
        # WHY: bool is a subclass of float's numeric tower — `True` passes the
        # [0, 1] range check and then serialises as a probability of 1.0.
        if isinstance(self.p_value, bool):
            raise UserInputError(
                func_name="MetricResult",
                field="p_value",
                value=self.p_value,
                expected=(
                    "a float probability, not a bool. A bool passes the "
                    "[0, 1] range check and then serialises as a probability "
                    "of 1.0."
                ),
                docs_path="api/evaluation-results#factrix.MetricResult",
            )
        if self.p_value is not None and (
            not math.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0
        ):
            raise UserInputError(
                func_name="MetricResult",
                field="p_value",
                value=self.p_value,
                expected="a finite probability inside [0, 1]",
                docs_path="api/evaluation-results#factrix.MetricResult",
            )
        if self.alternative is not None and self.alternative not in _ALTERNATIVES:
            raise UserInputError(
                func_name="MetricResult",
                field="alternative",
                value=self.alternative,
                candidates=sorted(_ALTERNATIVES),
                docs_path="api/evaluation-results#factrix.MetricResult",
            )
        if self.n_obs is not None and self.n_obs < 0:
            raise UserInputError(
                func_name="MetricResult",
                field="n_obs",
                value=self.n_obs,
                expected="a non-negative count",
                docs_path="api/evaluation-results#factrix.MetricResult",
            )
        # WHY: NaN is a legitimate `stat` — an estimator that ran but could not
        # form its statistic reports NaN, the same "computed, undefined" marker
        # `value` uses on a short-circuit. ±Inf is not: it always comes from a
        # division by an unguarded zero standard error, which silently reads
        # out as infinite significance downstream.
        if self.stat is not None and math.isinf(self.stat):
            raise UserInputError(
                func_name="MetricResult",
                field="stat",
                value=self.stat,
                expected=(
                    "a finite statistic, or NaN for one that could not be "
                    "formed. An infinite stat always comes from a division by "
                    "an unguarded zero standard error."
                ),
                docs_path="api/evaluation-results#factrix.MetricResult",
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
        forward_periods: The economic return horizon — the ``forward_periods``
            ``forward_return`` was built with, in periods of the price grid.
            Read from the panel's ``compute_forward_return`` stamp (or the
            declared fallback for a self-attached panel). A property of the
            data, not a per-metric knob; joins the hypothesis identity.
        overlap_periods: The overlap of adjacent observations on the
            evaluation grid — the quantity inference consumed (HAC bandwidth
            and effective df, non-overlapping stride, stride-scaled floors).
            Equal to ``forward_periods`` on the full grid; smaller on a
            coarser evaluation grid (``compute_forward_return(..., dates=)``).
            Bookkeeping only — it does **not** join the hypothesis identity,
            because the same horizon evaluated on two grids is one hypothesis
            estimated twice, not two hypotheses.
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
    overlap_periods: int
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

    def to_frame(self, *, metadata: Sequence[str] = ()) -> pl.DataFrame:
        r"""One row per produced metric, prefixed with bundle identity.

        Schema (column order is stable; ``metadata=`` columns, if any, follow
        ``warning_codes`` in the order requested):

        | column | dtype | source |
        |---|---|---|
        | ``factor`` | str | :attr:`factor` |
        | ``forward_periods`` | i64 | :attr:`forward_periods` |
        | *one column per* :attr:`params` *key* | inferred | :attr:`params` value |
        | ``overlap_periods`` | i64 | :attr:`overlap_periods` (bookkeeping, not identity) |
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

        **Exporting estimator metadata.** The fixed schema is the stable
        cross-metric contract; estimator-specific definitions (a spread's
        ``n_groups``, a turnover's ``rebalance_lag`` / ``mean_tail_size``)
        live in ``MetricResult.metadata`` and do not leak into it by default.
        ``metadata=`` names the keys to carry along, one column per key, so a
        CSV written from the stacked frame can be audited without the
        ``MetricResult`` objects:

        - the column is named after the key and keeps the row's
          ``factor`` / ``metric_name`` pairing — stacking many results keeps
          every metadata cell next to the metric it came from;
        - a metric that does not carry the key gets ``null`` (a spread row
          has no ``rebalance_lag``);
        - only **scalar** values are exported — ``bool`` / ``int`` /
          ``float`` / ``str`` (``NaN`` / ``Inf`` become ``null`` like
          ``value``; numpy scalars are unwrapped). A list, dict, tuple or
          other nested value raises ``UserInputError`` naming the metric and
          key — those are what :meth:`to_dict` is for;
        - a key colliding with a fixed column, a :attr:`params` key or a
          repeated key raises ``UserInputError``, so metadata never shadows
          identity or the fixed schema;
        - dtypes are inferred per column from the values present.

        Raises:
            UserInputError: a :attr:`params` key collides with a fixed column
                name; a ``metadata`` key collides, repeats, or names a
                non-scalar value on some metric.
        """
        collisions = sorted(set(self.params) & set(_TO_FRAME_SCHEMA))
        if collisions:
            raise UserInputError(
                func_name="EvaluationResult.to_frame",
                field="params",
                value=collisions,
                expected=(
                    f"params key(s) that do not collide with a fixed column "
                    f"name; rename them. Reserved: {sorted(_TO_FRAME_SCHEMA)}"
                ),
                docs_path="api/evaluation-results#to_frame",
            )
        meta_keys = _validate_metadata_keys(metadata, params=self.params)
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
                "overlap_periods": self.overlap_periods,
                "n_assets": self.n_assets,
                **_output_row(key, out, by_metric),
                **_metadata_columns(key, out, meta_keys),
            }
            for key, out in self.metrics.items()
        ]
        schema: dict[str, pl.DataType | type[pl.DataType] | None] = {
            "factor": pl.Utf8,
            "forward_periods": pl.Int64,
            # params values are caller-supplied Hashables — let polars infer.
            **dict.fromkeys(self.params),
            "overlap_periods": pl.Int64,
            **{k: v for k, v in _TO_FRAME_SCHEMA.items() if k != "factor"},
            # Metadata values are estimator-specific scalars — inferred too.
            **dict.fromkeys(meta_keys),
        }
        return pl.DataFrame(rows, schema=schema)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict view.

        Layout (top-level keys, stable order):

        - ``factor`` / ``cell`` / ``forward_periods`` / ``overlap_periods`` /
          ``n_periods`` / ``n_pairs`` / ``n_assets`` / ``params`` / ``metadata``
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
            "overlap_periods": self.overlap_periods,
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
            ("overlap_periods", self.overlap_periods),
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


# Columns ``to_frame`` always emits ahead of the fixed schema; a metadata key
# may not shadow any of them.
_TO_FRAME_IDENTITY: tuple[str, ...] = ("forward_periods", "overlap_periods")


def _validate_metadata_keys(
    metadata: object, *, params: Mapping[str, Hashable]
) -> tuple[str, ...]:
    """Normalise ``to_frame(metadata=)``: a sequence of distinct, non-reserved keys."""
    if isinstance(metadata, str) or not isinstance(metadata, Sequence):
        raise UserInputError(
            func_name="EvaluationResult.to_frame",
            field="metadata",
            value=metadata,
            expected=(
                "a sequence of metadata key names, e.g. "
                "metadata=('n_groups', 'rebalance_lag')"
            ),
            docs_path="api/evaluation-results#to_frame",
        )
    keys = tuple(metadata)
    bad = [k for k in keys if not isinstance(k, str)]
    if bad:
        raise UserInputError(
            func_name="EvaluationResult.to_frame",
            field="metadata",
            value=bad,
            expected="every metadata key a string",
            docs_path="api/evaluation-results#to_frame",
        )
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise UserInputError(
            func_name="EvaluationResult.to_frame",
            field="metadata",
            value=repeated,
            expected="distinct metadata keys",
            docs_path="api/evaluation-results#to_frame",
        )
    reserved = set(_TO_FRAME_SCHEMA) | set(_TO_FRAME_IDENTITY) | set(params)
    collisions = sorted(set(keys) & reserved)
    if collisions:
        raise UserInputError(
            func_name="EvaluationResult.to_frame",
            field="metadata",
            value=collisions,
            expected=(
                f"metadata key(s) that shadow neither a fixed column nor a "
                f"params key: the fixed schema and the hypothesis identity are "
                f"never shadowed. Reserved: {sorted(reserved)}"
            ),
            docs_path="api/evaluation-results#to_frame",
        )
    return keys


def _metadata_columns(
    key: str, out: MetricResult, meta_keys: Sequence[str]
) -> dict[str, Any]:
    """One scalar cell per requested metadata key; ``None`` where absent."""
    label = out.name or key
    return {k: _scalar_metadata(label, k, out.metadata.get(k)) for k in meta_keys}


def _scalar_metadata(label: str, key: str, value: object) -> Any:
    """Coerce a metadata value to a frame cell or refuse a nested one."""
    if value is None:
        return None
    # numpy scalars (np.float64 is a float subclass, np.int64 is not) unwrap
    # via ``item()``; strings expose no such method.
    if not isinstance(value, (bool, int, float, str)) and hasattr(value, "item"):
        # A multi-element array also has ``item()`` and refuses; it then falls
        # through to the nested-value error below.
        with contextlib.suppress(TypeError, ValueError):
            value = value.item()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return _float_or_none(value)
    raise UserInputError(
        func_name="EvaluationResult.to_frame",
        field=f"metadata[{key!r}]",
        value=value,
        expected=(
            f"a scalar on metric {label!r}: metadata= exports bool / int / "
            f"float / str cells only. Use to_dict() for nested values."
        ),
        docs_path="api/evaluation-results#to_frame",
    )


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
