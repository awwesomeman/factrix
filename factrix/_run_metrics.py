"""``run_metrics`` — descriptive batch runner parallel to ``evaluate``.

Where ``_evaluate`` dispatches a cell's primary inferential procedure
into a single ``FactorProfile``, ``run_metrics`` fans out across the
cell's standalone descriptive metrics into a single ``MetricsBundle``.
The two paths share the ``(panel, cfg)`` entry contract (#148) and
keep their result types disjoint.

v1 scope (#147): IC-cell stage-1 cache + panel-direct metrics for every
cell. Other stage-1 consumers (``caar`` / ``fama_macbeth`` / ``ts_beta``
/ ``mfe_mae_summary`` / series consumers / spread consumers) are listed
in :data:`factrix._metric_index._AUTO_DISCOVER_EXCLUDED` with explicit
import hints; per-cell stage-1 wiring is a v1.x follow-up.
"""

from __future__ import annotations

import html
import inspect
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import polars as pl

from factrix._errors import (
    InsufficientSampleError,
    RunMetricsError,
    UserInputError,
)
from factrix._metric_index import _AUTO_DISCOVER_EXCLUDED, user_facing_rows
from factrix._panel_input import PanelInput, _coerce_panel
from factrix._types import MetricOutput
from factrix.metrics._dispatch import is_batch_primitive, is_ic_consumer
from factrix.metrics._helpers import _short_circuit_output

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig

_logger = logging.getLogger("factrix.run_metrics")


@dataclass(frozen=True, slots=True, repr=False)
class MetricsBundle:
    """Cell-level descriptive metrics for one factor at one horizon.

    Parallel to :class:`factrix.FactorProfile` (the primary inferential
    artifact). The bundle holds every standalone metric ``run_metrics``
    successfully evaluated for the cell, plus a ``skipped`` map of
    metrics that could not auto-run (with reason).

    Hashing is disabled (``__hash__ = None``) because the bundle holds
    ``MetricOutput`` instances whose ``metadata`` is a mutable dict.
    Group bundles by ``identity`` (a hashable tuple), not by the bundle
    itself.

    Attributes:
        identity: ``(factor_id, forward_periods)`` — the hypothesis
            dimensions per #160. Aligns with ``FactorProfile.identity``
            so downstream functions (``compare`` / cross-slice
            analysis) can stack profiles and bundles by the same key.
        metrics: Metric name → ``MetricOutput`` mapping for every
            metric that produced a value (including short-circuit
            ``MetricOutput`` entries — ``value=NaN`` with a reason in
            ``metadata["reason"]``).
        skipped: Metric name → reason string for metrics that could
            not auto-run (excluded by the registry, requires explicit
            kwargs, stage-1 helper failed, …). v1 surfaces this through
            ``__repr__`` / ``_repr_html_`` and a single ``logging.info``.
        context: Sample-restriction / conditioning dimensions per #160.
            v1 always empty; downstream slicers (``factrix.by_slice``
            and the slice-test function pair) populate via
            ``dataclasses.replace`` once available, or callers stamp
            manually after panel-side filtering.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> bundle = fx.run_metrics(panel, fx.AnalysisConfig.individual_continuous())["factor"]
        >>> isinstance(bundle, fx.MetricsBundle)
        True
        >>> "ic" in bundle
        True
        >>> ic_output = bundle["ic"]
        >>> bundle.identity == ("factor", 5)
        True
    """

    identity: tuple[str, int]
    metrics: Mapping[str, MetricOutput] = field(default_factory=dict)
    skipped: Mapping[str, str] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]

    @property
    def factor_id(self) -> str:
        return self.identity[0]

    @property
    def forward_periods(self) -> int:
        return self.identity[1]

    def __getitem__(self, key: str) -> MetricOutput:
        return self.metrics[key]

    def __contains__(self, key: object) -> bool:
        return key in self.metrics

    def __iter__(self) -> Iterator[str]:
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    def _summary_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("factor_id", self.factor_id),
            ("forward_periods", self.forward_periods),
            ("n_metrics", len(self.metrics)),
        ]
        for k in sorted(self.context):
            rows.append((f"context.{k}", self.context[k]))
        if self.skipped:
            rows.append(("n_skipped", len(self.skipped)))
        return rows

    def __repr__(self) -> str:
        head = ", ".join(f"{k}={v!r}" for k, v in self._summary_rows())
        names = ", ".join(sorted(self.metrics))
        return f"MetricsBundle({head}, metrics=[{names}])"

    def _repr_html_(self) -> str:
        header = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in self._summary_rows()
        )
        metric_rows = "".join(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td style='text-align:right'>{out.value:.4g}</td>"
            f"<td>{html.escape(out.significance or '')}</td></tr>"
            for name, out in sorted(self.metrics.items())
        )
        skipped_block = ""
        if self.skipped:
            skipped_rows = "".join(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(reason)}</td></tr>"
                for name, reason in sorted(self.skipped.items())
            )
            skipped_block = (
                "<details><summary>skipped "
                f"({len(self.skipped)})</summary>"
                "<table><thead><tr><th>metric</th><th>reason</th>"
                "</tr></thead>"
                f"<tbody>{skipped_rows}</tbody></table></details>"
            )
        return (
            "<div class='factrix-metrics-bundle'>"
            "<table><caption>MetricsBundle</caption>"
            f"<tbody>{header}</tbody></table>"
            "<table><thead><tr><th>metric</th><th>value</th>"
            "<th>sig</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table>"
            f"{skipped_block}"
            "</div>"
        )

    def to_frame(self) -> pl.DataFrame:
        r"""Long-form view of the bundle: one row per metric.

        Schema (8 columns, stable regardless of bundle contents):

        | column | type | source |
        |---|---|---|
        | ``factor_id`` | str | ``identity[0]`` |
        | ``forward_periods`` | int | ``identity[1]`` |
        | ``metric`` | str | mapping key |
        | ``value`` | float | ``MetricOutput.value`` |
        | ``stat`` | float \\| null | ``MetricOutput.stat`` |
        | ``significance`` | str \\| null | ``MetricOutput.significance`` |
        | ``p_value`` | float \\| null | ``metadata["p_value"]`` |
        | ``short_circuit_reason`` | str \\| null | ``metadata["reason"]`` |

        ``metadata`` is **not** flattened — the per-metric shape is
        heterogeneous (per-regime dicts, per-horizon entries,
        Kolari-Pynnönen source labels, …) and a flat schema would
        fight every consumer. Reach into ``bundle["name"].metadata``
        for the rest.

        ``context.*`` is **not** flattened in v1 — ``context`` is
        always empty here; downstream slicers populating ``context``
        is a v1.x extension that may grow the schema.

        ``skipped`` entries do not appear; only successful and
        short-circuited ``MetricOutput`` rows are emitted.

        Examples:
            >>> import factrix as fx
            >>> from factrix.preprocess import compute_forward_return
            >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
            >>> panel = compute_forward_return(raw, forward_periods=5)
            >>> bundle = fx.run_metrics(panel, fx.AnalysisConfig.individual_continuous())["factor"]
            >>> df = bundle.to_frame()
            >>> set(df.columns) >= {"factor_id", "metric", "value", "p_value"}
            True
            >>> df.height >= 1
            True
        """
        rows = []
        for name, out in sorted(self.metrics.items()):
            meta = out.metadata or {}
            rows.append(
                {
                    "factor_id": self.factor_id,
                    "forward_periods": self.forward_periods,
                    "metric": name,
                    "value": float(out.value),
                    "stat": (None if out.stat is None else float(out.stat)),
                    "significance": out.significance,
                    "p_value": _coerce_optional_float(meta.get("p_value")),
                    "short_circuit_reason": _coerce_optional_str(meta.get("reason")),
                }
            )
        schema = {
            "factor_id": pl.String,
            "forward_periods": pl.Int64,
            "metric": pl.String,
            "value": pl.Float64,
            "stat": pl.Float64,
            "significance": pl.String,
            "p_value": pl.Float64,
            "short_circuit_reason": pl.String,
        }
        return pl.DataFrame(rows, schema=schema)


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _candidate_metric_names(scope: Any, signal: Any) -> list[str]:
    return [
        r.name
        for r in user_facing_rows()
        if r.cell.matches(scope, signal) and r.input_kind == "panel"
    ]


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    return name in inspect.signature(fn).parameters


def _cell_label(cfg: AnalysisConfig) -> str:
    return f"{cfg.scope.value}/{cfg.signal.value}"


def _raise_factor_col_error(*, value: object, expected: str) -> None:
    raise UserInputError(
        func_name="run_metrics",
        field="factor_cols",
        value=value,
        expected=expected,
        docs_path="api/run_metrics#factor_cols",
    )


def run_metrics(
    panel: PanelInput,
    cfg: AnalysisConfig,
    *,
    factor_cols: Sequence[str] = ("factor",),
    metrics: list[str] | None = None,
) -> dict[str, MetricsBundle]:
    """Run every standalone descriptive metric the cell exposes, for one or many factors.

    Parallel to :func:`factrix.evaluate`: both consume ``(panel, cfg)``,
    each produces a disjoint result type. ``run_metrics`` collects the
    cell's :mod:`factrix.metrics` surface — information coefficient (IC)
    family from one shared ``compute_ic`` (cache, batched across
    factors), batch-native primitives (``quantile_spread`` /
    ``monotonicity``) in one call per metric, every other panel-direct
    metric looped per factor on a thin projection — and returns one
    :class:`MetricsBundle` per factor keyed by factor name.

    Args:
        panel: Canonical-column panel (``date, asset_id, forward_return``
            plus every name in ``factor_cols``). Accepts
            ``pl.DataFrame`` or ``pl.LazyFrame`` (collected at the
            boundary). pandas users convert via :func:`factrix.adapt`
            or ``pl.from_pandas`` first; see
            :doc:`/guides/efficient-loading` for large-panel recipes.
        cfg: Validated :class:`AnalysisConfig`. ``cfg.scope`` and
            ``cfg.signal`` route metric discovery; ``cfg.forward_periods``
            (a single int) is the horizon every metric sees. Cross-horizon
            sweeps go through user comprehension into ``compare(bundles)``
            or ``bhy(profiles, expand_over=["forward_periods"])`` —
            ``run_metrics`` itself does not loop horizons (#147 §E /
            #186).
        factor_cols: Factor column names on ``panel`` to score. N=1 is
            the list-of-1 case of the general batch path. IC stage-1
            and batch-native primitives share one polars query across
            all factors; non-batch primitives are looped per factor on
            a thin ``select`` projection that renames the chosen factor
            to the canonical ``"factor"`` column. For 100+ factors
            where holding the full result dict pushes RSS, see
            :func:`run_metrics_chunked` — same dispatch path, scoped
            to one chunk's factors at a time.
        metrics: ``None`` (default) auto-discovers every applicable
            metric from :func:`factrix.list_metrics`. Pass an explicit
            list to run a subset; unknown or
            :data:`~factrix._metric_index._AUTO_DISCOVER_EXCLUDED`
            names raise :class:`factrix.UserInputError` with a fix
            path (a fuzzy suggestion plus the explicit-import recipe
            for stage-1 consumers).

    Returns:
        Dict mapping each factor name to its :class:`MetricsBundle`.
        Each bundle's ``metrics`` map carries every ``MetricOutput``
        produced (including short-circuit outputs for sample-floor
        failures) and ``skipped`` map carries per-metric reasons for
        metrics that could not auto-run.

    Raises:
        UserInputError: ``metrics`` contains an unknown name or a name
            that needs explicit kwargs not threaded by ``run_metrics``
            v1. Carries a fuzzy suggestion plus the documented reason
            from :data:`~factrix._metric_index._AUTO_DISCOVER_EXCLUDED`.
        RunMetricsError: Wraps an unexpected exception raised inside a
            metric callable or the IC stage-1 helper. Treat as a
            likely factrix bug; the original exception is chained via
            ``__cause__``. Sample-floor exceptions
            (:class:`factrix.InsufficientSampleError`) and metric-internal
            short-circuits are converted to short-circuit
            ``MetricOutput`` entries inside the bundle, **not** raised.

            For shared-compute paths — IC stage-1 (``compute_ic`` across
            all factors) and batch-native primitives (``quantile_spread``
            / ``monotonicity`` with ``factor_cols=cols``) — an
            unexpected exception aborts the entire batch; the cost of
            per-factor isolation would be paying the per-factor query
            plan we explicitly merged. Per-factor sample-floor
            short-circuits remain entries inside the bundle.

    Examples:
        Auto-discover every applicable metric for the canonical
        single-factor panel:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=250)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
        >>> bundles = fx.run_metrics(panel, cfg)
        >>> bundle = bundles["factor"]
        >>> ic_output = bundle["ic"]
        >>> long_frame = bundle.to_frame()

        Restrict to a subset by name:

        >>> bundles = fx.run_metrics(panel, cfg, metrics=["ic"])
    """
    panel = _coerce_panel(panel)
    cols = _validate_factor_cols(factor_cols)

    base_required = set(_DEFAULT_BASE_COLS)
    missing_base = base_required - set(panel.columns)
    if missing_base:
        hint = (
            "factrix.preprocess.compute_forward_return(panel) attaches forward_return"
            if "forward_return" in missing_base
            else f"add column(s) {sorted(missing_base)!r} to the panel"
        )
        _raise_factor_col_error(
            value=cols,
            expected=(
                f"panel with canonical columns "
                f"{{date, asset_id, forward_return}} plus every name in "
                f"factor_cols; missing {sorted(missing_base)!r} ({hint})"
            ),
        )

    missing_factors = [c for c in cols if c not in panel.columns]
    if missing_factors:
        _raise_factor_col_error(
            value=missing_factors,
            expected=(
                f"every name in factor_cols to exist on panel; "
                f"got columns {list(panel.columns)!r}"
            ),
        )

    if metrics is not None and not metrics:
        raise UserInputError(
            func_name="run_metrics",
            field="metrics",
            value=metrics,
            expected=(
                "a non-empty list of metric names, or None for "
                "auto-discover. An empty list is rejected to surface "
                "upstream filter mistakes that would otherwise produce "
                "an empty bundle silently."
            ),
            docs_path="api/run_metrics#metrics",
        )

    candidates = _candidate_metric_names(cfg.scope, cfg.signal)
    runnable = [m for m in candidates if m not in _AUTO_DISCOVER_EXCLUDED]
    excluded_reasons = {
        m: _AUTO_DISCOVER_EXCLUDED[m]
        for m in candidates
        if m in _AUTO_DISCOVER_EXCLUDED
    }

    if metrics is None:
        targets = runnable
    else:
        seen: set[str] = set()
        targets = []
        for name in metrics:
            if name in seen:
                continue
            seen.add(name)
            if name in runnable:
                targets.append(name)
                continue
            if name in excluded_reasons:
                raise UserInputError(
                    func_name="run_metrics",
                    field="metrics",
                    value=name,
                    expected=(
                        f"a metric run_metrics can dispatch directly. "
                        f"{name!r} is registered for this cell but "
                        f"requires explicit handling: "
                        f"{excluded_reasons[name]}"
                    ),
                    docs_path="api/run_metrics#explicit-required-kwargs",
                )
            raise UserInputError(
                func_name="run_metrics",
                field="metrics",
                value=name,
                candidates=runnable,
                docs_path="api/run_metrics#metrics",
            )

    skipped: dict[str, str] = dict(excluded_reasons) if metrics is None else {}
    results: dict[str, dict[str, MetricOutput]] = {c: {} for c in cols}
    ic_by_factor: dict[str, pl.DataFrame] | None = None
    # Per-factor projected views shared across every non-batch metric.
    # Built once per factor (not per metric × factor) — the underlying
    # polars buffers are refcounted so the projection is near-zero-copy.
    projected: dict[str, pl.DataFrame] = {c: _project_factor(panel, c) for c in cols}

    import factrix.metrics as _metrics

    for name in targets:
        fn = getattr(_metrics, name)
        kwargs: dict[str, Any] = {}
        if _accepts_kwarg(fn, "forward_periods"):
            kwargs["forward_periods"] = cfg.forward_periods
        stage = "consumer"
        try:
            if is_ic_consumer(fn):
                if ic_by_factor is None:
                    stage = "stage1"
                    ic_by_factor = _metrics.compute_ic(panel, factor_cols=cols)
                    stage = "consumer"
                for c in cols:
                    results[c][name] = _safe_call(name, fn, ic_by_factor[c], kwargs)
            elif is_batch_primitive(fn):
                # Batch-native primitives take one call across the whole
                # batch and return ``dict[str, MetricOutput]`` keyed per
                # factor. They surface sample-floor failures as
                # short-circuit entries inside that dict, not via
                # exception (so no per-factor ``_safe_call`` wrap
                # needed). An unexpected exception from the call aborts
                # the whole batch — see ``Raises``.
                out = fn(panel, factor_cols=cols, **kwargs)
                for c in cols:
                    results[c][name] = out[c]
            else:
                for c in cols:
                    results[c][name] = _safe_call(name, fn, projected[c], kwargs)
        except (InsufficientSampleError, RunMetricsError):
            raise
        except Exception as exc:
            raise RunMetricsError(
                f"run_metrics() failed in metric={name!r} "
                f"(cell={_cell_label(cfg)}, stage={stage}). "
                f"Likely a factrix bug — please report.",
                cell=_cell_label(cfg),
                metric_name=name,
                stage=stage,
            ) from exc

    if skipped:
        _logger.info(
            "run_metrics(cell=%s, factor_ids=%s, fwd=%d): ran=%d skipped=%d (%s)",
            _cell_label(cfg),
            cols,
            cfg.forward_periods,
            len(targets),
            len(skipped),
            ", ".join(sorted(skipped)),
        )

    return {
        c: MetricsBundle(
            identity=(c, cfg.forward_periods),
            metrics=MappingProxyType(results[c]),
            skipped=MappingProxyType(skipped),
            context=MappingProxyType({}),
        )
        for c in cols
    }


_DEFAULT_BASE_COLS: tuple[str, ...] = ("date", "asset_id", "forward_return")

# Per-factor in-memory amplification factor used by the chunk-size
# heuristic. Each chunk's peak RSS is dominated by panel materialise +
# polars query intermediates (``with_columns(rank_exprs)`` adds one
# ``_rank__<f>`` column per factor); ``4 ×`` an 8-byte-per-row column
# slice empirically tracks observed M-ic / S2 peak_rss within ±20%
# across the small / large presets — close enough for a budget knob.
_AUTO_CHUNK_OVERHEAD_FACTOR = 4

# Divisor applied to ``psutil.virtual_memory().available`` to derive
# the per-chunk peak budget. Dividing by 4 (i.e. targeting ~25%) leaves
# slack for OS, BLAS arenas, and the caller's downstream sink without
# forcing every batch back to chunk_size=1 on tight machines.
_AUTO_CHUNK_RSS_DIVISOR = 4


def _validate_factor_cols(factor_cols: Sequence[str]) -> list[str]:
    """Shared factor_cols validation for run_metrics + run_metrics_chunked."""
    cols = list(factor_cols)
    if not cols:
        _raise_factor_col_error(
            value=cols,
            expected="a non-empty list of factor column names",
        )
    if len(set(cols)) != len(cols):
        _raise_factor_col_error(
            value=cols,
            expected="factor_cols with no duplicates",
        )
    return cols


def _auto_chunk_size(n_rows: int, n_factors: int) -> int:
    """Pick chunk size targeting ~25% of available RAM as per-chunk peak.

    ``psutil`` is an optional dependency (``factrix[bench]`` extras);
    when absent, callers must pass ``chunk_size`` explicitly — the
    function raises so the user reaches for the right knob rather
    than silently getting a degenerate default.
    """
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError as exc:
        raise UserInputError(
            func_name="run_metrics_chunked",
            field="chunk_size",
            value=None,
            expected=(
                "an explicit positive integer when psutil is not installed; "
                "auto-sizing requires `pip install psutil` (or "
                "`pip install 'factrix[bench]'`)"
            ),
            docs_path="api/run_metrics_chunked#chunk_size",
        ) from exc

    available = psutil.virtual_memory().available
    per_factor_bytes = max(n_rows * 8 * _AUTO_CHUNK_OVERHEAD_FACTOR, 1)
    budget = max(available // _AUTO_CHUNK_RSS_DIVISOR, per_factor_bytes)
    return max(1, min(n_factors, budget // per_factor_bytes))


def run_metrics_chunked(
    panel: PanelInput,
    cfg: AnalysisConfig,
    *,
    factor_cols: Sequence[str],
    metrics: list[str] | None = None,
    chunk_size: int | None = None,
    base_cols: Sequence[str] = _DEFAULT_BASE_COLS,
) -> Iterator[dict[str, MetricsBundle]]:
    """Yield ``run_metrics`` output one chunk of factors at a time.

    Splits ``factor_cols`` into chunks, narrows ``panel`` to
    ``base_cols + chunk`` per iteration, calls :func:`run_metrics`, and
    yields each chunk's ``dict[factor_id, MetricsBundle]``. Peak RSS is
    bounded by the chunk size rather than ``len(factor_cols)`` — so a
    1000-factor screen that would otherwise exceed RAM can stream
    through a fixed working-set budget.

    Within a chunk the full batch-dispatch path runs unchanged: IC
    stage-1 is shared across the chunk's factors, batch-native
    primitives take one call per metric. **Across chunks, IC stage-1
    is recomputed per chunk** — chunking trades cross-chunk stage-1
    reuse for the RSS bound, so very small ``chunk_size`` (e.g. 1)
    pays the per-chunk dispatcher overhead without the batch-sharing
    benefit. Per-factor streaming yield is a separate concern.

    Args:
        panel: Same contract as :func:`run_metrics`. When passed a
            ``pl.LazyFrame``, the height is sampled via
            ``select(pl.len()).collect()`` (one row) and each chunk
            does a fresh ``panel.select([...]).collect()`` so
            projection pushdown applies per chunk — only the chunk's
            factor columns get scanned from the source.
        cfg: Same as :func:`run_metrics`.
        factor_cols: Factor columns to chunk over. Must be non-empty
            and contain no duplicates. ``base_cols`` plus every factor
            in this list must exist on ``panel`` — schema is checked
            eagerly before the first chunk yields.
        metrics: Same as :func:`run_metrics`.
        chunk_size: Number of factors per chunk. ``None`` (default)
            picks a chunk size targeting ~25% of available RAM via
            :func:`_auto_chunk_size`, which requires ``psutil`` (an
            optional dependency — install via ``pip install psutil``
            or ``pip install 'factrix[bench]'``). Pass an explicit
            value to override (e.g. when the working sink has its
            own batching cadence). An explicit ``chunk_size`` larger
            than ``len(factor_cols)`` is accepted and degenerates to
            a single chunk.
        base_cols: Panel columns required by every chunk regardless of
            which factor subset is active. Default
            ``("date", "asset_id", "forward_return")`` matches
            :func:`run_metrics`'s base contract. Override when extra
            columns are required (e.g. ``weight_col`` for
            ``quantile_spread_vw``).

    Yields:
        ``dict[factor_id, MetricsBundle]`` — same shape as
        :func:`run_metrics`, scoped to one chunk's factors. Iterate
        the generator to consume chunks sequentially; each chunk's
        bundles can be written to a sink and released before the
        next chunk is produced.

    Raises:
        UserInputError: ``factor_cols`` empty / contains duplicates,
            or ``panel`` missing a ``base_cols`` / factor column,
            or ``chunk_size=None`` and ``psutil`` is not installed.
        ValueError: ``chunk_size`` non-positive.
        TypeError: ``panel`` not ``pl.DataFrame`` or ``pl.LazyFrame``.

    Examples:
        Stream 1000 factors through a parquet sink, 100 per chunk:

        >>> import factrix as fx                                   # doctest: +SKIP
        >>> for bundles in fx.run_metrics_chunked(                 # doctest: +SKIP
        ...     panel, cfg, factor_cols=cols, chunk_size=100,
        ... ):
        ...     for fid, bundle in bundles.items():
        ...         sink.write(fid, bundle.to_frame())

        Auto-sized chunks (default):

        >>> for bundles in fx.run_metrics_chunked(                 # doctest: +SKIP
        ...     panel, cfg, factor_cols=cols,
        ... ):
        ...     ...
    """
    cols = _validate_factor_cols(factor_cols)
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")

    if not isinstance(panel, pl.DataFrame | pl.LazyFrame):
        raise TypeError(
            f"panel must be pl.DataFrame or pl.LazyFrame; got {type(panel).__name__}"
        )

    # Schema check up front so a wrong ``base_cols`` / missing factor
    # fails before the first yield (matches run_metrics' eager UX
    # contract instead of failing mid-iteration with a polars error).
    schema_cols = (
        set(panel.collect_schema().names())
        if isinstance(panel, pl.LazyFrame)
        else set(panel.columns)
    )
    base = list(base_cols)
    missing = (set(base) | set(cols)) - schema_cols
    if missing:
        _raise_factor_col_error(
            value=cols,
            expected=(
                f"panel with all of base_cols + factor_cols present; "
                f"missing {sorted(missing)!r}"
            ),
        )

    if chunk_size is None:
        n_rows = (
            panel.select(pl.len()).collect().item()
            if isinstance(panel, pl.LazyFrame)
            else panel.height
        )
        cs = _auto_chunk_size(n_rows, len(cols))
    else:
        cs = chunk_size

    from itertools import batched

    for chunk_tuple in batched(cols, cs):
        chunk = list(chunk_tuple)
        projection = [*base, *chunk]
        sub_panel = (
            panel.select(projection).collect()
            if isinstance(panel, pl.LazyFrame)
            else panel.select(projection)
        )
        yield run_metrics(sub_panel, cfg, factor_cols=chunk, metrics=metrics)


def _project_factor(panel: pl.DataFrame, col: str) -> pl.DataFrame:
    """Thin per-factor projection: rename ``col`` to canonical ``"factor"``.

    Non-batch primitives expect a panel whose factor column is literally
    ``"factor"``. Always projects to exactly the 4 canonical columns
    so primitives see an identical schema regardless of whether the
    caller's ``factor_cols`` happened to include ``"factor"`` itself
    or a sibling extra column.
    """
    return panel.select(
        pl.col("date"),
        pl.col("asset_id"),
        pl.col("forward_return"),
        pl.col(col).alias("factor"),
    )


def _safe_call(
    name: str,
    fn: Callable[..., MetricOutput],
    arg: pl.DataFrame,
    kwargs: dict[str, Any],
) -> MetricOutput:
    """Invoke ``fn(arg, **kwargs)`` converting sample-floor errors to short-circuit outputs."""
    try:
        return fn(arg, **kwargs)
    except InsufficientSampleError as exc:
        return _short_circuit_output(
            name,
            "insufficient_sample",
            actual_periods=exc.actual_periods,
            required_periods=exc.required_periods,
        )
