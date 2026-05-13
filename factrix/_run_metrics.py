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
from collections.abc import Iterator, Mapping
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
from factrix._types import MetricOutput
from factrix.metrics._helpers import _short_circuit_output

if TYPE_CHECKING:
    from collections.abc import Callable

    from factrix._analysis_config import AnalysisConfig

_logger = logging.getLogger("factrix.run_metrics")

# Metrics whose first positional argument is the ``compute_ic`` output
# (the ``(date, ic, tie_ratio)`` frame). v1 wires this single stage-1
# helper; expanding to ``compute_fm_betas`` / ``compute_ts_betas`` /
# ``compute_event_returns`` pipelines is tracked as v1.x follow-up.
_IC_CONSUMERS: frozenset[str] = frozenset({"ic", "ic_newey_west", "ic_ir"})


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
            so downstream verbs (``compare`` / cross-slice analysis)
            can stack profiles and bundles by the same key.
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
            and the slice-test verb pair) populate via
            ``dataclasses.replace`` once available, or callers stamp
            manually after panel-side filtering.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> bundle = fx.run_metrics(panel, fx.AnalysisConfig.individual_continuous())
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
        """Long-form view of the bundle: one row per metric.

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
            >>> bundle = fx.run_metrics(panel, fx.AnalysisConfig.individual_continuous())
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


def _raise_factor_col_error(*, value: str, expected: str) -> None:
    raise UserInputError(
        func_name="run_metrics",
        field="factor_col",
        value=value,
        expected=expected,
        docs_path="api/run_metrics#factor_col",
    )


def run_metrics(
    panel: pl.DataFrame,
    cfg: AnalysisConfig,
    *,
    factor_col: str = "factor",
    metrics: list[str] | None = None,
) -> MetricsBundle:
    """Run every standalone descriptive metric the cell exposes.

    Parallel to :func:`factrix.evaluate`: both consume ``(panel, cfg)``,
    each produces a disjoint result type. ``run_metrics`` collects the
    cell's :mod:`factrix.metrics` surface — IC family from one shared
    ``compute_ic`` (cache), every other panel-direct metric called
    directly — and returns a :class:`MetricsBundle` keyed by metric name.

    Args:
        panel: Canonical-column panel (``date, asset_id, factor,
            forward_return``). Renamed internally if ``factor_col`` is
            not ``"factor"``, mirroring :func:`factrix.evaluate`.
        cfg: Validated :class:`AnalysisConfig`. ``cfg.scope`` and
            ``cfg.signal`` route metric discovery; ``cfg.forward_periods``
            (a single int) is the horizon every metric sees. Cross-horizon
            sweeps go through user comprehension into ``compare(bundles)``
            or ``bhy(profiles, expand_over=["forward_periods"])`` —
            ``run_metrics`` itself does not loop horizons (#147 §E /
            #186).
        factor_col: Name of the signal column on ``panel``. Renamed to
            ``"factor"`` internally before dispatch so metric callables
            see the canonical schema.
        metrics: ``None`` (default) auto-discovers every applicable
            metric from :func:`factrix.list_metrics`. Pass an explicit
            list to run a subset; unknown or
            :data:`~factrix._metric_index._AUTO_DISCOVER_EXCLUDED`
            names raise :class:`factrix.UserInputError` with a fix
            path (a fuzzy suggestion plus the explicit-import recipe
            for stage-1 consumers).

    Returns:
        A :class:`MetricsBundle` whose ``metrics`` map carries every
        ``MetricOutput`` produced (including short-circuit outputs for
        sample-floor failures) and whose ``skipped`` map carries
        per-metric reasons for metrics that could not auto-run.

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

    Examples:
        Auto-discover every applicable metric for the cell:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=250)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> cfg = fx.AnalysisConfig.individual_continuous(forward_periods=5)
        >>> bundle = fx.run_metrics(panel, cfg)
        >>> ic_output = bundle["ic"]
        >>> long_frame = bundle.to_frame()
        >>> skipped = dict(bundle.skipped)

        Restrict to a subset by name:

        >>> bundle = fx.run_metrics(panel, cfg, metrics=["ic"])
    """
    missing = {"date", "asset_id", factor_col, "forward_return"} - set(panel.columns)
    if missing:
        hint = (
            "factrix.preprocess.compute_forward_return(panel) attaches forward_return"
            if "forward_return" in missing
            else f"add column(s) {sorted(missing)!r} to the panel"
        )
        _raise_factor_col_error(
            value=factor_col,
            expected=(
                f"panel with canonical columns "
                f"{{date, asset_id, {factor_col!r}, forward_return}}; "
                f"missing {sorted(missing)!r} ({hint})"
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

    if factor_col != "factor":
        if factor_col not in panel.columns:
            _raise_factor_col_error(
                value=factor_col,
                expected=f"a column on panel; got columns {list(panel.columns)!r}",
            )
        if "factor" in panel.columns:
            _raise_factor_col_error(
                value=factor_col,
                expected=(
                    "panel without an existing 'factor' column when "
                    "factor_col != 'factor' (drop the unused column "
                    "before calling)"
                ),
            )
        panel = panel.rename({factor_col: "factor"})

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
    results: dict[str, MetricOutput] = {}
    ic_df: pl.DataFrame | None = None

    import factrix.metrics as _metrics

    for name in targets:
        fn = getattr(_metrics, name)
        stage = "consumer"
        try:
            if name in _IC_CONSUMERS:
                if ic_df is None:
                    stage = "stage1"
                    ic_df = _metrics.compute_ic(panel)
                    stage = "consumer"
                first_arg: pl.DataFrame = ic_df
            else:
                first_arg = panel
            kwargs: dict[str, Any] = {}
            if _accepts_kwarg(fn, "forward_periods"):
                kwargs["forward_periods"] = cfg.forward_periods
            results[name] = fn(first_arg, **kwargs)
        except InsufficientSampleError as exc:
            results[name] = _short_circuit_output(
                name,
                "insufficient_sample",
                actual_periods=exc.actual_periods,
                required_periods=exc.required_periods,
            )
        except RunMetricsError:
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
            "run_metrics(cell=%s, factor_id=%s, fwd=%d): ran=%d skipped=%d (%s)",
            _cell_label(cfg),
            factor_col,
            cfg.forward_periods,
            len(results),
            len(skipped),
            ", ".join(sorted(skipped)),
        )

    return MetricsBundle(
        identity=(factor_col, cfg.forward_periods),
        metrics=MappingProxyType(results),
        skipped=MappingProxyType(skipped),
        context=MappingProxyType({}),
    )
