"""factrix — Single-factor evaluation toolkit.

Two orthogonal user-facing axes — ``FactorScope`` and ``FactorDensity`` —
plus an evaluate-time-derived ``DataStructure`` define the analysis
cell. Resolve metric specs via ``spec_by_name()`` (or register custom ones
with ``metric_spec`` + ``factrix.metrics.register``), dispatch through the
DAG executor via ``evaluate()``, inspect a panel's applicable metrics via
``inspect_panel()``, and aggregate across factors with ``multi_factor.bhy``
for FDR-corrected screening.

Single-factor::

    import factrix as fx

    ic = fx.spec_by_name("ic_newey_west")
    [result] = fx.evaluate(panel, metrics=[ic], factor_cols=["factor"])
    print(result.metrics["ic_newey_west"])

Batch + Benjamini-Hochberg-Yekutieli (BHY)::

    results = fx.evaluate(panel, metrics=[ic], factor_cols=candidate_cols)
    survivors = fx.multi_factor.bhy(results, primary=[ic], q=0.05)

LLM agent reference: ``llms-full.txt`` covers concepts, public API, and
typical usage patterns in a single fetch. Two access paths::

    # Web — deployed at the docs site root
    https://awwesomeman.github.io/factrix/llms-full.txt

    # Local — shipped inside the wheel as package data
    import importlib.resources
    text = importlib.resources.files("factrix").joinpath("llms-full.txt").read_text()
"""

import inspect
from typing import Any

import polars as pl

from factrix import datasets, estimators, multi_factor, preprocess
from factrix._axis import (  # noqa: F401  DataStructure re-exported for namespace access; intentionally not in __all__
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import InfoCode, StatCode, WarningCode
from factrix._compare import compare
from factrix._dag import CycleError, DagExecutor
from factrix._errors import (
    ConfigError,
    FactrixError,
    IncompatibleAxisError,
    InsufficientSampleError,
    UnknownEstimatorError,
    UserInputError,
)
from factrix._inspect import (
    MetricApplicability,
    PanelInspection,
    PanelProperties,
    PanelReasoning,
    _detect_structure,
    inspect_panel,
)
from factrix._metric_index import (
    MetricSpec,
    SampleThreshold,
    list_metrics,
    metric_spec,
    spec_by_name,
)
from factrix._panel_input import PanelInput, _coerce_panel
from factrix._results import EvaluationResult, MetricResult, Warning
from factrix._types import MetricOutput
from factrix.slicing import (
    SliceResult,
    by_slice,
    slice_joint_test,
    slice_pairwise_test,
)
from factrix.stats import list_estimators


def evaluate(
    panel: PanelInput,
    *,
    metrics: list[MetricSpec],
    factor_cols: list[str],
    forward_periods: int | None = None,
) -> list[EvaluationResult]:
    """Evaluate one or more factors against forward returns through the DAG executor.

    Closed-set DAG dispatch — every spec referenced by another spec's
    ``requires`` is auto-pulled into the executor, batched stage-1
    producers run once across the whole factor batch (IC's
    ``compute_ic`` etc.), and per-factor consumers run once per factor.

    Args:
        panel: Long-format panel satisfying the four-column floor
            ``(date, asset_id, <factor_col>, forward_return)``. The
            three fixed-name columns ``(date, asset_id, forward_return)``
            are validated eagerly; the factor-column name is dynamic
            and supplied via ``factor_cols=``. ``forward_return`` must
            already be attached via
            :func:`factrix.preprocess.compute_forward_return`. ``price``
            is **not** required at the baseline — it is an optional
            column consumed by event-study metrics (caar /
            event_around_return / mfe_mae_summary), which short-circuit
            to NaN with a ``reason`` when ``price`` is absent.
        metrics: User-facing :class:`MetricSpec` instances. Element type
            is checked at runtime; ``str`` / ``Callable`` /
            :class:`MetricOutput` are rejected. The first entry is
            tagged as the primary metric (drives ``EvaluationResult.n_obs``
            and the primary / diagnostic partition on
            :class:`MetricResult`); the rest become diagnostics.
        factor_cols: Names of density columns on ``panel``. List-only —
            single ``str`` is rejected. Non-empty, no duplicates, every
            name must exist on ``panel``.
        forward_periods: Forward-return horizon in rows of the panel's
            time axis. ``None`` (default) lets every metric's own
            ``forward_periods=`` default fire; pass an explicit value
            whenever the panel's forward-return horizon is not 5.
            Injected into ``kwargs_by_metric`` only for specs whose
            callable signature accepts the parameter.

    Returns:
        ``list[EvaluationResult]`` in ``factor_cols`` order. Always a
        list — single factor calls return a one-element list. The
        ``cell`` tuple stamped on each result derives from the first
        metric's cell and is informational only; ``DagExecutor`` does
        not consult it for dispatch.

    Raises:
        UserInputError: ``metrics`` not a ``list[MetricSpec]``;
            ``factor_cols`` empty / single ``str`` / contains duplicates
            / references a column not on ``panel``; ``panel`` missing
            any of the baseline columns ``(date, asset_id, forward_return)``;
            ``forward_periods=None`` but a closure metric's callable
            declares a ``forward_periods`` parameter.

    Examples:
        Single-factor IC + IC information ratio (IR):

        >>> import factrix as fx
        >>> from factrix._metric_index import spec_by_name
        >>> raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80)
        >>> panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        >>> specs = spec_by_name()
        >>> results = fx.evaluate(
        ...     panel,
        ...     metrics=[specs["ic"], specs["ic_ir"]],
        ...     factor_cols=["factor"],
        ...     forward_periods=5,
        ... )
        >>> len(results)
        1
        >>> "ic" in results[0].metrics and "ic_ir" in results[0].metrics
        True
    """
    _validate_metrics_arg(metrics)
    cols = _validate_factor_cols_arg(factor_cols)
    panel = _coerce_panel(panel)
    _validate_baseline_columns(panel)
    _validate_factor_cols_on_panel(panel, cols)

    closure_specs = _close_requires(metrics)
    _validate_forward_periods_when_required(closure_specs, forward_periods)
    _validate_primary_metric_applicable(metrics[0], panel)

    primary_cell = metrics[0].cell
    scope = (
        primary_cell.scope if primary_cell.scope is not None else FactorScope.INDIVIDUAL
    )
    density = (
        primary_cell.density
        if primary_cell.density is not None
        else FactorDensity.DENSE
    )
    fp = forward_periods if forward_periods is not None else 5
    kwargs_by_metric = _build_kwargs_by_metric(closure_specs, forward_periods)
    primary_names = (metrics[0].name,)
    executor = DagExecutor(closure_specs, primary_names=primary_names)
    result_dict = executor.execute(
        panel,
        cols,
        scope=scope,
        density=density,
        forward_periods=fp,
        kwargs_by_metric=kwargs_by_metric,
    )
    return [result_dict[c] for c in cols]


_DOCS_METRICS = "api/evaluate#metrics"
_DOCS_FACTOR_COLS = "api/evaluate#factor_cols"
_DOCS_PANEL = "api/evaluate#panel"
_DOCS_FORWARD_PERIODS = "api/evaluate#forward_periods"


def _is_metrics_overview(metrics: object) -> bool:
    """True when ``metrics`` is a ``list_metrics()`` family-grouped overview.

    The overview is ``dict[str, list[MetricSpec]]`` — a catalog, not a
    runnable argument. Recognising its exact shape lets ``evaluate``
    point the user at the right path instead of a generic type error.
    """
    return (
        isinstance(metrics, dict)
        and bool(metrics)
        and all(
            isinstance(v, list) and all(isinstance(s, MetricSpec) for s in v)
            for v in metrics.values()
        )
    )


def _validate_metrics_arg(metrics: object) -> None:
    if _is_metrics_overview(metrics):
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=f"<list_metrics() overview: {len(metrics)} families>",  # type: ignore[arg-type]
            expected=(
                "an explicit list[MetricSpec]. fx.list_metrics() returns an "
                "overview catalog (family -> specs), not a runnable list; "
                "pick the specs you want, or pre-filter the panel with "
                "[m.spec for m in factrix.inspect_panel(panel).usable]"
            ),
            docs_path=_DOCS_METRICS,
        )
    if not isinstance(metrics, list):
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=type(metrics).__name__,
            expected="list[MetricSpec] (non-empty)",
            docs_path=_DOCS_METRICS,
        )
    if not metrics:
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=metrics,
            expected="a non-empty list of MetricSpec",
            docs_path=_DOCS_METRICS,
        )
    for m in metrics:
        if not isinstance(m, MetricSpec):
            raise UserInputError(
                func_name="evaluate",
                field="metrics",
                value=type(m).__name__,
                expected="every element to be a MetricSpec",
                docs_path=_DOCS_METRICS,
            )


def _validate_factor_cols_arg(factor_cols: object) -> list[str]:
    if isinstance(factor_cols, str) or not isinstance(factor_cols, list):
        raise UserInputError(
            func_name="evaluate",
            field="factor_cols",
            value=factor_cols
            if isinstance(factor_cols, str)
            else type(factor_cols).__name__,
            expected="list[str] (single str is rejected)",
            docs_path=_DOCS_FACTOR_COLS,
        )
    if not factor_cols:
        raise UserInputError(
            func_name="evaluate",
            field="factor_cols",
            value=factor_cols,
            expected="a non-empty list of factor column names",
            docs_path=_DOCS_FACTOR_COLS,
        )
    if len(set(factor_cols)) != len(factor_cols):
        raise UserInputError(
            func_name="evaluate",
            field="factor_cols",
            value=factor_cols,
            expected="factor_cols with no duplicates",
            docs_path=_DOCS_FACTOR_COLS,
        )
    for c in factor_cols:
        if not isinstance(c, str):
            raise UserInputError(
                func_name="evaluate",
                field="factor_cols",
                value=type(c).__name__,
                expected="every element to be a str",
                docs_path=_DOCS_FACTOR_COLS,
            )
    return list(factor_cols)


def _validate_factor_cols_on_panel(panel: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise UserInputError(
            func_name="evaluate",
            field="factor_cols",
            value=missing,
            expected=(
                f"every name in factor_cols to exist on panel; "
                f"got columns {list(panel.columns)!r}"
            ),
            docs_path=_DOCS_FACTOR_COLS,
        )


_BASELINE_COLUMNS: tuple[str, ...] = ("date", "asset_id", "forward_return")


def _validate_baseline_columns(panel: pl.DataFrame) -> None:
    missing = [c for c in _BASELINE_COLUMNS if c not in panel.columns]
    if not missing:
        return
    if "forward_return" in missing:
        hint = (
            ". Attach forward_return:\n"
            "    panel = factrix.preprocess.compute_forward_return("
            "panel, forward_periods=<N>)"
        )
    else:
        hint = ""
    raise UserInputError(
        func_name="evaluate",
        field="panel",
        value=list(panel.columns),
        expected=(
            f"panel must include baseline columns {list(_BASELINE_COLUMNS)!r}; "
            f"missing {missing!r}{hint}"
        ),
        docs_path=_DOCS_PANEL,
    )


def _close_requires(metrics: list[MetricSpec]) -> list[MetricSpec]:
    by_name = spec_by_name()
    closed: dict[str, MetricSpec] = {m.name: m for m in metrics}
    stack: list[MetricSpec] = list(metrics)
    while stack:
        s = stack.pop()
        for producer in s.requires.values():
            pname = producer.__name__
            if pname in closed:
                continue
            if pname not in by_name:
                raise UserInputError(
                    func_name="evaluate",
                    field="metrics",
                    value=pname,
                    expected=(
                        f"{s.name!r} requires producer {pname!r} but no "
                        f"MetricSpec for it exists in the registry"
                    ),
                    docs_path=_DOCS_METRICS,
                )
            producer_spec = by_name[pname]
            closed[pname] = producer_spec
            stack.append(producer_spec)
    return list(closed.values())


def _resolve_callable(name: str):
    from factrix._dag import _registry_callable_table

    return _registry_callable_table().get(name)


def _signature_has_forward_periods(name: str) -> bool:
    fn = _resolve_callable(name)
    if fn is None or not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "forward_periods" in sig.parameters


def _validate_forward_periods_when_required(
    closure_specs: list[MetricSpec], forward_periods: int | None
) -> None:
    if forward_periods is not None:
        return
    needing = [s.name for s in closure_specs if _signature_has_forward_periods(s.name)]
    if needing:
        raise UserInputError(
            func_name="evaluate",
            field="forward_periods",
            value=None,
            expected=(
                f"int — required by metric(s) {needing!r} that consume the "
                f"forward-return horizon"
            ),
            docs_path=_DOCS_FORWARD_PERIODS,
        )


def _build_kwargs_by_metric(
    closure_specs: list[MetricSpec], forward_periods: int | None
) -> dict[str, dict[str, Any]]:
    if forward_periods is None:
        return {}
    return {
        s.name: {"forward_periods": forward_periods}
        for s in closure_specs
        if _signature_has_forward_periods(s.name)
    }


def _validate_primary_metric_applicable(
    primary: MetricSpec, panel: pl.DataFrame
) -> None:
    """Reject a primary metric whose ``cell.structure`` disagrees with the panel.

    DataStructure is the cheap pre-flight gate: IC / FM / quantile_spread declare
    ``cell.structure = DataStructure.PANEL`` and produce only NaN short-circuits on a
    single-asset panel; surfacing that as a UserInputError keeps the
    diagnostic specific instead of letting the executor return a result
    bundle whose every metric carries an ``UPSTREAM_UNAVAILABLE`` warning.
    Cell-axis (scope / density) applicability requires panel detection
    and is left to ``fx.inspect_panel`` for the explicit pre-flight path.
    """
    cell_structure = primary.cell.structure
    if cell_structure is None:
        return
    panel_structure = _detect_structure(panel)
    if cell_structure is panel_structure:
        return
    n_assets = int(panel["asset_id"].n_unique())
    raise UserInputError(
        func_name="evaluate",
        field="metrics",
        value=primary.name,
        expected=(
            f"primary metric {primary.name!r} declares "
            f"cell.structure={cell_structure.value!r} but panel has "
            f"structure={panel_structure.value!r} (n_assets={n_assets}); "
            f"call fx.inspect_panel(panel) to see metrics applicable "
            f"to this panel shape"
        ),
        docs_path=_DOCS_METRICS,
    )


__version__ = "0.13.0"

__all__ = [
    # Axis enums (DataStructure intentionally NOT exported — it is
    # evaluate-time-derived from N. It stays importable from
    # factrix._axis for internal callers.)
    "FactorScope",
    "FactorDensity",
    # Code enums
    "InfoCode",
    "StatCode",
    "WarningCode",
    # Errors
    "ConfigError",
    "FactrixError",
    "IncompatibleAxisError",
    "InsufficientSampleError",
    "UnknownEstimatorError",
    "UserInputError",
    # Result + dispatch
    "CycleError",
    "DagExecutor",
    "EvaluationResult",
    "MetricOutput",
    "MetricResult",
    "PanelInput",
    "Warning",
    "compare",
    "evaluate",
    # Introspection
    "MetricApplicability",
    "PanelInspection",
    "PanelProperties",
    "PanelReasoning",
    "SampleThreshold",
    "inspect_panel",
    "list_estimators",
    "list_metrics",
    # Slicing dispatcher + cross-slice inference functions
    "SliceResult",
    "by_slice",
    "slice_joint_test",
    "slice_pairwise_test",
    # Multi-factor namespace
    "multi_factor",
    # Synthetic panels
    "datasets",
    # Forward-return preprocessing
    "preprocess",
    # Estimator entry points (lowercase callables consumed by metric impls)
    "estimators",
    # Metric registration surface
    "MetricSpec",
    "metric_spec",
]
