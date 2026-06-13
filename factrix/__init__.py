"""factrix — Single-factor evaluation toolkit.

Two orthogonal user-facing axes — ``FactorScope`` and ``FactorDensity`` —
plus an evaluate-time-derived ``DataStructure`` define the analysis
cell. Resolve metric specs via ``spec_by_name()`` (or register custom ones
with ``metric_spec`` + ``factrix.metrics.register``), dispatch through the
DAG executor via ``evaluate()``, inspect a panel's applicable metrics via
``inspect_data()``, and aggregate across factors with ``multi_factor.bhy``
for FDR-corrected screening.

Single-factor::

    import factrix as fx
    from factrix.metrics import ic

    results = fx.evaluate(
        panel,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=["factor"],
    )
    print(results["factor"].metrics["ic"])

Batch + Benjamini-Hochberg-Yekutieli (BHY)::

    results = fx.evaluate(
        panel,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=candidate_cols,
    )
    ic_results = {col: er.metrics["ic"] for col, er in results.items()}
    survivors = fx.multi_factor.bhy(ic_results, primary=["ic"], q=0.05)

LLM agent reference: ``llms-full.txt`` covers concepts, public API, and
typical usage patterns in a single fetch. Two access paths::

    # Web — deployed at the docs site root
    https://awwesomeman.github.io/factrix/llms-full.txt

    # Local — shipped inside the wheel as package data
    import importlib.resources
    text = importlib.resources.files("factrix").joinpath("llms-full.txt").read_text()
"""

import dataclasses
import inspect
import math
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from factrix.metrics._base import MetricBase

from factrix import datasets, estimators, inference, multi_factor, preprocess
from factrix._axis import (  # noqa: F401  DataStructure re-exported for namespace access; intentionally not in __all__
    DataStructure,
    FactorDensity,
    FactorScope,
    Tier,
)
from factrix._codes import InfoCode, WarningCode
from factrix._compare import compare
from factrix._dag import CycleError, DagExecutor, _Node
from factrix._data_input import DataInput, _coerce_data
from factrix._errors import (
    FactrixError,
    IncompatibleAxisError,
    InsufficientSampleError,
    UserInputError,
)
from factrix._inspect import (
    DataInspection,
    DataProperties,
    MetricApplicability,
    MetricApplicabilityGroup,
    _detect_structure,
    inspect_data,
)
from factrix._metric_index import (
    MetricSpec,
    SampleThreshold,
    list_metrics,
    metric_spec,
    spec_by_name,
)
from factrix._results import (
    EvaluationResult,
    MetricResult,
    MetricResultGroup,
    Warning,
)
from factrix.slicing import (
    SliceResult,
    by_slice,
    slice_joint_test,
    slice_pairwise_test,
)


def evaluate(
    data: DataInput,
    *,
    metrics: "dict[str, MetricBase]",
    factor_cols: list[str],
    forward_periods: int | None = None,
    strict: bool = True,
) -> "dict[str, EvaluationResult]":
    """Evaluate one or more factors against forward returns through the DAG executor.

    Closed-set DAG dispatch — every spec referenced by another spec's
    ``requires`` is auto-pulled into the executor, batched stage-1
    producers run once across the whole factor batch (IC's
    ``compute_ic`` etc.), and per-factor consumers run once per factor.

    Args:
        data: Long-format data satisfying the four-column floor
            ``(date, asset_id, <factor_col>, forward_return)``. The
            three fixed-name columns ``(date, asset_id, forward_return)``
            are validated eagerly; ``forward_return`` must already be
            attached via :func:`factrix.preprocess.compute_forward_return`.
            ``price`` is optional — consumed by event-study metrics which
            short-circuit to NaN when it is absent.
        metrics: ``dict[str, Metric]`` mapping a caller-chosen label to a
            metric **instance** from :mod:`factrix.metrics` (e.g.
            ``{"ic_5d": ic(), "spread": quantile_spread(n_groups=5)}``).
            Results key by these labels. Passing the bare class (``ic``
            rather than ``ic()``), a ``str`` or a :class:`MetricSpec` is
            rejected with a targeted error. One metric class may run under
            several labels with **different** config — e.g.
            ``{"ic_5d": ic(), "ic_20d": ic(forward_periods=20)}`` — via
            by-value DAG dedup; shared upstream producers are computed
            once per distinct config.
        factor_cols: Names of factor columns on ``data``. List-only —
            single ``str`` is rejected. Non-empty, no duplicates, every
            name must exist on ``data``.
        forward_periods: Default forward-return horizon (rows of the data's
            time axis) for metrics left at their signature default. A
            per-instance value — ``ic(forward_periods=20)`` — always overrides
            it. ``None`` (default) leaves every metric at its own default.
            Stamped on every :class:`EvaluationResult` as
            ``forward_periods``; per-metric resolved horizons are in
            ``result.metrics[label].metadata["forward_periods"]``.
        strict: When ``True`` (default), raise if any metric is
            inapplicable to the data (apply-time short-circuit to NaN).
            When ``False``, keep those as NaN outputs with attached
            warnings. Config-time / construct-time failures always raise.

    Returns:
        ``dict[str, EvaluationResult]`` keyed by factor column name, in
        ``factor_cols`` insertion order. Each value carries the full
        per-metric outputs, panel structural stats (``n_periods``,
        ``n_pairs``), and warnings for that factor.

    Raises:
        UserInputError: ``metrics`` not a ``dict[str, Metric]`` of
            instances; ``factor_cols`` empty / single ``str`` / contains
            duplicates / references a column not on ``data``; ``data``
            missing a baseline column; a metric ``requires`` a producer
            absent from the registry; under ``strict=True``, a metric
            inapplicable to the data.

    Examples:
        Single-factor IC + IC information ratio (IR):

        >>> import factrix as fx
        >>> from factrix.metrics import ic, ic_ir
        >>> raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80)
        >>> data = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        >>> results = fx.evaluate(
        ...     data,
        ...     metrics={"ic": ic(), "ic_ir": ic_ir()},
        ...     factor_cols=["factor"],
        ...     forward_periods=5,
        ... )
        >>> "ic" in results["factor"].metrics
        True
        >>> "ic_ir" in results["factor"].metrics
        True
    """
    _validate_metrics_arg(metrics)
    cols = _validate_factor_cols_arg(factor_cols)
    data = _coerce_data(data)
    _validate_baseline_columns(data)
    _validate_factor_cols_on_data(data, cols)

    label_spec = {label: type(inst).spec() for label, inst in metrics.items()}
    label_params = {
        label: _resolve_instance_params(inst, forward_periods)
        for label, inst in metrics.items()
    }

    nodes, label_to_node, node_kwargs = _build_nodes(label_spec, label_params)
    node_to_label = {nid: label for label, nid in label_to_node.items()}

    _validate_all_metrics_applicable(label_spec, data, strict)

    fp = forward_periods if forward_periods is not None else 5

    executor = DagExecutor(nodes)
    result_dict = executor.execute(
        data,
        cols,
        scope=FactorScope.INDIVIDUAL,
        density=FactorDensity.DENSE,
        forward_periods=fp,
        kwargs_by_metric=node_kwargs,
    )
    return {
        c: _relabel_result(result_dict[c], label_to_node, node_to_label, strict)
        for c in cols
    }


_DOCS_METRICS = "api/evaluate#metrics"
_DOCS_FACTOR_COLS = "api/evaluate#factor_cols"
_DOCS_DATA = "api/evaluate#data"


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
    from factrix.metrics._base import MetricBase

    if _is_metrics_overview(metrics):
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=f"<list_metrics() overview: {len(metrics)} families>",  # type: ignore[arg-type]
            expected=(
                "a dict[str, Metric] of metric instances. fx.list_metrics() "
                "returns an overview catalog (family -> specs), not runnable "
                "metrics; construct the ones you want, e.g. {'ic': ic()}, or "
                "pre-filter with factrix.inspect_data(data).usable"
            ),
            docs_path=_DOCS_METRICS,
        )
    if not isinstance(metrics, dict):
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=type(metrics).__name__,
            expected="dict[str, Metric] (label -> metric instance), e.g. {'ic_5d': ic()}",
            docs_path=_DOCS_METRICS,
        )
    if not metrics:
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=metrics,
            expected="a non-empty dict[str, Metric]",
            docs_path=_DOCS_METRICS,
        )
    for key, val in metrics.items():
        if not isinstance(key, str):
            raise UserInputError(
                func_name="evaluate",
                field="metrics",
                value=type(key).__name__,
                expected="every key to be a str label",
                docs_path=_DOCS_METRICS,
            )
        if isinstance(val, type) and issubclass(val, MetricBase):
            raise UserInputError(
                func_name="evaluate",
                field="metrics",
                value=f"{key!r} -> {val.__name__} (the class)",
                expected=f"a metric instance, not the class — call it: {val.__name__}()",
                docs_path=_DOCS_METRICS,
            )
        if not isinstance(val, MetricBase):
            raise UserInputError(
                func_name="evaluate",
                field="metrics",
                value=f"{key!r} -> {type(val).__name__}",
                expected=(
                    "every value to be a metric instance imported from "
                    "factrix.metrics, e.g. ic() / quantile_spread(n_quantiles=5)"
                ),
                docs_path=_DOCS_METRICS,
            )

        from factrix._axis import SpecRole

        if val.__class__.spec().role is SpecRole.PIPELINE:
            raise UserInputError(
                func_name="evaluate",
                field="metrics",
                value=f"{key!r} -> {val.__class__.__name__}() (role=PIPELINE)",
                expected=(
                    "a standalone metric. Pipeline producers (like compute_ic) "
                    "cannot be evaluated directly; they are pulled automatically "
                    "when you evaluate a downstream metric via its `requires=` dependency."
                ),
                docs_path=_DOCS_METRICS,
            )


_NO_DEFAULT = object()


def _forward_periods_default(cls: "type[MetricBase]") -> object:
    """The ``forward_periods`` signature default declared by a metric class.

    Read from the wrapped ``_impl`` so the evaluate-level ``forward_periods``
    fallback can tell an instance still at its default apart from an explicit
    per-instance override. Returns a unique sentinel when the metric has no
    ``forward_periods`` parameter.
    """
    try:
        param = inspect.signature(cls._impl).parameters.get("forward_periods")
    except (TypeError, ValueError):
        return _NO_DEFAULT
    if param is None or param.default is inspect.Parameter.empty:
        return _NO_DEFAULT
    return param.default


def _resolve_instance_params(
    inst: "MetricBase", top_forward_periods: int | None
) -> dict[str, Any]:
    """Per-instance params with ``forward_periods`` resolved against the fallback.

    Per-instance override is the rule: an instance's configured
    ``forward_periods`` always wins. The top-level ``evaluate(forward_periods=)``
    is a *default fallback* — it fills only instances still sitting at the
    metric's signature default (and only when the caller passed one). Passing
    the default value explicitly is indistinguishable from not passing it and
    is therefore also treated as "use the fallback".
    """
    params = dict(inst._params())
    if top_forward_periods is None or "forward_periods" not in params:
        return params
    default = _forward_periods_default(type(inst))
    if params["forward_periods"] == default:
        params["forward_periods"] = top_forward_periods
    return params


def _build_nodes(
    label_spec: "dict[str, MetricSpec]",
    label_params: dict[str, dict[str, Any]],
) -> tuple[list[_Node], dict[str, str], dict[str, dict[str, Any]]]:
    """Build the by-value execution node graph for the DAG executor.

    Dedups user metrics into **consumer nodes** keyed by ``(name, config)`` —
    so one class under two labels with *identical* config is a single node
    (harmless alias) while *different* config is two nodes. Then closes ``requires``: each consumer pulls a **producer
    node** whose config inherits only the consumer's ``forward_periods`` (the
    sole consumer param any ``requires``-pulled producer accepts), reusing one
    producer node across consumers that share that inherited config.

    Returns ``(nodes, label -> node_id, node_id -> kwargs)``.
    """
    by_name = spec_by_name()
    node_by_key: dict[tuple[str, frozenset], str] = {}
    node_kwargs: dict[str, dict[str, Any]] = {}
    node_spec: dict[str, MetricSpec] = {}
    name_counts: dict[str, int] = {}
    creation_order: list[str] = []

    def intern(spec: MetricSpec, params: dict[str, Any]) -> str:
        key = (spec.name, frozenset(params.items()))
        nid = node_by_key.get(key)
        if nid is None:
            i = name_counts.get(spec.name, 0)
            nid = spec.name if i == 0 else f"{spec.name}#{i}"
            name_counts[spec.name] = i + 1
            node_by_key[key] = nid
            node_kwargs[nid] = dict(params)
            node_spec[nid] = spec
            creation_order.append(nid)
        return nid

    label_to_node = {
        label: intern(spec, label_params[label]) for label, spec in label_spec.items()
    }

    requires_nodes: dict[str, tuple[tuple[str, str], ...]] = {}
    queue = list(creation_order)
    while queue:
        nid = queue.pop(0)
        if nid in requires_nodes:
            continue
        spec = node_spec[nid]
        consumer_params = node_kwargs[nid]
        edges: list[tuple[str, str]] = []
        for param_key, producer in spec.requires.items():
            pname = producer.__name__
            if pname not in by_name:
                raise UserInputError(
                    func_name="evaluate",
                    field="metrics",
                    value=pname,
                    expected=(
                        f"{spec.name!r} requires producer {pname!r} but no "
                        f"MetricSpec for it exists in the registry"
                    ),
                    docs_path=_DOCS_METRICS,
                )
            accepted = set(getattr(producer, "_param_names", ()))
            inherited = {
                k: consumer_params[k]
                for k in ("forward_periods",)
                if k in consumer_params and k in accepted
            }
            pid = intern(by_name[pname], inherited)
            edges.append((param_key, pid))
            if pid not in requires_nodes:
                queue.append(pid)
        requires_nodes[nid] = tuple(edges)

    nodes = [
        _Node(spec=node_spec[nid], node_id=nid, requires_nodes=requires_nodes[nid])
        for nid in creation_order
    ]
    return nodes, label_to_node, node_kwargs


def _enforce_strict(label_outputs: "dict[str, MetricResult]") -> None:
    """Raise when any requested metric could not produce a value (``strict=True``).

    Apply-time short-circuits surface as a ``MetricResult`` with NaN ``value``
    and a ``metadata['reason']`` (panel missing data / sample / upstream skip).
    Config-time / construct-time failures already raise upstream of ``strict``.
    """
    failed = [
        (label, str(out.metadata.get("reason")))
        for label, out in label_outputs.items()
        if isinstance(out, MetricResult)
        and isinstance(out.value, float)
        and math.isnan(out.value)
        and out.metadata.get("reason")
    ]
    if failed:
        detail = "; ".join(f"{label}: {reason}" for label, reason in failed)
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=f"{len(failed)} metric(s) inapplicable to this panel",
            expected=(
                f"all metrics applicable to the panel. Inapplicable — {detail}. "
                f"Pass strict=False to keep them as NaN with warnings, or "
                f"pre-filter with [m.name for m in factrix.inspect_data(data).usable]."
            ),
            docs_path=_DOCS_METRICS,
        )


def _relabel_result(
    result: EvaluationResult,
    label_to_node: dict[str, str],
    node_to_label: dict[str, str],
    strict: bool,
) -> EvaluationResult:
    """Re-key a node-keyed executor result onto the user's labels."""
    node_outputs = result.metrics.outputs
    label_outputs: dict[str, MetricResult] = {}
    for label, nid in label_to_node.items():
        out = node_outputs.get(nid)
        if out is not None:
            if isinstance(out, MetricResult):
                label_outputs[label] = dataclasses.replace(out, name=label)
            else:
                label_outputs[label] = out
    if strict:
        _enforce_strict(label_outputs)
    warnings = [
        dataclasses.replace(w, source=node_to_label[w.source])
        if w.source is not None and w.source in node_to_label
        else w
        for w in result.warnings
    ]
    return dataclasses.replace(
        result,
        metrics=MetricResultGroup(outputs=label_outputs),
        warnings=warnings,
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


def _validate_factor_cols_on_data(data: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise UserInputError(
            func_name="evaluate",
            field="factor_cols",
            value=missing,
            expected=(
                f"every name in factor_cols to exist on data; "
                f"got columns {list(data.columns)!r}"
            ),
            docs_path=_DOCS_FACTOR_COLS,
        )


_BASELINE_COLUMNS: tuple[str, ...] = ("date", "asset_id", "forward_return")


def _validate_baseline_columns(data: pl.DataFrame) -> None:
    missing = [c for c in _BASELINE_COLUMNS if c not in data.columns]
    if not missing:
        return
    if "forward_return" in missing:
        hint = (
            ". Attach forward_return:\n"
            "    data = factrix.preprocess.compute_forward_return("
            "data, forward_periods=<N>)"
        )
    else:
        hint = ""
    raise UserInputError(
        func_name="evaluate",
        field="data",
        value=list(data.columns),
        expected=(
            f"data must include baseline columns {list(_BASELINE_COLUMNS)!r}; "
            f"missing {missing!r}{hint}"
        ),
        docs_path=_DOCS_DATA,
    )


def _validate_all_metrics_applicable(
    label_spec: "dict[str, MetricSpec]", data: pl.DataFrame, strict: bool
) -> None:
    """Reject any metric whose declared ``cell.structure`` disagrees with the data.

    DataStructure is the cheap pre-flight gate: panel metrics (IC / FM /
    quantile_spread) produce only NaN short-circuits on single-asset data;
    surfacing that eagerly keeps the diagnostic specific.

    Under ``strict=False`` this guard is skipped and structure mismatches
    flow through to NaN outputs + warnings.
    """
    if not strict:
        return
    data_structure = _detect_structure(data)
    n_assets = int(data["asset_id"].n_unique())
    for label, spec in label_spec.items():
        cell_structure = spec.cell.structure
        if cell_structure is None or cell_structure is data_structure:
            continue
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=label,
            expected=(
                f"metric {label!r} declares "
                f"cell.structure={cell_structure.value!r} but data has "
                f"structure={data_structure.value!r} (n_assets={n_assets}); "
                f"call fx.inspect_data(data) to see metrics applicable "
                f"to this data shape"
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
    "Tier",
    # Code enums
    "InfoCode",
    "WarningCode",
    # Errors
    "FactrixError",
    "IncompatibleAxisError",
    "InsufficientSampleError",
    "UserInputError",
    # Result + dispatch
    "CycleError",
    "DagExecutor",
    "EvaluationResult",
    "MetricResult",
    "MetricResultGroup",
    "DataInput",
    "Warning",
    "compare",
    "evaluate",
    # Introspection
    "MetricApplicability",
    "MetricApplicabilityGroup",
    "DataInspection",
    "DataProperties",
    "SampleThreshold",
    "inspect_data",
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
    # Curated statistical inference methods (e.g. ic(inference=...))
    "inference",
    # Metric registration surface
    "MetricSpec",
    "metric_spec",
]
