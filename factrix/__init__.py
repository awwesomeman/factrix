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
    survivors = fx.multi_factor.bhy(list(results.values()), metrics=["ic"], q=0.05)

LLM agent reference: ``llms-full.txt`` covers concepts, public API, and
typical usage patterns in a single fetch. Two access paths::

    # Web — deployed at the docs site root
    https://awwesomeman.github.io/factrix/llms-full.txt

    # Local — shipped inside the wheel as package data
    import importlib.resources
    text = importlib.resources.files("factrix").joinpath("llms-full.txt").read_text()
"""

import dataclasses
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from factrix.metrics._base import MetricBase

from factrix import datasets, inference, multi_factor, preprocess
from factrix._axis import (  # DataStructure used by the structure pre-flight; re-exported for namespace access, intentionally not in __all__
    DataStructure,
    FactorDensity,
    FactorScope,
    Tier,
)
from factrix._codes import WarningCode
from factrix._compare import compare
from factrix._dag import CycleError, DagExecutor, _Node
from factrix._data_input import (
    _BASELINE_COLUMNS,
    _FORWARD_PERIODS_COL,
    DataInput,
    _coerce_data,
    _read_forward_periods_stamp,
)
from factrix._errors import (
    FactrixError,
    IncompatibleAxisError,
    IncompatibleInferenceError,
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
    metrics_summary,
    spec_by_name,
)
from factrix._results import (
    EvaluationResult,
    MetricResult,
    Warning,
)
from factrix.slicing import (
    by_slice,
    slice_joint_test,
    slice_pairwise_test,
    slice_period_joint_test,
    slice_period_pairwise_test,
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
            ``{"spread5": quantile_spread(n_groups=5), "spread10":
            quantile_spread(n_groups=10)}`` — via by-value DAG dedup; shared
            upstream producers are computed once per distinct config.
            ``forward_periods`` is **not** a metric knob: every metric runs at
            the data's single overlap horizon. To compare horizons, build two
            panels and evaluate each.
        factor_cols: Names of factor columns on ``data``. List-only —
            single ``str`` is rejected. Non-empty, no duplicates, every
            name must exist on ``data``.
        forward_periods: The data's overlap horizon (rows of the time axis).
            Normally omitted — :func:`factrix.preprocess.compute_forward_return`
            stamps it on the panel and it is read from there. Pass it only to
            **declare** the horizon for a self-attached ``forward_return``
            column that carries no stamp; a value disagreeing with the stamp is
            rejected. Stamped on every :class:`EvaluationResult` as
            ``forward_periods`` and injected into each metric (surfaced in
            ``result.metrics[label].metadata["forward_periods"]``).
        strict: When ``True`` (default), raise if a metric that *fits* the
            data could not produce a value — a data shortage
            (``insufficient_*``), a missing input / config (``no_*``), or a
            structure mismatch (a metric whose ``cell.structure`` disagrees
            with the data, e.g. a PANEL metric on TIMESERIES data). A
            ``not_applicable*`` type-routing verdict (the metric's *type* does
            not fit this factor, e.g. a continuous-magnitude metric on a
            discrete ±k signal) is **not** a strict failure even under
            ``True``: it surfaces as a NaN ``MetricResult`` with
            ``is_applicable=False`` and ``metadata["reason"]`` while the
            applicable metrics in the same call still return — so a mixed
            battery is not aborted by one inapplicable metric. When
            ``strict=False``, *every* such case (including data shortages and
            structure mismatches) is kept as a NaN output with attached
            warnings instead of raising. Config-time / construct-time
            failures always raise.

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
            inapplicable to the data; ``data`` carries no horizon stamp and
            none is declared via ``forward_periods``, or the declared
            ``forward_periods`` disagrees with the stamp.

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
    _validate_factor_cols_numeric(data, cols)

    # The overlap horizon is a property of the data: read the stamp left by
    # compute_forward_return, then strip it so it never reaches a metric,
    # projection, or to_frame. A self-attached forward_return panel (no stamp)
    # must declare its horizon once via forward_periods= (path B).
    fp = _resolve_forward_periods(data, forward_periods)
    if _FORWARD_PERIODS_COL in data.columns:
        data = data.drop(_FORWARD_PERIODS_COL)

    label_spec = {
        label: dataclasses.replace(
            type(inst).spec(),
            sample_threshold=type(inst)._resolve_sample_threshold(inst),
        )
        for label, inst in metrics.items()
    }
    label_params = {label: dict(inst._params()) for label, inst in metrics.items()}

    # Structure pre-flight: a metric whose declared cell.structure disagrees
    # with the data structure (e.g. a PANEL metric on TIMESERIES data) can
    # only produce a structurally invalid result. Under strict=True this
    # raises; under strict=False the metric is *not executed* — it is dropped
    # from the DAG and surfaces as a NaN short-circuit + warning.
    mismatches = _structure_mismatches(label_spec, data)
    if strict and mismatches:
        _raise_structure_mismatch(mismatches)
    runnable_spec = {
        label: spec for label, spec in label_spec.items() if label not in mismatches
    }
    runnable_params = {label: label_params[label] for label in runnable_spec}

    nodes, label_to_node, node_kwargs = _build_nodes(runnable_spec, runnable_params)
    node_to_label = {nid: label for label, nid in label_to_node.items()}

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
        c: _relabel_result(
            result_dict[c],
            label_to_node,
            node_to_label,
            strict,
            ordered_labels=list(label_spec),
            mismatches=mismatches,
        )
        for c in cols
    }


def evaluate_horizons(
    data: DataInput,
    *,
    metrics: "dict[str, MetricBase]",
    factor_cols: list[str],
    forward_periods: list[int],
    strict: bool = True,
) -> list[EvaluationResult]:
    """Sweep ``evaluate`` across several overlap horizons of one raw panel.

    A thin composition over the existing primitives — for each horizon it
    rebuilds the panel with
    :func:`factrix.preprocess.compute_forward_return` and runs a single
    :func:`evaluate`, then flattens the per-factor results into one list.
    No new type is introduced and the single-horizon contract of
    ``evaluate`` is untouched: every inner run still evaluates one panel at
    one stamped horizon.

    The horizon **must** be rebuilt from the raw panel for each value —
    ``compute_forward_return`` is not idempotent (it drops the last
    ``forward_periods + 1`` rows per asset and stamps the horizon), so a
    horizon cannot be re-derived from an already-attached panel. This
    wrapper exists to make that rebuild-per-horizon loop hard to get wrong.

    Identity of a swept result is the composite ``(factor, forward_periods)``,
    not a unique scalar factor key — so the return is a flat
    ``list[EvaluationResult]`` (the native shape of the aggregation layer),
    not the factor-keyed ``dict`` that ``evaluate`` returns at a fixed
    horizon. ``factor`` and ``forward_periods`` are existing native
    attributes of :class:`EvaluationResult`; the list feeds straight into
    :func:`compare` and into
    ``bhy(..., expand_over=('forward_periods',))``, which partitions each
    horizon into its own step-up family.

    Args:
        data: A **raw** panel carrying ``date``, ``asset_id``, ``price`` and
            the factor columns, **without** a ``forward_return`` column — it
            is rebuilt per horizon. Passing an already-attached panel is
            rejected by ``compute_forward_return``. Only forward-return is
            computed; winsorize / abnormal-return are out of scope, build
            those panels and call :func:`evaluate` per horizon by hand.
        metrics: Same contract as :func:`evaluate` — a ``dict[str, Metric]``
            of metric instances. Applied identically at every horizon.
        factor_cols: Same contract as :func:`evaluate`. Every column is
            evaluated at every horizon; the flat result has one entry per
            ``(factor, horizon)``.
        forward_periods: The horizons to sweep, as a non-empty ``list[int]``
            of distinct positive row counts (e.g. ``[5, 20, 60]``).
            Duplicates are rejected — they would yield a duplicate
            ``(factor, forward_periods)`` identity that ``compare`` / ``bhy``
            reject downstream.
        strict: Forwarded unchanged to each inner :func:`evaluate`.

    Returns:
        Flat ``list[EvaluationResult]`` grouped by horizon (outer) then
        ``factor_cols`` order (inner). Each entry carries its own stamped
        ``forward_periods``.

    Raises:
        UserInputError: ``forward_periods`` is not a non-empty ``list[int]``
            of distinct positive values; plus any error raised by the inner
            :func:`evaluate` / ``compute_forward_return`` (e.g. ``data``
            already carries ``forward_return``).

    Notes:
        Comparability across horizons is a *scale* alignment, not a free
        lunch: ``compute_forward_return`` divides by ``N`` so rank-IC is
        directly comparable across horizons, but signed-return-mean metrics
        carry a compounding bias that grows with ``N`` (see
        :func:`factrix.preprocess.compute_forward_return` Notes). Treat a
        cross-horizon sweep of signed-mean metrics as descriptive.

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics import ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=300)
        >>> results = fx.evaluate_horizons(
        ...     raw,
        ...     metrics={"ic": ic()},
        ...     factor_cols=["factor"],
        ...     forward_periods=[5, 10, 20],
        ... )
        >>> [r.forward_periods for r in results]
        [5, 10, 20]
        >>> board = fx.compare(results, metrics=["ic"])  # one row per horizon
        >>> board.height
        3
    """
    horizons = _validate_forward_periods_sweep(forward_periods)
    raw = _coerce_data(data)
    results: list[EvaluationResult] = []
    for horizon in horizons:
        panel = preprocess.compute_forward_return(raw, forward_periods=horizon)
        per_factor = evaluate(
            panel,
            metrics=metrics,
            factor_cols=factor_cols,
            strict=strict,
        )
        results.extend(per_factor.values())
    return results


_DOCS_FORWARD_PERIODS_SWEEP = "api/multi-horizon"


def _validate_forward_periods_sweep(forward_periods: object) -> list[int]:
    """Validate the ``evaluate_horizons`` horizon list and return it.

    Non-empty ``list[int]`` of distinct positive values. Duplicates fail
    here rather than surfacing later as a duplicate
    ``(factor, forward_periods)`` identity in ``compare`` / ``bhy``.
    """
    if not isinstance(forward_periods, list) or not forward_periods:
        raise UserInputError(
            func_name="evaluate_horizons",
            field="forward_periods",
            value=forward_periods,
            expected="a non-empty list[int] of horizons, e.g. [5, 20, 60]",
            docs_path=_DOCS_FORWARD_PERIODS_SWEEP,
        )
    for h in forward_periods:
        if not isinstance(h, int) or h <= 0:
            raise UserInputError(
                func_name="evaluate_horizons",
                field="forward_periods",
                value=h,
                expected="every horizon to be a positive int (row count)",
                docs_path=_DOCS_FORWARD_PERIODS_SWEEP,
            )
    if len(set(forward_periods)) != len(forward_periods):
        raise UserInputError(
            func_name="evaluate_horizons",
            field="forward_periods",
            value=forward_periods,
            expected=(
                "distinct horizons — a repeated value yields a duplicate "
                "(factor, forward_periods) identity that compare/bhy reject"
            ),
            docs_path=_DOCS_FORWARD_PERIODS_SWEEP,
        )
    return list(forward_periods)


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
                    "factrix.metrics, e.g. ic() / quantile_spread(n_groups=5)"
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


def _resolve_forward_periods(data: pl.DataFrame, declared: int | None) -> int:
    """Resolve the panel's single overlap horizon for this evaluation.

    Path A (primary): a panel built by ``compute_forward_return`` carries a
    horizon stamp — the single source of truth. Path B (escape hatch): a
    self-attached ``forward_return`` panel carries no stamp, so the caller must
    declare the horizon once via ``forward_periods=`` (a statement about the
    data's overlap, not a per-metric knob). A declaration that disagrees with
    the stamp is rejected rather than silently resolved.
    """
    stamp = _read_forward_periods_stamp(data)
    if stamp is not None:
        if declared is not None and declared != stamp:
            raise UserInputError(
                func_name="evaluate",
                field="forward_periods",
                value=declared,
                expected=(
                    f"forward_periods to match the data's stamped overlap "
                    f"horizon ({stamp}, set by compute_forward_return). The "
                    f"horizon is a property of the data — omit forward_periods, "
                    f"or rebuild forward_return at horizon {declared}."
                ),
                docs_path="api/evaluate#forward_periods",
            )
        return stamp
    if declared is not None:
        return declared
    raise UserInputError(
        func_name="evaluate",
        field="forward_periods",
        value=None,
        expected=(
            "the data's overlap horizon. Either build forward_return via "
            "factrix.preprocess.compute_forward_return(data, forward_periods=N) "
            "(which stamps the horizon), or, for a self-attached forward_return "
            "column, declare it once with evaluate(..., forward_periods=N)."
        ),
        docs_path="api/evaluate#forward_periods",
    )


def _build_nodes(
    label_spec: "dict[str, MetricSpec]",
    label_params: dict[str, dict[str, Any]],
) -> tuple[list[_Node], dict[str, str], dict[str, dict[str, Any]]]:
    """Build the by-value execution node graph for the DAG executor.

    Dedups user metrics into **consumer nodes** keyed by ``(name, config)`` —
    so one class under two labels with *identical* config is a single node
    (harmless alias) while *different* config is two nodes. Then closes
    ``requires``: each consumer pulls a **producer node** carrying no inherited
    config (the overlap horizon, once the sole inherited param, is now injected
    globally from the data stamp at dispatch), so one producer node is reused
    across every consumer that names it.

    Returns ``(nodes, label -> node_id, node_id -> kwargs)``.
    """
    by_name = spec_by_name()
    node_by_key: dict[tuple[str, frozenset], str] = {}
    node_kwargs: dict[str, dict[str, Any]] = {}
    node_spec: dict[str, MetricSpec] = {}
    name_counts: dict[str, int] = {}
    creation_order: list[str] = []

    def intern(spec: MetricSpec, params: dict[str, Any]) -> str:
        # List params (e.g. ``offsets=[...]``) are unhashable; coerce to a
        # tuple so the dedup key works. Only the key is affected — the stored
        # kwargs below keep the original list the metric signature expects.
        key = (
            spec.name,
            frozenset(
                (k, tuple(v) if isinstance(v, list) else v) for k, v in params.items()
            ),
        )
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
            # Producers carry no inherited consumer config: the sole parameter a
            # requires-pulled producer ever shared was ``forward_periods``, now
            # injected globally from the data stamp at dispatch. One producer
            # node is reused across all consumers naming it.
            pid = intern(by_name[pname], {})
            edges.append((param_key, pid))
            if pid not in requires_nodes:
                queue.append(pid)
        requires_nodes[nid] = tuple(edges)

    nodes = [
        _Node(spec=node_spec[nid], node_id=nid, requires_nodes=requires_nodes[nid])
        for nid in creation_order
    ]
    return nodes, label_to_node, node_kwargs


def _is_type_routing_reason(reason: object) -> bool:
    """True for a ``not_applicable*`` short-circuit reason.

    A ``not_applicable*`` outcome means the metric's *type* does not fit this
    factor (e.g. a continuous-magnitude metric on a discrete ±k signal) — the
    type-routing verdict, not a failure. ``strict=True`` does not hard-fail on
    it: in a mixed battery an inapplicable metric is expected, and aborting
    would discard the applicable results too. Data shortages (``insufficient_*``)
    and missing input / config (``no_*``) remain strict failures — there the
    metric *fits* but the data or call is deficient, which is worth failing loud.
    """
    return isinstance(reason, str) and reason.startswith("not_applicable")


def _enforce_strict(label_outputs: "dict[str, MetricResult]") -> None:
    """Raise when a requested metric that *fits* the data could not produce a value.

    Apply-time short-circuits surface as a ``MetricResult`` with NaN ``value``
    and a ``metadata['reason']``. Data-shortage / missing-input reasons raise
    under ``strict=True``; ``not_applicable*`` type-routing verdicts do not
    (see :func:`_is_type_routing_reason`). Config-time / construct-time failures
    already raise upstream of ``strict``.
    """
    failed = [
        (label, str(out.metadata.get("reason")))
        for label, out in label_outputs.items()
        if math.isnan(out.value)
        and out.metadata.get("reason")
        and not _is_type_routing_reason(out.metadata.get("reason"))
    ]
    if failed:
        detail = "; ".join(f"{label}: {reason}" for label, reason in failed)
        raise UserInputError(
            func_name="evaluate",
            field="metrics",
            value=f"{len(failed)} metric(s) inapplicable to this panel",
            expected=(
                f"all metrics applicable to the panel. Inapplicable — {detail}. "
                f"For sensitivity grids, pass strict=False and stack "
                f"EvaluationResult.to_frame() to inspect is_applicable/reason. Or "
                f"pre-filter with [m.name for m in factrix.inspect_data(data).usable]."
            ),
            docs_path=_DOCS_METRICS,
        )


def _relabel_result(
    result: EvaluationResult,
    label_to_node: dict[str, str],
    node_to_label: dict[str, str],
    strict: bool,
    *,
    ordered_labels: list[str],
    mismatches: "dict[str, tuple[DataStructure, DataStructure, int]]",
) -> EvaluationResult:
    """Re-key a node-keyed executor result onto the user's labels.

    Structure-mismatched labels never reached the executor; their NaN
    short-circuit ``MetricResult`` is synthesized here so the returned dict
    still carries every requested label, in the caller's request order.
    """
    node_outputs = result.metrics
    label_outputs: dict[str, MetricResult] = {}
    mismatch_warnings: list[Warning] = []
    for label in ordered_labels:
        if label in mismatches:
            cell_structure, data_structure, _ = mismatches[label]
            label_outputs[label] = _structure_mismatch_output(
                label, cell_structure, data_structure
            )
            mismatch_warnings.append(
                Warning(code=WarningCode.STRUCTURE_MISMATCH, source=label)
            )
            continue
        out = node_outputs.get(label_to_node[label])
        if out is not None:
            label_outputs[label] = dataclasses.replace(out, name=label)
    if strict:
        _enforce_strict(label_outputs)
    warnings = [
        dataclasses.replace(w, source=node_to_label[w.source])
        if w.source is not None and w.source in node_to_label
        else w
        for w in result.warnings
    ]
    warnings.extend(mismatch_warnings)
    return dataclasses.replace(
        result,
        metrics=MappingProxyType(label_outputs),
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


def _validate_factor_cols_numeric(data: pl.DataFrame, cols: list[str]) -> None:
    non_numeric = [(c, data.schema[c]) for c in cols if not data.schema[c].is_numeric()]
    if not non_numeric:
        return
    col, dtype = non_numeric[0]
    raise UserInputError(
        func_name="evaluate",
        field="factor_cols",
        value=f"{col!r} has dtype {dtype}",
        expected=(
            "factor columns to be numeric. Encode categorical/string signals "
            "before evaluate(), or pass a numeric exposure / event indicator."
        ),
        docs_path=_DOCS_FACTOR_COLS,
    )


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


def _structure_mismatches(
    label_spec: "dict[str, MetricSpec]", data: pl.DataFrame
) -> "dict[str, tuple[DataStructure, DataStructure, int]]":
    """Map each structure-mismatched label to ``(cell, data, n_assets)``.

    DataStructure is the cheap pre-flight gate: panel metrics (IC / FM /
    quantile_spread) produce only structurally invalid output on
    single-asset data. A metric with ``cell.structure is None`` is
    structure-agnostic and never mismatches.
    """
    data_structure = _detect_structure(data)
    n_assets = int(data["asset_id"].n_unique())
    return {
        label: (spec.cell.structure, data_structure, n_assets)
        for label, spec in label_spec.items()
        if spec.cell.structure is not None and spec.cell.structure is not data_structure
    }


def _raise_structure_mismatch(
    mismatches: "dict[str, tuple[DataStructure, DataStructure, int]]",
) -> None:
    """Raise the ``strict=True`` structure-mismatch error (first offender)."""
    label, (cell_structure, data_structure, n_assets) = next(iter(mismatches.items()))
    raise IncompatibleAxisError(
        f"evaluate(): metric {label!r} declares "
        f"cell.structure={cell_structure.value!r} but data has "
        f"structure={data_structure.value!r} (n_assets={n_assets}); "
        f"call fx.inspect_data(data) to see metrics applicable "
        f"to this data shape, or pass strict=False to keep mismatched "
        f"metrics as NaN with warnings"
    )


def _structure_mismatch_output(
    label: str,
    cell_structure: "DataStructure",
    data_structure: "DataStructure",
) -> MetricResult:
    """Canonical NaN short-circuit for a structure-mismatched metric.

    Marked ``descriptive`` (``p_value=None``) because no test was run — the
    metric never executed, so it carries no statistic to report and is
    excluded from downstream BHY rather than counted as a failed test.
    """
    from factrix.metrics._helpers import _short_circuit_output

    return _short_circuit_output(
        label,
        WarningCode.STRUCTURE_MISMATCH.value,
        descriptive=True,
        cell_structure=cell_structure.value,
        data_structure=data_structure.value,
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
    "WarningCode",
    # Errors
    "FactrixError",
    "IncompatibleAxisError",
    "IncompatibleInferenceError",
    "InsufficientSampleError",
    "UserInputError",
    # Result + dispatch
    "CycleError",
    "DagExecutor",
    "EvaluationResult",
    "MetricResult",
    "DataInput",
    "Warning",
    "compare",
    "evaluate",
    "evaluate_horizons",
    # Introspection
    "MetricApplicability",
    "MetricApplicabilityGroup",
    "DataInspection",
    "DataProperties",
    "SampleThreshold",
    "inspect_data",
    "list_metrics",
    "metrics_summary",
    # Slicing dispatcher + cross-slice inference functions
    "by_slice",
    "slice_joint_test",
    "slice_pairwise_test",
    "slice_period_joint_test",
    "slice_period_pairwise_test",
    # Multi-factor namespace
    "multi_factor",
    # Synthetic panels
    "datasets",
    # Forward-return preprocessing
    "preprocess",
    # Curated statistical inference methods (e.g. ic(inference=...))
    "inference",
    # Metric registration surface
    "MetricSpec",
    "metric_spec",
]
