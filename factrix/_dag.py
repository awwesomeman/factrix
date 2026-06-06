"""DAG executor for ``MetricSpec.requires`` / ``MetricSpec.batchable`` dispatch.

Single execution path that:

- topologically orders a closed set of :class:`MetricSpec` (every
  spec referenced by another spec's ``requires`` must be present),
  raising :class:`CycleError` on a cycle,
- runs ``batchable=True`` producers once across the whole factor batch
  and ``batchable=False`` callables once per factor on a thin
  projection,
- caches every producer's output per ``(spec, factor)`` so any
  number of downstream consumers naming the same callable see one
  computation,
- propagates short-circuit :class:`MetricResult` (NaN value with
  ``metadata["reason"]``) from upstream to downstream consumers
  without invoking the consumer,
- emits a stable, diff-friendly plan string per execute call and
  stamps it onto every :class:`EvaluationResult` returned.

Wired as the user-facing default for :func:`factrix.evaluate`.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import polars as pl

from factrix._axis import DataStructure, FactorDensity, FactorScope, SpecRole
from factrix._codes import WarningCode
from factrix._metric_index import MetricSpec
from factrix._results import (
    EvaluationResult,
    MetricResult,
    MetricResultGroup,
    Warning,
)


def _project_factor(data: pl.DataFrame, col: str) -> pl.DataFrame:
    """Project ``data`` to the canonical 4-column per-factor view.

    Non-batch primitives expect data whose factor column is literally
    named ``"factor"``; this projection renames ``col`` and drops any
    sibling columns so primitives see an identical schema regardless of
    the caller's ``factor_cols`` choice.
    """
    return data.select(
        pl.col("date"),
        pl.col("asset_id"),
        pl.col("forward_return"),
        pl.col(col).alias("factor"),
    )


class CycleError(ValueError):
    """Raised when ``MetricSpec.requires`` declares a dependency cycle."""


@dataclasses.dataclass(frozen=True)
class _Node:
    """A configured execution node — a ``MetricSpec`` at one ``forward_periods`` /
    param configuration.

    By-value dedup keys the executor on ``node_id`` rather than
    ``spec.name`` so the same metric class can run at several configs in one
    call (``{"ic_5d": ic(), "ic_20d": ic(forward_periods=20)}``). ``node_id``
    is ``spec.name`` for the sole config of a name and ``f"{spec.name}#{i}"``
    for additional configs; the per-node kwargs live in ``execute``'s
    ``kwargs_by_metric`` keyed by ``node_id``.

    Attributes:
        spec: The underlying :class:`MetricSpec` (drives callable resolution
            by ``spec.name`` and the batchable / role / requires shape).
        node_id: Stable identity within one executor run.
        requires_nodes: ``(consumer_param, producer_node_id)`` pairs — the
            resolved upstream producer **node** for each ``requires`` key,
            so a consumer at one config reads exactly the producer node
            built for that config.
    """

    spec: MetricSpec
    node_id: str
    requires_nodes: tuple[tuple[str, str], ...]


class _PlanStep(NamedTuple):
    """One topologically-ordered step in an execution plan."""

    node: _Node
    role: str
    requires: tuple[str, ...]


class DagExecutor:
    """Run a closed set of :class:`MetricSpec` against a panel batch.

    Args:
        specs: Closed under ``requires`` — every callable referenced
            by another spec's ``requires`` dict must have its own
            :class:`MetricSpec` in this sequence. Callers needing the
            transitive closure should build it from the registry
            (e.g. ``spec_by_name``) before constructing the executor.
        fn_resolver: Optional override mapping a spec to its callable.
            Defaults to ``getattr(import_module(producer.__module__),
            spec.name)`` via the spec's declaring module. Tests pass an
            explicit resolver to use locally-defined callables without
            building a fake module tree.
        primary_names: Names of specs whose :class:`MetricResult` is
            the bundle's primary p-value driver. Empty by default —
            the primary-vs-diagnostic classification is a cell-level
            policy decision and lives outside this executor.

    Raises:
        CycleError: if ``requires`` introduces a cycle.
        ValueError: if a referenced producer is missing from
            ``specs`` (the executor will not silently auto-close the
            set — the caller declared the dispatch graph and the
            executor refuses to repair it).
    """

    def __init__(
        self,
        specs: Sequence[MetricSpec | _Node],
        *,
        fn_resolver: Callable[[str], Callable[..., Any]] | None = None,
        primary_names: Sequence[str] = (),
    ) -> None:
        import logging

        self._logger = logging.getLogger("factrix.dag")
        # Back-compat: a plain spec is wrapped one node per spec with
        # ``node_id == spec.name`` and requires resolved by producer name;
        # pre-built ``_Node`` items (the by-value path) pass straight through.
        nodes: list[_Node] = []
        for item in specs:
            if isinstance(item, _Node):
                nodes.append(item)
            else:
                nodes.append(
                    _Node(
                        spec=item,
                        node_id=item.name,
                        requires_nodes=tuple(
                            (k, p.__name__) for k, p in item.requires.items()
                        ),
                    )
                )
        self._nodes: tuple[_Node, ...] = tuple(nodes)
        self._fn_resolver = fn_resolver or _default_fn_resolver
        self._fn_cache: dict[str, Callable[..., Any]] = {}
        self._primary_ids = frozenset(primary_names)
        self._plan_steps: list[_PlanStep] = self._build_plan()

    def _fn(self, spec: MetricSpec) -> Callable[..., Any]:
        if spec.name not in self._fn_cache:
            self._fn_cache[spec.name] = self._fn_resolver(spec.name)
        return self._fn_cache[spec.name]

    def _build_plan(self) -> list[_PlanStep]:
        return [
            _PlanStep(
                node=node,
                role="batchable" if node.spec.batchable else "per-factor",
                requires=tuple(pid for _, pid in node.requires_nodes),
            )
            for node in _topo_sort_nodes(self._nodes)
        ]

    @property
    def plan(self) -> str:
        """Multi-line numbered execution plan."""
        lines: list[str] = []
        for idx, step in enumerate(self._plan_steps, start=1):
            parts = [f"{idx}. {step.node.node_id} [{step.role}]"]
            if step.requires:
                parts.append(f"requires={','.join(step.requires)}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def execute(
        self,
        data: pl.DataFrame,
        factor_cols: Sequence[str],
        *,
        scope: FactorScope,
        density: FactorDensity,
        forward_periods: int,
        kwargs_by_metric: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, EvaluationResult]:
        """Run every spec against every factor and return one bundle per factor.

        Producer outputs are cached per ``(spec, factor)``. When
        ``spec.batchable`` is true, the producer is called once with
        ``factor_cols=`` and the per-factor entries of its returned
        dict are unpacked into the cache. When false, the producer is
        called once per factor on a thin projection (only when at
        least one downstream consumer needs it; raw-data consumers
        skip projection).
        """
        self._logger.debug("Executing DAG with topological plan:\n%s", self.plan)

        kwargs_by_metric = kwargs_by_metric or {}
        cols = list(factor_cols)
        n_assets = data.select(pl.col("asset_id").n_unique()).item()
        structure = DataStructure.PANEL if n_assets > 1 else DataStructure.TIMESERIES
        projections: dict[str, pl.DataFrame] = {}

        def project(c: str) -> pl.DataFrame:
            view = projections.get(c)
            if view is None:
                view = _project_factor(data, c)
                projections[c] = view
            return view

        producer_outputs: dict[tuple[str, str], Any] = {}
        metric_outputs: dict[tuple[str, str], MetricResult] = {}

        for node in (step.node for step in self._plan_steps):
            nid = node.node_id
            handle = self._batch_handle(node.spec, kwargs_by_metric.get(nid, {}))

            # Split factors by upstream short-circuit: dead factors get the
            # propagated skip output and never reach the metric.
            live: list[str] = []
            for c in cols:
                skip = _check_upstream_short_circuit(node, c, producer_outputs)
                if skip is not None:
                    self._logger.debug(
                        "Short-circuit propagation: skipping node %s for factor %s because upstream %s failed (%s)",
                        nid,
                        c,
                        skip.metadata.get("upstream"),
                        skip.metadata.get("upstream_reason"),
                    )
                    producer_outputs[(nid, c)] = skip
                    metric_outputs[(nid, c)] = dataclasses.replace(skip, name=nid)
                else:
                    live.append(c)
            if not live:
                self._logger.debug("Node %s skipped entirely (no live factors)", nid)
                continue

            if node.spec.batchable:
                self._logger.debug(
                    "Batched hit: executing batchable node %s across factors %r",
                    nid,
                    live,
                )
            else:
                self._logger.debug(
                    "Executing node %s individually for factors %r", nid, live
                )

            upstream = self._gather_upstream_batch(node, live, producer_outputs)
            result = handle(data, live, project=project, upstream=upstream)
            _validate_batch_result(node, live, result)
            for c in live:
                out = result[c]
                producer_outputs[(nid, c)] = out
                if isinstance(out, MetricResult):
                    metric_outputs[(nid, c)] = dataclasses.replace(out, name=nid)

        return self._assemble(
            cols, scope, density, forward_periods, structure, n_assets, metric_outputs
        )

    def _batch_handle(
        self, spec: MetricSpec, kwargs: Mapping[str, Any]
    ) -> Callable[..., dict[str, Any]]:
        """Return the spec's unified batch dispatcher.

        Registry ``MetricBase`` classes expose ``__call_batch__`` directly
        (bound to a configured instance); bare ``fn_resolver`` callables are
        wrapped through the same :func:`_dispatch_batch` so both paths share
        one dispatch body.
        """
        from factrix.metrics._base import MetricBase, _dispatch_batch

        fn = self._fn(spec)
        kw = dict(kwargs)
        if isinstance(fn, type) and issubclass(fn, MetricBase):
            return fn(**kw).__call_batch__

        bare: Callable[..., Any] = fn

        def handle(
            data: pl.DataFrame,
            factor_cols: Sequence[str],
            *,
            project: Callable[[str], pl.DataFrame],
            upstream: dict[str, dict[str, Any]],
        ) -> dict[str, Any]:
            return _dispatch_batch(
                name=spec.name,
                call_one=lambda *a, **k: bare(*a, **{**kw, **k}),
                run_batch=lambda: bare(
                    data, factor_cols=list(factor_cols), **upstream, **kw
                ),
                batchable=spec.batchable,
                requires=tuple(spec.requires),
                input_shape=spec.input_shape,
                factor_cols=factor_cols,
                project=project,
                upstream=upstream,
            )

        return handle

    def _gather_upstream_batch(
        self,
        node: _Node,
        cols: Sequence[str],
        producer_outputs: Mapping[tuple[str, str], Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, producer_id in node.requires_nodes:
            out[key] = {c: producer_outputs[(producer_id, c)] for c in cols}
        return out

    def _assemble(
        self,
        cols: Sequence[str],
        scope: FactorScope,
        density: FactorDensity,
        forward_periods: int,
        structure: DataStructure,
        n_assets: int,
        metric_outputs: Mapping[tuple[str, str], MetricResult],
    ) -> dict[str, EvaluationResult]:
        public_nodes = [n for n in self._nodes if n.spec.role is SpecRole.METRIC]
        applicable = [n.node_id for n in public_nodes]
        primary = [n.node_id for n in public_nodes if n.node_id in self._primary_ids]
        diagnostic = [
            n.node_id for n in public_nodes if n.node_id not in self._primary_ids
        ]
        plan = self.plan

        results: dict[str, EvaluationResult] = {}
        for c in cols:
            outputs: dict[str, MetricResult] = {}
            warnings: list[Warning] = []
            for node in public_nodes:
                label = node.node_id
                key = (label, c)
                if key not in metric_outputs:
                    continue
                out = metric_outputs[key]
                outputs[label] = out
                reason = out.metadata.get("reason")
                if isinstance(reason, str) and math.isnan(out.value):
                    warnings.append(
                        Warning(
                            code=WarningCode.UPSTREAM_UNAVAILABLE,
                            source=label,
                            message=reason,
                        )
                    )
                # Lift the metric's typed advisory codes into per-source
                # Warning records so to_frame() / to_dict() surface them.
                for code in out.warning_codes:
                    warnings.append(
                        Warning(code=WarningCode(code), source=label, message="")
                    )
            n_obs = _resolve_n_obs(primary, c, metric_outputs)
            results[c] = EvaluationResult(
                factor=c,
                cell=(scope, density, structure),
                forward_periods=forward_periods,
                n_obs=n_obs,
                n_assets=n_assets,
                metrics=MetricResultGroup(
                    applicable=applicable,
                    primary=primary,
                    diagnostic=diagnostic,
                    outputs=outputs,
                ),
                plan=plan,
                warnings=warnings,
            )
        return results


def _default_fn_resolver(name: str) -> Callable[..., Any]:
    table = _registry_callable_table()
    if name not in table:
        raise LookupError(
            f"cannot resolve callable for spec {name!r}; pass "
            f"`fn_resolver=` to DagExecutor for non-registry specs."
        )
    return table[name]


@functools.cache
def _registry_callable_table() -> dict[str, Callable[..., Any]]:
    """Single-pass ``{spec.name: callable}`` across every registered metric.

    Walks first-party ``factrix.metrics.*`` modules and the third-party
    registry populated via :func:`factrix.metrics.register`. Cache is
    cleared by :func:`factrix.metrics.register` so newly-registered
    callables become resolvable on the next executor construction.
    """
    from factrix._metric_index import _METRIC_REGISTRY, _all_specs, import_path_for

    table: dict[str, Callable[..., Any]] = {}
    for stem, spec in _all_specs():
        mod = importlib.import_module(import_path_for(stem))
        fn = getattr(mod, spec.name, None)
        if callable(fn):
            table[spec.name] = fn
    import factrix.metrics as _metrics_pkg

    for name in _METRIC_REGISTRY:
        fn = getattr(_metrics_pkg, name, None)
        if callable(fn):
            table[name] = fn
    return table


def _validate_batch_result(
    node: _Node, factor_cols: Sequence[str], result: Any
) -> None:
    """Check a batch dispatch returned a ``{factor: output}`` mapping.

    Surfaces meaningfully for ``batchable`` specs whose ``_impl`` builds the
    mapping itself; per-factor results are always well-formed dicts.
    """
    if not isinstance(result, Mapping):
        raise TypeError(
            f"{node.node_id}: batchable=True spec must return a "
            f"Mapping[factor, output]; got {type(result).__name__}."
        )
    for c in factor_cols:
        if c not in result:
            raise KeyError(f"{node.node_id}: batchable result missing factor {c!r}")


def _check_upstream_short_circuit(
    node: _Node,
    factor: str,
    producer_outputs: Mapping[tuple[str, str], Any],
) -> MetricResult | None:
    for key, producer_id in node.requires_nodes:
        upstream = producer_outputs.get((producer_id, factor))
        if isinstance(upstream, MetricResult) and math.isnan(upstream.value):
            return MetricResult(
                value=float("nan"),
                metadata={
                    "reason": "upstream_unavailable",
                    "upstream": producer_id,
                    "upstream_reason": upstream.metadata.get("reason", "unknown"),
                    "consumer_param": key,
                },
            )
    return None


def _resolve_n_obs(
    primary: Sequence[str],
    factor: str,
    metric_outputs: Mapping[tuple[str, str], MetricResult],
) -> int:
    for name in primary:
        out = metric_outputs.get((name, factor))
        if out is not None and out.n_obs is not None:
            return out.n_obs
    return 0


def _topo_sort_nodes(nodes: Sequence[_Node]) -> list[_Node]:
    """Kahn topo-order configured nodes by ``node_id`` / ``requires_nodes``.

    The by-value sibling of :func:`_topo_sort` (which orders bare specs by
    name). Raises :class:`CycleError` on a cycle and ``ValueError`` when a
    referenced producer node is absent — the caller must close the node graph
    before constructing the executor.
    """
    by_id = {n.node_id: n for n in nodes}
    indegree: dict[str, int] = {n.node_id: 0 for n in nodes}
    deps: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    for n in nodes:
        for _, producer_id in n.requires_nodes:
            if producer_id not in by_id:
                raise ValueError(
                    f"DagExecutor: {n.node_id!r} requires node {producer_id!r} "
                    f"but that producer node is absent. The caller must close "
                    f"the requires-graph before constructing the executor."
                )
            deps[producer_id].append(n.node_id)
            indegree[n.node_id] += 1

    ready = [n for n in nodes if indegree[n.node_id] == 0]
    ordered: list[_Node] = []
    cursor = 0
    while cursor < len(ready):
        node = ready[cursor]
        cursor += 1
        ordered.append(node)
        for downstream_id in deps[node.node_id]:
            indegree[downstream_id] -= 1
            if indegree[downstream_id] == 0:
                ready.append(by_id[downstream_id])

    if len(ordered) != len(nodes):
        remaining = [n.node_id for n in nodes if n not in ordered]
        raise CycleError(f"DagExecutor: cycle detected among nodes {sorted(remaining)}")
    return ordered


def _topo_sort(specs: Sequence[MetricSpec]) -> list[MetricSpec]:
    by_name = {s.name: s for s in specs}
    indegree: dict[str, int] = {s.name: 0 for s in specs}
    deps: dict[str, list[str]] = {s.name: [] for s in specs}
    for s in specs:
        for producer in s.requires.values():
            pname = producer.__name__
            if pname not in by_name:
                raise ValueError(
                    f"DagExecutor: {s.name!r} requires {pname!r} but that "
                    f"producer has no MetricSpec in the executor's `specs` "
                    f"set. The caller must close the requires-graph before "
                    f"constructing the executor."
                )
            deps[pname].append(s.name)
            indegree[s.name] += 1

    ready = [s for s in specs if indegree[s.name] == 0]
    ordered: list[MetricSpec] = []
    cursor = 0
    while cursor < len(ready):
        node = ready[cursor]
        cursor += 1
        ordered.append(node)
        for downstream_name in deps[node.name]:
            indegree[downstream_name] -= 1
            if indegree[downstream_name] == 0:
                ready.append(by_name[downstream_name])

    if len(ordered) != len(specs):
        remaining = [s.name for s in specs if s not in ordered]
        raise CycleError(f"DagExecutor: cycle detected among specs {sorted(remaining)}")
    return ordered
