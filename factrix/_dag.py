"""DAG executor for ``MetricSpec.requires`` / ``MetricSpec.batchable`` dispatch (#442).

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
- propagates short-circuit :class:`MetricOutput` (NaN value with
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
from factrix._results import EvaluationResult, MetricResult, Warning
from factrix._types import MetricOutput


def _project_factor(panel: pl.DataFrame, col: str) -> pl.DataFrame:
    """Project ``panel`` to the canonical 4-column per-factor view.

    Non-batch primitives expect a panel whose factor column is literally
    named ``"factor"``; this projection renames ``col`` and drops any
    sibling columns so primitives see an identical schema regardless of
    the caller's ``factor_cols`` choice.
    """
    return panel.select(
        pl.col("date"),
        pl.col("asset_id"),
        pl.col("forward_return"),
        pl.col(col).alias("factor"),
    )


class CycleError(ValueError):
    """Raised when ``MetricSpec.requires`` declares a dependency cycle."""


class _PlanStep(NamedTuple):
    """One topologically-ordered step in an execution plan."""

    spec: MetricSpec
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
        primary_names: Names of specs whose :class:`MetricOutput` is
            the bundle's primary p-value driver. Empty by default —
            the primary-vs-diagnostic classification is a cell-level
            policy decision and lives outside this executor (#438).

    Raises:
        CycleError: if ``requires`` introduces a cycle.
        ValueError: if a referenced producer is missing from
            ``specs`` (the executor will not silently auto-close the
            set — the caller declared the dispatch graph and the
            executor refuses to repair it).
    """

    def __init__(
        self,
        specs: Sequence[MetricSpec],
        *,
        fn_resolver: Callable[[str], Callable[..., Any]] | None = None,
        primary_names: Sequence[str] = (),
    ) -> None:
        self._specs: tuple[MetricSpec, ...] = tuple(specs)
        self._fn_resolver = fn_resolver or _default_fn_resolver
        self._fn_cache: dict[str, Callable[..., Any]] = {}
        self._primary_names = frozenset(primary_names)
        self._plan_steps: list[_PlanStep] = self._build_plan()

    def _fn(self, spec: MetricSpec) -> Callable[..., Any]:
        if spec.name not in self._fn_cache:
            self._fn_cache[spec.name] = self._fn_resolver(spec.name)
        return self._fn_cache[spec.name]

    def _build_plan(self) -> list[_PlanStep]:
        return [
            _PlanStep(
                spec=spec,
                role="batchable" if spec.batchable else "per-factor",
                requires=tuple(p.__name__ for p in spec.requires.values()),
            )
            for spec in _topo_sort(self._specs)
        ]

    @property
    def plan(self) -> str:
        """Multi-line numbered execution plan."""
        lines: list[str] = []
        for idx, step in enumerate(self._plan_steps, start=1):
            parts = [f"{idx}. {step.spec.name} [{step.role}]"]
            if step.requires:
                parts.append(f"requires={','.join(step.requires)}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def execute(
        self,
        panel: pl.DataFrame,
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
        least one downstream consumer needs it; raw-panel consumers
        skip projection).
        """
        kwargs_by_metric = kwargs_by_metric or {}
        cols = list(factor_cols)
        n_assets = panel.select(pl.col("asset_id").n_unique()).item()
        structure = DataStructure.PANEL if n_assets > 1 else DataStructure.TIMESERIES
        projections: dict[str, pl.DataFrame] = {}

        producer_outputs: dict[tuple[str, str], Any] = {}
        metric_outputs: dict[tuple[str, str], MetricOutput] = {}

        for spec in (step.spec for step in self._plan_steps):
            fn = self._fn(spec)
            kwargs = dict(kwargs_by_metric.get(spec.name, {}))
            if spec.batchable:
                upstream_batch = self._gather_upstream_batch(
                    spec, cols, producer_outputs
                )
                result = fn(panel, factor_cols=cols, **upstream_batch, **kwargs)
                if not isinstance(result, Mapping):
                    raise TypeError(
                        f"{spec.name}: batchable=True spec must return a "
                        f"Mapping[factor, output]; got {type(result).__name__}."
                    )
                for c in cols:
                    if c not in result:
                        raise KeyError(
                            f"{spec.name}: batchable result missing factor {c!r}"
                        )
                    producer_outputs[(spec.name, c)] = result[c]
                    if isinstance(result[c], MetricOutput):
                        metric_outputs[(spec.name, c)] = _stamp_spec(result[c], spec)
                continue

            for c in cols:
                short_circuit = _check_upstream_short_circuit(spec, c, producer_outputs)
                if short_circuit is not None:
                    out = short_circuit
                else:
                    upstream_kwargs = {
                        key: producer_outputs[(producer.__name__, c)]
                        for key, producer in spec.requires.items()
                    }
                    if spec.requires:
                        out = fn(**upstream_kwargs, **kwargs)
                    else:
                        view = projections.get(c)
                        if view is None:
                            view = _project_factor(panel, c)
                            projections[c] = view
                        out = fn(view, **kwargs)
                producer_outputs[(spec.name, c)] = out
                if isinstance(out, MetricOutput):
                    metric_outputs[(spec.name, c)] = _stamp_spec(out, spec)

        return self._assemble(
            cols, scope, density, forward_periods, structure, n_assets, metric_outputs
        )

    def _gather_upstream_batch(
        self,
        spec: MetricSpec,
        cols: Sequence[str],
        producer_outputs: Mapping[tuple[str, str], Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, producer in spec.requires.items():
            pname = producer.__name__
            out[key] = {c: producer_outputs[(pname, c)] for c in cols}
        return out

    def _assemble(
        self,
        cols: Sequence[str],
        scope: FactorScope,
        density: FactorDensity,
        forward_periods: int,
        structure: DataStructure,
        n_assets: int,
        metric_outputs: Mapping[tuple[str, str], MetricOutput],
    ) -> dict[str, EvaluationResult]:
        public_specs = [s for s in self._specs if s.role is SpecRole.METRIC]
        primary = [s for s in public_specs if s.name in self._primary_names]
        diagnostic = [s for s in public_specs if s.name not in self._primary_names]
        plan = self.plan

        results: dict[str, EvaluationResult] = {}
        for c in cols:
            outputs: dict[str, MetricOutput] = {}
            warnings: list[Warning] = []
            for spec in public_specs:
                key = (spec.name, c)
                if key not in metric_outputs:
                    continue
                out = metric_outputs[key]
                label = spec.name
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
            n_obs = _resolve_n_obs(primary, c, metric_outputs)
            results[c] = EvaluationResult(
                factor=c,
                cell=(scope, density, structure),
                forward_periods=forward_periods,
                n_obs=n_obs,
                n_assets=n_assets,
                metrics=MetricResult(
                    applicable=public_specs,
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


def _check_upstream_short_circuit(
    spec: MetricSpec,
    factor: str,
    producer_outputs: Mapping[tuple[str, str], Any],
) -> MetricOutput | None:
    for key, producer in spec.requires.items():
        upstream = producer_outputs.get((producer.__name__, factor))
        if isinstance(upstream, MetricOutput) and math.isnan(upstream.value):
            return MetricOutput(
                name=spec.name,
                value=float("nan"),
                metadata={
                    "reason": "upstream_unavailable",
                    "upstream": producer.__name__,
                    "upstream_reason": upstream.metadata.get("reason", "unknown"),
                    "consumer_param": key,
                },
            )
    return None


def _stamp_spec(out: MetricOutput, spec: MetricSpec) -> MetricOutput:
    return dataclasses.replace(out, spec=spec)


def _resolve_n_obs(
    primary: Sequence[MetricSpec],
    factor: str,
    metric_outputs: Mapping[tuple[str, str], MetricOutput],
) -> int:
    for spec in primary:
        out = metric_outputs.get((spec.name, factor))
        if out is not None and out.n_obs is not None:
            return out.n_obs
    return 0


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
