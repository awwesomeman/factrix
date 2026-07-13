"""Typed metric-spec SSOT for ``factrix/metrics/*.py`` modules.

Each public metric module decorates its public callables with
``@metric``, registering one :class:`MetricSpec` per callable in the
shared registry. This is the single source of truth for "which
standalone metrics live in which cell" and drives:

- :func:`factrix.list_metrics` — runtime metric discovery API
- ``scripts/mkdocs_hooks/gen_metric_matrix.py`` — docs matrix table
- ``scripts/mkdocs_hooks/gen_metric_name_index.py`` — docs name index

Consumers iterate via :func:`public_specs` (visibility-filtered) or
:func:`_all_specs` (everything, internal). Derived fields use the
small helpers :func:`import_path_for` / :func:`docs_anchor_for`.

Why typed: IDE completion, mypy-checkable axes, refactor-safe field
access. Adding a new metric just decorates the callable with
``@metric``, which builds and registers its :class:`MetricSpec`::

    @metric(
        cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL),
        aggregation=Aggregation.CS_THEN_TS,
    )
    def my_metric(panel: pl.DataFrame) -> MetricResult:
        ...
"""

from __future__ import annotations

import functools
import importlib
import inspect
import pathlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    OutputShape,
    SpecRole,
    Tier,
)

if TYPE_CHECKING:
    import polars as pl

    from factrix._inspect import DataProperties
    from factrix.metrics._base import MetricBase

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_METRICS_DIR = _REPO_ROOT / "factrix" / "metrics"

# Canonical reverse-index URL convention for
# :func:`docs_anchor_for`: docs-root-relative path +
# mkdocstrings symbol fragment. Centralised here so a future docs URL
# change touches one literal — external prose in
# ``factrix._describe.list_metrics`` and ``docs/api/metric-output.md``
# cite this constant by name rather than restating the string.
DOCS_ANCHOR_FMT: str = "api/metrics/{module}.md#factrix.metrics.{module}.{name}"

# ---------------------------------------------------------------------------
# Cell + Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """Parsed ``(scope, density, structure)`` cell tuple.

    ``None`` represents the ``*`` wildcard along an axis. ``raw``
    preserves the canonical display label rendered into the docs
    matrix.

    :meth:`matches` filters on whichever axes the caller supplies a
    concrete value for. Pass ``structure=`` to enforce structure applicability
    (e.g. ``IC`` cell declares ``structure=DataStructure.PANEL`` because IC has no
    cross-section in TIMESERIES); omit it for axis-only queries that
    do not care about runtime structure.
    """

    scope: FactorScope | None
    density: FactorDensity | None
    structure: DataStructure | None
    raw: str

    def matches(
        self,
        scope: FactorScope,
        density: FactorDensity,
        structure: DataStructure | None = None,
    ) -> bool:
        """Return True if this cell is applicable to ``(scope, density[, structure])``.

        ``structure=None`` (default) skips the structure axis check — useful
        for purely structural axis queries.
        """
        scope_ok = self.scope is None or self.scope == scope
        density_ok = self.density is None or self.density == density
        structure_ok = (
            structure is None or self.structure is None or self.structure == structure
        )
        return scope_ok and density_ok and structure_ok


def _axis_token(value: FactorScope | FactorDensity | DataStructure | None) -> str:
    """Render an axis enum (or ``None`` = wildcard) as its uppercase token."""
    if value is None:
        return "*"
    return value.value.upper()


def cell(
    scope: FactorScope | None,
    density: FactorDensity | None,
    structure: DataStructure | None = None,
    *,
    raw: str | None = None,
) -> Cell:
    """Build a :class:`Cell` with an auto-rendered uppercase ``raw`` label.

    Pass ``raw=`` explicitly only when the prose label carries
    information not encoded in the axis tuple — the only current case
    is ``factrix.metrics.spanning`` whose metrics consume a
    post-pipeline factor-return series.
    """
    if raw is None:
        raw = (
            f"({_axis_token(scope)}, {_axis_token(density)}, {_axis_token(structure)})"
        )
    return Cell(scope=scope, density=density, structure=structure, raw=raw)


class AxisVerdict(NamedTuple):
    """One shape axis's tier together with the data the decision used.

    ``floor`` / ``warn`` are the spec's ``min_<axis>`` / ``warn_<axis>``
    bounds (``None`` when ungated); ``n`` is the panel's actual count.
    """

    axis: str
    n: int
    floor: int | None
    warn: int | None
    tier: Tier


@dataclass(frozen=True, slots=True)
class SampleThreshold:
    """Per-metric statistical pre-flight gates against panel shape.

    Declares the sample-size floors below which a metric is
    statistically unusable (``min_*``) or runs with a documented
    bias warning (``warn_*``). :func:`factrix.inspect_data`
    evaluates these against the inspected panel's
    :class:`DataProperties` and partitions specs into
    ``usable`` / ``unusable``.

    The same constants are read by the metric's own short-circuit
    logic at run time — single source of truth for what counts as
    "thin sample" per metric.

    Per axis the ``min_*`` floor must not exceed the ``warn_*`` floor:
    a metric is unusable below ``min``, degraded between ``min`` and
    ``warn``, and clean at or above ``warn``, so ``min <= warn`` is
    required for that ordering to hold. :meth:`__post_init__` enforces
    it at construction.
    """

    min_periods: int | None = None
    warn_periods: int | None = None
    min_assets: int | None = None
    warn_assets: int | None = None
    min_pairs: int | None = None
    warn_pairs: int | None = None
    min_events: int | None = None
    warn_events: int | None = None

    _AXES: ClassVar[tuple[str, ...]] = ("periods", "assets", "pairs", "events")

    def __post_init__(self) -> None:
        for axis in self._AXES:
            lo, hi = self._bounds(axis)
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(
                    f"SampleThreshold.{axis}: min ({lo}) must not exceed warn ({hi})"
                )

    def _bounds(self, axis: str) -> tuple[int | None, int | None]:
        """Return ``(min_<axis>, warn_<axis>)`` for one shape axis."""
        return getattr(self, f"min_{axis}"), getattr(self, f"warn_{axis}")

    def iter_verdicts(self, properties: DataProperties) -> Iterator[AxisVerdict]:
        """Yield the per-axis verdict for each shape axis, in ``_AXES`` order.

        Single source of truth for the tier decision: ``per_axis_verdict``,
        ``verdict`` and ``factrix.inspect_data``'s blocker/warning assembly
        all consume this so the floor → tier mapping lives in exactly one
        place. Each yielded record also carries the actual ``n`` and the
        ``floor``/``warn`` bounds the decision was made against, so callers
        can build messages without re-reading the spec.
        """
        for axis in self._AXES:
            floor, warn = self._bounds(axis)
            n = getattr(properties, f"n_{axis}")
            if floor is not None and n < floor:
                tier = Tier.UNUSABLE
            elif warn is not None and n < warn:
                tier = Tier.DEGRADED
            else:
                tier = Tier.CLEAN
            yield AxisVerdict(axis=axis, n=n, floor=floor, warn=warn, tier=tier)

    def per_axis_verdict(self, properties: DataProperties) -> dict[str, Tier]:
        """Return the usability tier for each shape axis (periods, assets, pairs, events)."""
        return {v.axis: v.tier for v in self.iter_verdicts(properties)}

    def verdict(self, properties: DataProperties) -> Tier:
        """Return the worst usability tier across all shape axes."""
        worst = Tier.CLEAN
        for v in self.iter_verdicts(properties):
            if v.tier is Tier.UNUSABLE:
                return Tier.UNUSABLE
            if v.tier is Tier.DEGRADED:
                worst = Tier.DEGRADED
        return worst


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Per-callable typed spec built and registered by the ``@metric`` decorator.

    One :class:`MetricSpec` per public callable in the declaring
    ``factrix/metrics/*.py`` module.

    Fields:

    - ``name``: function name in the declaring module.
    - ``cell``: applicable ``(scope, density, structure)`` cell;
      ``None`` along any axis denotes ``*`` wildcard.
    - ``aggregation``: how the cross-section and time-series reductions
      compose. Load-bearing for the DAG executor / FDR. Distinct from
      the *concept family* (``ic`` / ``decay`` / ``quantile``), which is
      the declaring module's stem and is derived from file location
      (the ``file = family`` invariant), never stored on the spec.
    - ``input_shape``: shape of data the callable directly receives
      (``PANEL`` / ``SERIES`` / ``SCALAR``).
    - ``output_shape``: shape of the returned value. ``METRIC`` specs
      must have ``SCALAR`` (enforced by ``__post_init__``).
    - ``role``: ``METRIC`` for user-facing result-producing callables
      (default); ``PIPELINE`` for stage-1 helpers excluded from
      ``list_metrics`` / ``inspection.metrics.*`` / result dict keys
      but pulled by the DAG via ``requires``.
    - ``requires``: ``{consumer_param_name: producer_callable}``. Key
      is a parameter on the declaring callable; value is another
      ``@metric`` callable whose per-factor output the DAG executor
      injects at that parameter. Empty dict means no upstream
      dependency.
    - ``batchable``: ``True`` when the callable accepts
      ``factor_cols=`` and returns ``dict[factor_name, output]`` so
      the DAG executor calls it once across the whole batch.
    - ``sample_threshold``: :class:`SampleThreshold` declaring the
      panel-shape thresholds below which the metric is statistically
      unusable / degraded. Defaults to ``SampleThreshold()`` (all
      ``None`` — no pre-flight gate); :func:`inspect_data` applies
      the cell-match check and any declared thresholds.
    - ``requires_continuous_magnitude``: ``True`` when the metric needs a
      continuous-magnitude factor (``|factor|`` must vary across events);
      :func:`inspect_data` blocks it on a discrete ±k signal, matching the
      metric's run-time ``not_applicable_discrete_signal`` short-circuit.
      Defaults to ``False``.
    - ``slice_boundary_sensitive``: ``True`` when evaluating independently on
      date-axis slices changes the computation by truncating ordered history,
      resetting a sampling phase, or refitting a time-series model. This is a
      capability of the estimator, not an inference from ``aggregation``.
    """

    name: str
    cell: Cell
    aggregation: Aggregation
    input_shape: InputShape = InputShape.PANEL
    output_shape: OutputShape = OutputShape.SCALAR
    role: SpecRole = SpecRole.METRIC
    requires: dict[str, Callable] = field(default_factory=dict)
    batchable: bool = False
    sample_threshold: SampleThreshold = field(default_factory=SampleThreshold)
    # Pre-flight gate beyond cell/sample: the metric needs a continuous-magnitude
    # factor (``|factor|`` varies across events). ``inspect_data`` blocks it on a
    # discrete ±k signal, matching the metric's run-time short-circuit.
    requires_continuous_magnitude: bool = False
    slice_boundary_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.role is SpecRole.METRIC and self.output_shape is not OutputShape.SCALAR:
            raise ValueError(
                f"MetricSpec '{self.name}': role=METRIC requires output_shape=SCALAR, "
                f"got output_shape={self.output_shape.name}."
            )


# ---------------------------------------------------------------------------
# Loader + caches
# ---------------------------------------------------------------------------


def _public_metric_stems() -> list[str]:
    """Return sorted stems of every public ``factrix/metrics/*.py``."""
    return sorted(
        p.stem for p in _METRICS_DIR.glob("*.py") if not p.stem.startswith("_")
    )


def _load_module_specs(stem: str) -> tuple[MetricSpec, ...]:
    """Import ``factrix.metrics.<stem>`` and return its registered specs.

    Raises ``ValueError`` when the module registers no ``@metric``
    classes — coverage enforced by ``tests/test_docs_matrix.py``.
    """
    mod = importlib.import_module(f"factrix.metrics.{stem}")
    from factrix.metrics._registry import REGISTRY

    specs = tuple(
        cls.spec()
        for cls in REGISTRY.values()
        if cls.__module__ == f"factrix.metrics.{stem}"
    )
    if not specs:
        raise ValueError(
            f"factrix.metrics.{stem}: no @metric classes registered. "
            f"Decorate each public callable with @metric. See "
            f"factrix._metric_index.MetricSpec docstring."
        )
    for spec in specs:
        if spec.requires:
            _validate_requires(stem, spec, mod)
    return specs


def _validate_requires(stem: str, spec: MetricSpec, mod: object) -> None:
    """Check every ``MetricSpec.requires`` entry matches the consumer signature.

    Decoration-time-equivalent check: runs once at module import via
    :func:`_load_module_specs` so a typo'd parameter name or a
    producer callable without its own ``MetricSpec`` raises at load
    time rather than at first DAG dispatch. Replaces the runtime
    ``@batch_primitive`` / ``@ic_consumer`` validation.
    """
    consumer = getattr(mod, spec.name, None)
    if not callable(consumer):
        raise ValueError(
            f"factrix.metrics.{stem}.{spec.name}: declares `requires=` "
            f"but no callable of that name is exported from the module."
        )
    consumer_to_inspect = consumer._impl if hasattr(consumer, "_impl") else consumer
    consumer_params = set(inspect.signature(consumer_to_inspect).parameters)
    from factrix.metrics._registry import REGISTRY

    for key, producer in spec.requires.items():
        if key not in consumer_params:
            raise ValueError(
                f"factrix.metrics.{stem}.{spec.name}: `requires` key "
                f"{key!r} is not a parameter of {spec.name}{tuple(consumer_params)}. "
                f"The DAG executor injects upstream output via this kwarg, "
                f"so the name must match the consumer signature exactly."
            )
        if not callable(producer):
            raise ValueError(
                f"factrix.metrics.{stem}.{spec.name}: `requires[{key!r}]` "
                f"is not callable (got {type(producer).__name__})."
            )

        producer_module_name = producer.__module__
        producer_name = producer.__name__

        if producer_name not in REGISTRY:
            raise ValueError(
                f"factrix.metrics.{stem}.{spec.name}: producer "
                f"{producer_module_name}.{producer_name} is required by "
                f"key {key!r} but is not a registered @metric class."
            )

        producer_spec = REGISTRY[producer_name].spec()
        if producer_spec.output_shape.value != spec.input_shape.value:
            raise ValueError(
                f"factrix.metrics.{stem}.{spec.name}: `requires` shape mismatch. "
                f"Consumer expects {spec.input_shape.value!r} but producer "
                f"{producer_name!r} outputs {producer_spec.output_shape.value!r}."
            )


@functools.cache
def _all_specs() -> tuple[tuple[str, MetricSpec], ...]:
    """Return ``((module_stem, spec), ...)`` across every public metric module.

    Cached — spec tuples are module-level constants, so repeated
    callers (``list_metrics`` in agentic loops, docs-hook regeneration)
    avoid re-importing every public module.
    """
    out: list[tuple[str, MetricSpec]] = []
    seen: set[str] = set()

    # 1. Public modules: ``@metric`` classes surfaced by
    #    ``_load_module_specs``, which also validates each spec's
    #    ``requires`` against its consumer signature.
    for stem in _public_metric_stems():
        try:
            specs = _load_module_specs(stem)
        except ImportError:
            continue
        for spec in specs:
            if spec.name not in seen:
                out.append((stem, spec))
                seen.add(spec.name)

    # 2. Registry classes not owned by a public module (``_primitives/*``
    #    pipeline producers, third-party or test registrations). These never
    #    pass through ``_load_module_specs``, so validate their ``requires``
    #    here — public-module classes are already validated in step 1.
    from factrix.metrics._registry import REGISTRY

    for cls in REGISTRY.values():
        spec = cls.spec()
        if spec.name in seen:
            continue
        if spec.requires:
            _validate_registry_requires(cls, spec)
        module = cls.__module__
        stem = (
            module.removeprefix("factrix.metrics.")
            if module.startswith("factrix.metrics.")
            else module.split(".")[-1]
        )
        out.append((stem, spec))
        seen.add(spec.name)

    return tuple(out)


def _validate_registry_requires(cls: type[MetricBase], spec: MetricSpec) -> None:
    """Validate ``requires`` for a ``@metric`` class outside a public module.

    Mirrors :func:`_validate_requires` for registry-only classes (pipeline
    primitives, third-party / test registrations) whose consumer is the
    class's own ``_impl``.
    """
    from factrix.metrics._registry import REGISTRY

    consumer_params = set(inspect.signature(cls._impl).parameters)
    for key, producer in spec.requires.items():
        if key not in consumer_params:
            raise ValueError(
                f"Metric {spec.name!r}: `requires` key {key!r} is not a "
                f"parameter of {spec.name}."
            )
        if not callable(producer):
            raise ValueError(
                f"Metric {spec.name!r}: `requires[{key!r}]` is not callable."
            )
        if producer.__name__ not in REGISTRY:
            raise ValueError(
                f"Metric {spec.name!r}: required producer "
                f"{producer.__name__!r} is not registered."
            )

        producer_spec = REGISTRY[producer.__name__].spec()
        if producer_spec.output_shape.value != spec.input_shape.value:
            raise ValueError(
                f"Metric {spec.name!r}: `requires` shape mismatch. "
                f"Consumer expects {spec.input_shape.value!r} but producer "
                f"{producer.__name__!r} outputs {producer_spec.output_shape.value!r}."
            )


@functools.cache
def public_specs() -> tuple[tuple[str, MetricSpec], ...]:
    """Return ``((stem, spec), ...)`` for every ``visibility=PUBLIC`` spec.

    Sorted by ``(stem, spec.name)``. Stage-1 helpers
    (``visibility=INTERNAL``) are filtered out — they are pulled by
    the DAG executor via :attr:`MetricSpec.requires` and do not
    surface in :func:`factrix.list_metrics` or result dict keys.
    """
    out = [(stem, spec) for stem, spec in _all_specs() if spec.role is SpecRole.METRIC]
    out.sort(key=lambda pair: (pair[0], pair[1].name))
    return tuple(out)


def import_path_for(stem: str) -> str:
    """Return the public import path for a metric module stem."""
    return f"factrix.metrics.{stem}"


def docs_anchor_for(stem: str, name: str) -> str:
    """Return the docs-root-relative anchor for a metric callable."""
    return DOCS_ANCHOR_FMT.format(module=stem, name=name)


@functools.cache
def module_specs(stem: str) -> tuple[MetricSpec, ...]:
    """Return the spec tuple declared by one metric module."""
    return _load_module_specs(stem)


@functools.cache
def _first_party_spec_by_name() -> dict[str, MetricSpec]:
    """Return ``{name: spec}`` for first-party specs only (cached)."""
    return {spec.name: spec for _, spec in _all_specs()}


def spec_by_name() -> dict[str, MetricSpec]:
    """Return ``{name: spec}`` unioning first-party + third-party registered specs.

    First-party specs are auto-discovered from the ``@metric`` classes
    registered by ``factrix.metrics.*`` modules (cached). Third-party
    specs are added via :func:`register`; the third-party slice is
    reflected here without recomputing the first-party walk.
    """
    out = dict(_first_party_spec_by_name())
    out.update(_METRIC_REGISTRY)
    return out


# ---------------------------------------------------------------------------
# Third-party metric registration surface
# ---------------------------------------------------------------------------


_METRIC_REGISTRY: dict[str, MetricSpec] = {}
"""Third-party metric registry populated via :func:`register`.

Kept separate from the first-party cache so first-party discovery
stays a single import-time walk and third-party additions don't
invalidate that cache. :func:`spec_by_name` unions both sources.
"""


def metric_spec(spec: MetricSpec) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Stamp ``__metric_spec__`` onto a metric callable.

    Pure metadata stamp — does **not** register. Pair with an explicit
    :func:`register` call to make the spec visible to
    :func:`list_metrics` / :func:`spec_by_name` / DAG dispatch::

        @metric_spec(MetricSpec(name="custom_ic", cell=..., ...))
        def custom_ic(panel, ...): ...

        factrix.metrics.register(custom_ic)

    Accepts the full :class:`MetricSpec` object rather than spreading
    kwargs so the decorator surface stays evergreen across MetricSpec
    field changes (e.g. the structural rework in the
    panel-dataclass-unification follow-up).
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__metric_spec__ = spec  # type: ignore[attr-defined]
        return fn

    return _wrap


def register(fn: Callable[..., Any]) -> None:
    """Register a third-party metric class or callable."""
    from factrix.metrics._base import MetricBase

    # If it is a MetricBase subclass, use the new registry
    if isinstance(fn, type) and issubclass(fn, MetricBase):
        from factrix.metrics._registry import register as reg

        reg(fn)
        return

    if not callable(fn):
        raise TypeError(f"register(): expected a callable; got {type(fn).__name__}.")
    spec = getattr(fn, "__metric_spec__", None)
    if not isinstance(spec, MetricSpec):
        raise TypeError(
            "register(): callable has no __metric_spec__ attribute. "
            "Apply @metric_spec(MetricSpec(...)) first."
        )
    name = spec.name
    if name in _METRIC_REGISTRY:
        raise ValueError(
            f"register(): metric {name!r} already registered (third-party)."
        )
    if name in _first_party_spec_by_name():
        raise ValueError(
            f"register(): metric {name!r} clashes with a first-party spec; "
            f"pick a different name."
        )

    import factrix.metrics as _metrics_pkg

    _METRIC_REGISTRY[name] = spec
    setattr(_metrics_pkg, name, fn)

    _first_party_spec_by_name.cache_clear()
    public_specs.cache_clear()
    _all_specs.cache_clear()

    from factrix._dag import _registry_callable_table

    _registry_callable_table.cache_clear()


# ---------------------------------------------------------------------------
# list_metrics — programmatic standalone-metric discovery
# ---------------------------------------------------------------------------


def _metrics_overview() -> dict[str, list[MetricSpec]]:
    """Family-grouped catalog of every public metric spec.

    Keys are concept-family names (the declaring module stem, per the
    ``file = family`` invariant); values are that module's public specs
    in registry order — sorted, because :func:`public_specs` yields
    ``(stem, name)`` sorted. See :func:`list_metrics` for the public
    contract.
    """
    out: dict[str, list[MetricSpec]] = {}
    for stem, spec in public_specs():
        out.setdefault(stem, []).append(spec)
    return out


def list_metrics() -> dict[str, list[MetricSpec]]:
    """Family-grouped catalog of every public standalone metric.

    Returns a ``dict[str, list[MetricSpec]]`` keyed by concept family
    (the declaring module stem, per the ``file = family`` invariant);
    values are that module's public specs in registry order. This is a
    catalog, *not* runnable — pass concrete metric callables to
    :func:`factrix.evaluate`.

    For per-cell applicability — which metrics actually run on a given
    panel, and which are degraded or blocked — inspect a real panel with
    :func:`factrix.inspect_data` and read its ``usable`` / ``degraded``
    / ``unusable`` partitions. (The former ``list_metrics(scope, density)``
    cell filter is retired; ``inspect_data`` subsumes it with a
    structure-aware, sample-floor-aware verdict.)

    Source of truth is the registered ``@metric`` classes in each
    ``factrix/metrics/*.py`` module, loaded by :mod:`factrix._metric_index`.

    Examples:
        >>> import factrix as fx
        >>> overview = fx.list_metrics()
        >>> "ic" in overview
        True
    """
    return _metrics_overview()


def metrics_summary() -> pl.DataFrame:
    """Compact one-line-per-metric catalog for discovery.

    The readable companion to :func:`list_metrics`: a ``pl.DataFrame`` with
    columns ``family`` (concept family / module stem), ``metric`` (the public
    callable name you import and pass to :func:`factrix.evaluate`), and
    ``summary`` (the first line of the callable's docstring). Sorted by
    ``(family, metric)``.

    Use this to *browse* the catalog; reach for :func:`list_metrics` when you
    need the full ``MetricSpec`` (cell, aggregation, sample thresholds) for
    programmatic filtering, and :func:`factrix.inspect_data` for which metrics
    actually run on a given panel.

    Examples:
        >>> import factrix as fx
        >>> summary = fx.metrics_summary()
        >>> summary.columns
        ['family', 'metric', 'summary']
        >>> "ic" in summary["metric"].to_list()
        True
    """
    import polars as pl

    from factrix.metrics._registry import REGISTRY

    rows = []
    for stem, spec in public_specs():
        cls = REGISTRY.get(spec.name)
        doc = (cls.__doc__ or "").strip() if cls is not None else ""
        summary = doc.split("\n", 1)[0].strip()
        rows.append({"family": stem, "metric": spec.name, "summary": summary})
    return pl.DataFrame(
        rows,
        schema={"family": pl.String, "metric": pl.String, "summary": pl.String},
        orient="row",
    )
