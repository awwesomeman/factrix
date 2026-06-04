"""Typed metric-spec SSOT for ``factrix/metrics/*.py`` modules.

Each public metric module declares a module-level ``__metric_specs__``
tuple of :class:`MetricSpec` dataclasses, one entry per public callable.
This is the single source of truth for "which standalone metrics live
in which cell" and drives:

- :func:`factrix.list_metrics` — runtime metric discovery API
- ``scripts/mkdocs_hooks/gen_metric_matrix.py`` — docs matrix table
- ``scripts/mkdocs_hooks/gen_metric_name_index.py`` — docs name index
- ``scripts/mkdocs_hooks/gen_evaluate_metric_table.py`` — dispatch table
- ``tests/test_docs_matrix.py`` — coverage + invariant tests

Consumers iterate via :func:`public_specs` (visibility-filtered) or
:func:`_all_specs` (everything, internal). Derived fields use the
small helpers :func:`import_path_for` / :func:`docs_anchor_for`.

Why typed: IDE completion, mypy-checkable axes, refactor-safe field
access. Adding a new metric module just imports :class:`MetricSpec` and
:func:`cell` and declares::

    __metric_specs__ = (
        MetricSpec(
            name="my_metric",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T,
            se_method=SEMethod.HAC,
        ),
    )
"""

from __future__ import annotations

import functools
import importlib
import inspect
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, overload

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    OutputShape,
    SEMethod,
    SpecRole,
    TestMethod,
)
from factrix._errors import IncompatibleAxisError

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_METRICS_DIR = _REPO_ROOT / "factrix" / "metrics"

# Canonical reverse-index URL convention for
# :func:`docs_anchor_for` (#125): docs-root-relative path +
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


@dataclass(frozen=True, slots=True)
class SampleThreshold:
    """Per-metric statistical pre-flight gates against panel shape.

    Declares the sample-size floors below which a metric is
    statistically unusable (``min_*``) or runs with a documented
    bias warning (``warn_*``). :func:`factrix.inspect_panel`
    evaluates these against the inspected panel's
    :class:`PanelProperties` and partitions specs into
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

    _AXES: ClassVar[tuple[str, ...]] = ("periods", "assets", "pairs")

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


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Per-callable typed spec declared via module-level ``__metric_specs__``.

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
    - ``test_method``: primary statistical test family.
    - ``se_method``: standard-error / variance-estimation family.
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
      callable that has a :class:`MetricSpec` in its module's
      ``__metric_specs__`` whose per-factor output the DAG executor
      injects at that parameter. Empty dict means no upstream
      dependency.
    - ``batchable``: ``True`` when the callable accepts
      ``factor_cols=`` and returns ``dict[factor_name, output]`` so
      the DAG executor calls it once across the whole batch.
    - ``sample_threshold``: :class:`SampleThreshold` declaring the
      panel-shape thresholds below which the metric is statistically
      unusable / degraded. Defaults to ``SampleThreshold()`` (all
      ``None`` — no pre-flight gate); :func:`inspect_panel` applies
      the cell-match check and any declared thresholds.
    """

    name: str
    cell: Cell
    aggregation: Aggregation
    test_method: TestMethod
    se_method: SEMethod
    input_shape: InputShape = InputShape.PANEL
    output_shape: OutputShape = OutputShape.SCALAR
    role: SpecRole = SpecRole.METRIC
    requires: dict[str, Callable] = field(default_factory=dict)
    batchable: bool = False
    sample_threshold: SampleThreshold = field(default_factory=SampleThreshold)

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
    """Import ``factrix.metrics.<stem>`` and return its specs.

    Raises ``ValueError`` when the module does not declare a non-empty
    ``__metric_specs__`` tuple or register `@metric` classes — coverage
    enforced by ``tests/test_docs_matrix.py``.
    """
    mod = importlib.import_module(f"factrix.metrics.{stem}")
    specs = getattr(mod, "__metric_specs__", None)
    if not specs:
        from factrix.metrics._registry import REGISTRY

        reg_specs = []
        for _, cls in REGISTRY.items():
            if cls.__module__ == f"factrix.metrics.{stem}":
                reg_specs.append(cls.spec())
        if reg_specs:
            specs = tuple(reg_specs)

    if not specs:
        raise ValueError(
            f"factrix.metrics.{stem}: module-level `__metric_specs__` tuple "
            f"or registered @metric classes are required. See "
            f"factrix._metric_index.MetricSpec docstring."
        )
    if not all(isinstance(s, MetricSpec) for s in specs):
        raise TypeError(
            f"factrix.metrics.{stem}: every spec entry must be a MetricSpec instance."
        )
    for spec in specs:
        if spec.requires:
            _validate_requires(stem, spec, mod)
    return tuple(specs)


def _validate_requires(stem: str, spec: MetricSpec, mod: object) -> None:
    """Check every ``MetricSpec.requires`` entry matches the consumer signature.

    Decoration-time-equivalent check: runs once at module import via
    :func:`_load_module_specs` so a typo'd parameter name or a
    producer callable without its own ``MetricSpec`` raises at load
    time rather than at first DAG dispatch. Replaces the runtime
    ``@batch_primitive`` / ``@ic_consumer`` validation that #440
    retired.
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
        producer_module = importlib.import_module(producer_module_name)
        module_specs = getattr(producer_module, "__metric_specs__", ())

        is_registered = (
            any(s.name == producer_name for s in module_specs)
            or producer_name in REGISTRY
        )
        if not is_registered:
            raise ValueError(
                f"factrix.metrics.{stem}.{spec.name}: producer "
                f"{producer_module_name}.{producer_name} is required by "
                f"key {key!r} but has no MetricSpec in its module's "
                f"`__metric_specs__` tuple or registry."
            )


@functools.cache
def _all_specs() -> tuple[tuple[str, MetricSpec], ...]:
    """Return ``((module_stem, spec), ...)`` across every public metric module.

    Cached — spec tuples are module-level constants, so repeated
    callers (``list_metrics`` in agentic loops, docs-hook regeneration)
    avoid re-importing every public module.
    """
    out: list[tuple[str, MetricSpec]] = []

    # 1. Load legacy module specs
    for stem in _public_metric_stems():
        try:
            mod = importlib.import_module(f"factrix.metrics.{stem}")
            specs = getattr(mod, "__metric_specs__", None)
            if specs:
                for spec in specs:
                    if isinstance(spec, MetricSpec):
                        out.append((stem, spec))
        except (ImportError, ValueError, AttributeError):
            pass

    # 2. Load from the new class-based registry
    from factrix.metrics._registry import REGISTRY

    for _, cls in REGISTRY.items():
        if cls.__module__.startswith("factrix.metrics."):
            stem = cls.__module__.removeprefix("factrix.metrics.")
        else:
            stem = cls.__module__.split(".")[-1]
        spec = cls.spec()
        if not any(s.name == spec.name for _, s in out):
            out.append((stem, spec))

    # Validate registry metrics' requires
    for name, cls in REGISTRY.items():
        spec = cls.spec()
        if spec.requires:
            consumer_params = set(inspect.signature(cls._impl).parameters)
            for key, producer in spec.requires.items():
                if key not in consumer_params:
                    raise ValueError(
                        f"Metric {name!r}: `requires` key {key!r} is not a parameter of {name}."
                    )
                if not callable(producer):
                    raise ValueError(
                        f"Metric {name!r}: `requires[{key!r}]` is not callable."
                    )
                producer_name = producer.__name__
                if producer_name not in REGISTRY and not any(
                    s.name == producer_name for _, s in out
                ):
                    raise ValueError(
                        f"Metric {name!r}: required producer {producer_name!r} is not registered."
                    )

    return tuple(out)


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

    First-party specs are auto-discovered from ``factrix.metrics.*``
    modules' ``__metric_specs__`` tuples (cached). Third-party specs
    are added via :func:`register`; the third-party slice is reflected
    here without recomputing the first-party walk.
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


@overload
def list_metrics() -> dict[str, list[MetricSpec]]: ...
@overload
def list_metrics(
    scope: FactorScope,
    density: FactorDensity,
    *,
    format: Literal["text", "json"] = ...,
    with_import: bool = ...,
) -> list[str] | list[dict[str, Any]]: ...
def list_metrics(
    scope: FactorScope | None = None,
    density: FactorDensity | None = None,
    *,
    format: Literal["text", "json"] = "text",
    with_import: bool = False,
) -> dict[str, list[MetricSpec]] | list[str] | list[dict[str, Any]]:
    """Discover standalone metrics — family-grouped overview or cell filter.

    Two call shapes:

    - **No arguments** → a family-grouped overview
      ``dict[str, list[MetricSpec]]`` keyed by concept family (the
      module stem). This is a catalog, *not* runnable — see
      :func:`_metrics_overview`.
    - **Both ``scope`` and ``density``** → the metrics applicable to that
      ``(scope, density)`` cell, as names (``format="text"``) or
      JSON-serialisable rows (``format="json"``).

    Passing exactly one axis is a usage error.

    DataStructure is intentionally not an input — applicability does not change
    across PANEL / TIMESERIES (per ``docs/reference/metric-applicability.md``).
    Source of truth is the module-level ``__metric_specs__`` tuple in
    each metric module, loaded by :mod:`factrix._metric_index`.

    Args:
        scope: Cell axis to filter on (``FactorScope.INDIVIDUAL`` or
            ``FactorScope.COMMON``). Omit together with ``density`` for
            the overview.
        density: Cell axis to filter on (``FactorDensity.DENSE`` or
            ``FactorDensity.SPARSE``). Omit together with ``scope`` for
            the overview.
        format: ``"text"`` (default) returns metric names sorted by
            ``(module, name)``. ``"json"`` returns ``list[dict]`` rows
            with keys ``name``, ``module``, ``family``, ``cell``,
            ``aggregation``, ``test_method``, ``se_method``,
            ``import_path``, ``input_shape``, ``docs_anchor`` —
            JSON-serialisable, suitable for tooling. ``family`` is the
            concept family (module stem); ``aggregation`` is the
            cross-section/time reduction order.
            ``docs_anchor`` follows
            :data:`factrix._metric_index.DOCS_ANCHOR_FMT` (a
            docs-root-relative path + mkdocstrings symbol fragment).
            ``name`` == ``MetricResult.name`` for all current specs
            (function name = registry key = emitted label).
        with_import: ``"text"`` only. When ``True``, returns a
            two-column ``"name → factrix.metrics.<module>"`` list so
            each row is copy-paste-ready into
            ``from factrix.metrics import <name>``. Ignored under
            ``format="json"`` (the ``import_path`` field is always
            present there).

    Raises:
        ValueError: exactly one of ``scope`` / ``density`` is given
            (pass both for a filtered list, or neither for the overview).
        IncompatibleAxisError: ``(scope, density)`` matches no
            registered metric. In practice all four combos are
            populated, so this is defensive.

    Examples:
        Family-grouped overview (no arguments):

        >>> import factrix as fx
        >>> overview = fx.list_metrics()
        >>> "ic" in overview
        True

        Discover standalone metrics for an INDIVIDUAL × DENSE cell:

        >>> names = fx.list_metrics(
        ...     fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE,
        ... )

        JSON form (for tooling — adds module / family / import_path keys):

        >>> rows = fx.list_metrics(
        ...     fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE, format="json",
        ... )
    """
    if scope is None and density is None:
        return _metrics_overview()
    if scope is None or density is None:
        raise ValueError(
            "list_metrics() takes either no arguments (family-grouped "
            "overview) or both scope and density (cell-filtered list); "
            "got exactly one axis."
        )
    matches = [
        (stem, spec)
        for stem, spec in public_specs()
        if spec.cell.matches(scope, density)
    ]
    if not matches:
        raise IncompatibleAxisError(
            f"no standalone metrics registered for "
            f"(scope={scope.value}, density={density.value})"
        )
    if format == "json":
        return [
            {
                "name": spec.name,
                "module": stem,
                "family": stem,
                "cell": spec.cell.raw,
                "aggregation": spec.aggregation.value,
                "test_method": spec.test_method.value,
                "se_method": spec.se_method.value,
                "import_path": import_path_for(stem),
                "input_shape": spec.input_shape.value,
                "docs_anchor": docs_anchor_for(stem, spec.name),
            }
            for stem, spec in matches
        ]
    if with_import:
        width = max(len(spec.name) for _, spec in matches)
        return [
            f"{spec.name:<{width}} → {import_path_for(stem)}" for stem, spec in matches
        ]
    return [spec.name for _, spec in matches]
