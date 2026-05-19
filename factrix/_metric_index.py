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
small helpers :func:`import_path_for` / :func:`docs_anchor_for` /
:func:`emitted_name_of`.

Why typed: IDE completion, mypy-checkable axes, refactor-safe field
access. Adding a new metric module just imports :class:`MetricSpec` and
:func:`cell` and declares::

    __metric_specs__ = (
        MetricSpec(
            name="my_metric",
            cell=cell(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, mode=Mode.PANEL),
            family="cs-first",
            inference="NW HAC / cross-asset t",
            primitives=("_calc_t_stat", "_p_value_from_t"),
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
from typing import Any, Literal

from factrix._axis import FactorScope, Metric, Mode, Signal, Visibility

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_METRICS_DIR = _REPO_ROOT / "factrix" / "metrics"

# Canonical reverse-index URL convention for
# :func:`docs_anchor_for` (#125): docs-root-relative path +
# mkdocstrings symbol fragment. Centralised here so a future docs URL
# change touches one literal — external prose in
# ``factrix._describe.list_metrics`` and ``docs/api/metric-output.md``
# cite this constant by name rather than restating the string.
DOCS_ANCHOR_FMT: str = "api/metrics/{module}.md#factrix.metrics.{module}.{name}"

# Metrics that ``run_metrics`` cannot auto-discover even though their
# spec is user-facing. Each entry carries a short reason rendered into
# ``MetricsBundle.skipped`` and into the ``UserInputError`` raised when
# a caller names one explicitly via ``metrics=[...]``.
_AUTO_DISCOVER_EXCLUDED: dict[str, str] = {
    "caar": (
        "consumes caar_df; call factrix.metrics.compute_event_returns + "
        "compute_caar then caar(caar_df) directly"
    ),
    "fama_macbeth": (
        "consumes beta_df; call factrix.metrics.compute_fm_betas then "
        "fama_macbeth(beta_df) directly"
    ),
    "beta_sign_consistency": (
        "consumes beta_df; call factrix.metrics.compute_fm_betas then "
        "beta_sign_consistency(beta_df) directly"
    ),
    "ts_beta": (
        "consumes ts_betas_df; call factrix.metrics.compute_ts_betas then "
        "ts_beta(ts_betas_df) directly"
    ),
    "mean_r_squared": (
        "consumes ts_betas_df; call factrix.metrics.compute_ts_betas then "
        "mean_r_squared(ts_betas_df) directly"
    ),
    "ts_beta_sign_consistency": (
        "consumes ts_betas_df; call factrix.metrics.compute_ts_betas then "
        "ts_beta_sign_consistency(ts_betas_df) directly"
    ),
    "mfe_mae_summary": (
        "consumes mfe_mae_df; call factrix.metrics.compute_mfe_mae then "
        "mfe_mae_summary(mfe_mae_df) directly"
    ),
    "hit_rate": (
        "consumes a (date, value) series; e.g. "
        "hit_rate(compute_ic(panel).rename({'ic': 'value'}))"
    ),
    "multi_split_oos_decay": (
        "consumes a (date, value) series; e.g. "
        "multi_split_oos_decay(compute_spread_series(panel)"
        ".rename({'spread': 'value'}))"
    ),
    "ic_trend": (
        "consumes a (date, value) series; e.g. "
        "ic_trend(compute_ic(panel).rename({'ic': 'value'}))"
    ),
    "spanning_alpha": (
        "consumes factor_spread; call after compute_spread_series on the "
        "challenger factor"
    ),
    "greedy_forward_selection": (
        "consumes factor_spreads dict; call after compute_spread_series on "
        "each candidate factor"
    ),
}


# ---------------------------------------------------------------------------
# Cell + Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """Parsed ``(scope, signal, metric, mode)`` cell tuple.

    ``None`` represents the ``*`` wildcard along an axis. ``raw``
    preserves the canonical display label rendered into the docs
    matrix.

    :meth:`matches` filters on whichever axes the caller supplies a
    concrete value for. Pass ``mode=`` to enforce mode applicability
    (e.g. ``IC`` cell declares ``mode=Mode.PANEL`` because IC has no
    cross-section in TIMESERIES); omit it for axis-only queries that
    do not care about runtime mode.
    """

    scope: FactorScope | None
    signal: Signal | None
    metric: Metric | None
    mode: Mode | None
    raw: str

    def matches(
        self,
        scope: FactorScope,
        signal: Signal,
        mode: Mode | None = None,
    ) -> bool:
        """Return True if this cell is applicable to ``(scope, signal[, mode])``.

        ``mode=None`` (default) skips the mode axis check — useful
        for purely structural axis queries.
        """
        scope_ok = self.scope is None or self.scope == scope
        signal_ok = self.signal is None or self.signal == signal
        mode_ok = mode is None or self.mode is None or self.mode == mode
        return scope_ok and signal_ok and mode_ok


def _axis_token(value: FactorScope | Signal | Metric | Mode | None) -> str:
    """Render an axis enum (or ``None`` = wildcard) as its uppercase token."""
    if value is None:
        return "*"
    return value.value.upper()


def cell(
    scope: FactorScope | None,
    signal: Signal | None,
    metric: Metric | None = None,
    mode: Mode | None = None,
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
            f"({_axis_token(scope)}, {_axis_token(signal)}, "
            f"{_axis_token(metric)}, {_axis_token(mode)})"
        )
    return Cell(scope=scope, signal=signal, metric=metric, mode=mode, raw=raw)


@dataclass(frozen=True, slots=True)
class SampleFloor:
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
    """

    min_periods: int | None = None
    warn_periods: int | None = None
    min_assets: int | None = None
    warn_assets: int | None = None
    min_pairs: int | None = None
    warn_pairs: int | None = None


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Per-callable typed spec declared via module-level ``__metric_specs__``.

    One :class:`MetricSpec` per public callable in the declaring
    ``factrix/metrics/*.py`` module.

    Fields:

    - ``name``: function name in the declaring module.
    - ``cell``: applicable ``(scope, signal, metric, mode)`` cell;
      ``None`` along any axis denotes ``*`` wildcard.
    - ``family``: aggregation / inference family label —
      ``"cs-first"`` (cross-section step then time aggregation),
      ``"per-event"``, ``"ts-only"``, ``"ts-first"``, ``"static-cs"``,
      or free-form prose for special cases (e.g. spanning's
      ``"factor-return-series consumer (post-PANEL pipeline)"``).
    - ``inference``: inference / standard-error family — e.g.
      ``"NW HAC / cross-asset t"``, ``"binomial"``,
      ``"nonparametric rank"``, ``"no formal H_0"``.
    - ``primitives``: informational tuple of underlying helper-function
      names the public callable composes. Surfaces in the docs matrix
      for primitive-graph completeness.
    - ``input_kind``: ``"panel"`` (date-keyed DataFrame, eligible for
      date-slicing dispatchers like :func:`factrix.by_slice`) or
      ``"scalar"`` (pre-aggregated-scalar utility like
      :func:`factrix.metrics.breakeven_cost`).
    - ``emitted_name``: literal ``MetricOutput.name`` string at
      runtime. ``None`` (default) means the runtime label matches
      ``name`` — the common case. Set explicitly when the callable
      emits a different label (e.g. ``fama_macbeth`` emits
      ``fm_beta``). Consumers holding a :class:`~factrix.MetricOutput`
      should resolve via :func:`emitted_name_of`.
    - ``requires``: ``{consumer_param_name: producer_callable}``. Key
      is a parameter on the declaring callable; value is another
      callable that has a :class:`MetricSpec` in its module's
      ``__metric_specs__`` whose per-factor output the DAG executor
      injects at that parameter. Empty dict means no upstream
      dependency.
    - ``batchable``: ``True`` when the callable accepts
      ``factor_cols=`` and returns ``dict[factor_name, output]`` so
      the DAG executor calls it once across the whole batch.
    - ``visibility``: ``PUBLIC`` for user-facing metric (default);
      ``INTERNAL`` for stage-1 helpers (excluded from
      ``list_metrics`` / ``inspection.metrics.*`` / result dict keys
      but pulled by the DAG via ``requires``).
    - ``sample_floor``: optional :class:`SampleFloor` declaring the
      panel-shape thresholds below which the metric is statistically
      unusable / degraded. ``None`` (default) means no pre-flight
      sample-size gate is declared; :func:`inspect_panel` will only
      apply the cell-match check for this spec.
    """

    name: str
    cell: Cell
    family: str
    inference: str
    primitives: tuple[str, ...] = ()
    input_kind: Literal["panel", "scalar"] = "panel"
    emitted_name: str | None = None
    requires: dict[str, Callable] = field(default_factory=dict)
    batchable: bool = False
    visibility: Visibility = Visibility.PUBLIC
    sample_floor: SampleFloor | None = None


# ---------------------------------------------------------------------------
# Loader + caches
# ---------------------------------------------------------------------------


def _public_metric_stems() -> list[str]:
    """Return sorted stems of every public ``factrix/metrics/*.py``."""
    return sorted(
        p.stem for p in _METRICS_DIR.glob("*.py") if not p.stem.startswith("_")
    )


def _load_module_specs(stem: str) -> tuple[MetricSpec, ...]:
    """Import ``factrix.metrics.<stem>`` and return its ``__metric_specs__``.

    Raises ``ValueError`` when the module does not declare a non-empty
    ``__metric_specs__`` tuple — coverage enforced by
    ``tests/test_docs_matrix.py``.
    """
    mod = importlib.import_module(f"factrix.metrics.{stem}")
    specs = getattr(mod, "__metric_specs__", None)
    if not specs:
        raise ValueError(
            f"factrix.metrics.{stem}: module-level `__metric_specs__` tuple "
            f"is required (one MetricSpec per public callable). See "
            f"factrix._metric_index.MetricSpec docstring."
        )
    if not all(isinstance(s, MetricSpec) for s in specs):
        raise TypeError(
            f"factrix.metrics.{stem}: every `__metric_specs__` entry must be "
            f"a MetricSpec instance."
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
    consumer_params = set(inspect.signature(consumer).parameters)
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
        producer_module = importlib.import_module(producer.__module__)
        module_specs = getattr(producer_module, "__metric_specs__", ())
        if not any(s.name == producer.__name__ for s in module_specs):
            raise ValueError(
                f"factrix.metrics.{stem}.{spec.name}: producer "
                f"{producer.__module__}.{producer.__name__} is required by "
                f"key {key!r} but has no MetricSpec in its module's "
                f"`__metric_specs__` tuple."
            )


@functools.cache
def _all_specs() -> tuple[tuple[str, MetricSpec], ...]:
    """Return ``((module_stem, spec), ...)`` across every public metric module.

    Cached — spec tuples are module-level constants, so repeated
    callers (``list_metrics`` in agentic loops, docs-hook regeneration)
    avoid re-importing every public module.
    """
    out: list[tuple[str, MetricSpec]] = []
    for stem in _public_metric_stems():
        for spec in _load_module_specs(stem):
            out.append((stem, spec))
    return tuple(out)


@functools.cache
def public_specs() -> tuple[tuple[str, MetricSpec], ...]:
    """Return ``((stem, spec), ...)`` for every ``visibility=PUBLIC`` spec.

    Sorted by ``(stem, spec.name)``. Stage-1 helpers
    (``visibility=INTERNAL``) are filtered out — they are pulled by
    the DAG executor via :attr:`MetricSpec.requires` and do not
    surface in :func:`factrix.list_metrics` or result dict keys.
    """
    out = [
        (stem, spec)
        for stem, spec in _all_specs()
        if spec.visibility is Visibility.PUBLIC
    ]
    out.sort(key=lambda pair: (pair[0], pair[1].name))
    return tuple(out)


def import_path_for(stem: str) -> str:
    """Return the public import path for a metric module stem."""
    return f"factrix.metrics.{stem}"


def docs_anchor_for(stem: str, name: str) -> str:
    """Return the docs-root-relative anchor for a metric callable."""
    return DOCS_ANCHOR_FMT.format(module=stem, name=name)


def emitted_name_of(spec: MetricSpec) -> str:
    """Return the runtime ``MetricOutput.name`` label for a spec."""
    return spec.emitted_name or spec.name


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
    """Register a third-party metric callable carrying ``__metric_spec__``.

    The callable must have been decorated with :func:`metric_spec` (or
    have ``__metric_spec__`` attached by other means). Registration
    side-effects:

    1. Adds the spec to ``_METRIC_REGISTRY`` so :func:`spec_by_name`
       and the DAG executor can resolve it by name.
    2. Sets ``factrix.metrics.<spec.name>`` to the callable so IDE /
       mypy / interactive use see a real attribute (parallel to
       first-party metrics imported into the namespace).
    3. Clears any caches that would otherwise mask the new spec.

    Raises:
        TypeError: ``fn`` is not callable or has no ``__metric_spec__``.
        ValueError: a spec with the same name is already registered
            (third-party) or exists as a first-party spec; explicit
            re-registration is disallowed to keep the registry
            authoritative.
    """
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

    from factrix._dag import _registry_callable_table

    _registry_callable_table.cache_clear()
