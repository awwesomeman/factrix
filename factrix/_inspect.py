"""``fx.inspect_data`` — typed data introspection with per-metric verdict.

Two-stage applicability model:

1. **Cell match** — ``(scope, density, structure)`` axes on
   :class:`MetricSpec.cell` must agree with the inspected data's
   :class:`DataProperties`. DataStructure is integral to the match — a metric
   whose cell declares ``structure=DataStructure.PANEL`` (e.g. IC, which has no
   cross-section in TIMESERIES) is unusable when the data is
   single-asset, not just "degraded".
2. **Sample threshold** — :class:`MetricSpec.sample_threshold` declares the
   data-shape thresholds below which the metric short-circuits
   (``min_*``) or runs with a documented bias warning (``warn_*``).
   :func:`inspect_data` evaluates these against
   :class:`DataProperties`.

Each public spec receives one :class:`MetricApplicability` verdict
exposing ``usable`` / ``warnings`` / ``blockers``. The flat
``list[MetricApplicability]`` on :class:`DataInspection.metrics` is
the single source of truth; the ``usable`` / ``degraded`` /
``unusable`` properties expose it as a mutually exclusive partition.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

import polars as pl

from factrix._axis import DataStructure, FactorDensity, FactorScope, Tier
from factrix._codes import WarningCode, cross_section_tier
from factrix._data_input import _FORWARD_PERIODS_COL
from factrix._metric_index import MetricSpec, public_specs
from factrix._results import Warning
from factrix._types import MIN_IC_ASSETS_HARD, MIN_IC_ASSETS_WARN
from factrix.metrics._primitives._fm_betas import (
    MIN_FM_ASSETS_HARD,
    MIN_FM_ASSETS_WARN,
)

if TYPE_CHECKING:
    from factrix.metrics._base import MetricBase

_SPARSITY_THRESHOLD: float = 0.5
_INSPECT_RESERVED: frozenset[str] = frozenset(
    {"date", "asset_id", "forward_return", "price", _FORWARD_PERIODS_COL}
)


def _detect_structure(data: Any) -> DataStructure:
    """Return ``TIMESERIES`` if the data has a single asset, else ``PANEL``."""
    return (
        DataStructure.TIMESERIES
        if data["asset_id"].n_unique() <= 1
        else DataStructure.PANEL
    )


def _detect_density(raw: Any) -> tuple[FactorDensity, str, float]:
    """Sparsity ratio in ``factor`` ≥ 0.5 → SPARSE, else DENSE.

    Returns ``(density, reason, sparsity)`` where ``sparsity`` is the
    zero-ratio over non-null factor cells.
    """
    factor = raw["factor"].drop_nulls()
    n = len(factor)
    if n == 0:
        return (
            FactorDensity.DENSE,
            "factor column has no non-null cells: defaulting to DENSE",
            math.nan,
        )
    n_zero = int((factor == 0).sum())
    sparsity = n_zero / n
    density = (
        FactorDensity.SPARSE if sparsity >= _SPARSITY_THRESHOLD else FactorDensity.DENSE
    )
    reason = (
        f"sparsity ratio = {sparsity:.2f} "
        f"(threshold {_SPARSITY_THRESHOLD}): → {density.value.upper()}"
    )
    return density, reason, sparsity


def _detect_scope(raw: Any) -> tuple[FactorScope, str]:
    """COMMON if factor is constant per date across assets, else INDIVIDUAL."""
    n_assets = int(raw["asset_id"].n_unique())
    if n_assets <= 1:
        return (
            FactorScope.COMMON,
            f"n_assets = {n_assets}: scope axis trivially COMMON at N=1",
        )
    per_date_unique = raw.group_by("date").agg(
        pl.col("factor").n_unique().alias("n_unique_per_date")
    )
    is_broadcast = bool(
        (per_date_unique["n_unique_per_date"] == 1).all(),
    )
    decision = "COMMON" if is_broadcast else "INDIVIDUAL"
    return (
        FactorScope.COMMON if is_broadcast else FactorScope.INDIVIDUAL,
        f"factor varies across assets at given date: "
        f"{'NO' if is_broadcast else 'YES'} → {decision}",
    )


@dataclass(frozen=True, slots=True)
class DataProperties:
    """Inspected data properties driving cell dispatch.

    Carries the dispatch axes (``scope`` / ``density`` / ``structure``
    as typed enums), the human-readable rationale for each axis decision
    (``scope_reason`` / ``density_reason`` / ``structure_reason``), and
    the data-shape numerics the user typically wants next to them
    (``n_assets`` / ``n_periods`` / ``n_pairs`` / ``sparse_ratio``).
    Named ``Properties`` rather than ``Axes`` because the numeric fields
    are not axes — they are observations supporting the axis decisions.

    Each ``*_reason`` sits next to the enum it explains so detection
    verdict and rationale travel together as one value.

    Attributes:
        scope: Detected :class:`FactorScope`.
        scope_reason: Human-readable rationale for ``scope``.
        density: Detected :class:`FactorDensity`.
        density_reason: Human-readable rationale for ``density``.
        structure: Detected :class:`DataStructure` — ``TIMESERIES`` iff
            ``n_assets == 1`` (single-asset data), ``PANEL`` otherwise.
        structure_reason: Human-readable rationale for ``structure``.
        n_assets: Unique ``asset_id`` count under any-non-null union.
        n_periods: Unique ``date`` count under any-non-null union.
        n_pairs: Non-null ``(date, asset_id)`` factor observation
            count — the upper bound on usable sample size for any
            pair-counting metric (IC, rank-IC, FM cross-section).
            Equals ``data.height`` for a dense panel; smaller when
            the factor column has nulls.
        n_events: Non-zero ``factor`` observation count — the event
            sample size for event-driven metrics (CAAR, MFE/MAE,
            corrado-rank, event-quality). Non-null AND non-zero cells,
            matching those metrics' ``factor != 0`` event filter. For a
            dense continuous factor this is ~``n_pairs`` (the event
            axis only gates SPARSE-cell metrics). ``caar`` counts event
            *dates* rather than rows, so its pre-flight reads this as a
            loose upper bound; its in-body short-circuit on event dates
            stays authoritative.
        sparse_ratio: Zero-ratio in the ``factor`` column (denominator
            is non-null cell count). ``math.nan`` for an empty data.
    """

    scope: FactorScope
    scope_reason: str
    density: FactorDensity
    density_reason: str
    structure: DataStructure
    structure_reason: str
    n_assets: int
    n_periods: int
    n_pairs: int
    n_events: int
    sparse_ratio: float


@dataclass(frozen=True, slots=True)
class MetricApplicability:
    """Per-metric pre-flight verdict against one data's properties.

    Attributes:
        metric: The :class:`~factrix.metrics._base.MetricBase` subclass
            this verdict is about — the callable a caller would
            instantiate to run it. Lets consumers reach the class (and
            its ``compute`` / params) without going through
            :attr:`spec`.
        name: The metric's registry name (``metric.__name__`` ==
            ``spec.name``), surfaced directly so callers can key on it
            without reaching through :attr:`spec`.
        spec: The :class:`MetricSpec` being evaluated.
        usable: ``True`` iff the metric passes cell match AND every
            ``min_*`` floor on its :class:`SampleThreshold`. ``warn_*``
            violations do NOT flip ``usable`` to ``False`` — they
            attach a degraded :class:`Warning` to :attr:`warnings`.
        warnings: ``warn_*``-tier diagnostics (the metric will run
            but its inference is degraded). Empty when no warning
            threshold applies.
        blockers: Concrete reasons the metric is unusable
            (cell mismatch, ``min_*`` floor violation). Empty when
            ``usable`` is True.
    """

    metric: type[MetricBase]
    name: str
    spec: MetricSpec
    usable: bool
    warnings: list[Warning] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ICStage1Profile:
    """Pre-flight shape of the IC series after ``compute_ic``'s date filter."""

    n_periods: int
    min_assets_per_period: int
    max_assets_per_period: int


@dataclass(frozen=True, slots=True)
class _FMStage1Profile:
    """Pre-flight shape of the FM beta series after per-date OLS filters."""

    n_periods: int
    min_assets_per_period: int
    max_assets_per_period: int


def _default_constructible(metric: Any) -> bool:
    """True when the metric class can be instantiated with no arguments.

    Scalar-input utilities (e.g. ``breakeven_cost`` / ``net_spread``) declare
    required parameters (``turnover`` / ``forward_periods`` …) and consume
    pre-aggregated scalars rather than a data, so they are not part of the
    zero-config discovery -> :func:`factrix.evaluate` flow.
    """
    return all(
        f.default is not MISSING or f.default_factory is not MISSING
        for f in fields(metric)
    )


class MetricApplicabilityGroup(list["MetricApplicability"]):
    """A tier partition of :class:`MetricApplicability` verdicts.

    A ``list`` subclass — every list operation works — with discovery
    helpers layered on. Returned by :attr:`DataInspection.usable` /
    ``degraded`` / ``unusable``.
    """

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return MetricApplicabilityGroup(result)
        return result

    def __add__(self, other):
        return MetricApplicabilityGroup(super().__add__(list(other)))

    @property
    def names(self) -> list[str]:
        """The metric name of each verdict, in order."""
        return [m.name for m in self]

    def to_metrics_dict(self) -> dict[str, MetricBase]:
        """Normalise the group into ``{name: metric_instance}`` for ``evaluate``.

        The canonical discovery bridge: pick a tier (usually ``usable``) and
        feed its default-constructed instances straight to
        :func:`factrix.evaluate`::

            info = fx.inspect_data(data)
            results = fx.evaluate(data, metrics=info.usable.to_metrics_dict(),
                                  factor_cols=[...], forward_periods=5)

        Metrics whose class needs explicit construction arguments (the
        scalar-input utilities) are omitted — construct and add those by hand.
        """
        return {m.name: m.metric() for m in self if _default_constructible(m.metric)}


@dataclass(frozen=True, slots=True)
class DataInspection:
    """Result of :func:`inspect_data`.

    Pure data — no execution methods.

    Attributes:
        properties: :class:`DataProperties` with typed enum axes, the
            per-axis rationale strings, and shape numerics.
        metrics: Flat ``list[MetricApplicability]`` — one verdict
            per ``visibility=PUBLIC`` spec the inspector considered.
            Single source of truth; the :attr:`usable` /
            :attr:`degraded` / :attr:`unusable` properties expose it
            as a mutually exclusive partition.
        warnings: Data-level sample-shape diagnostics (NW HAC SE
            unreliable, cross-asset df low). ``source=None`` on
            every entry because these are data-level, not
            per-metric. Per-metric degraded warnings live inside
            each :class:`MetricApplicability`.
    """

    properties: DataProperties
    metrics: list[MetricApplicability]
    warnings: list[Warning] = field(default_factory=list)

    @property
    def usable(self) -> MetricApplicabilityGroup:
        """Metrics that are applicable AND carry no degraded warning.

        The production-safe set: every verdict here passed cell match
        and every ``min_*`` floor with zero ``warn_*`` violations.

        ``usable`` / :attr:`degraded` / :attr:`unusable` form a
        **mutually exclusive** partition of :attr:`metrics` — a
        verdict appears in exactly one. ``usable`` deliberately
        *excludes* degraded verdicts so it can be the single safe set
        a bulk discovery flow runs without re-filtering. The
        "usable but warned" verdicts live in :attr:`degraded`.

        Returns a :class:`MetricApplicabilityGroup` (a ``list`` subclass);
        ``.to_metrics_dict()`` normalises it straight into the
        ``metrics=`` argument of :func:`factrix.evaluate`.
        """
        return MetricApplicabilityGroup(
            m for m in self.metrics if m.usable and not m.warnings
        )

    @property
    def degraded(self) -> MetricApplicabilityGroup:
        """Applicable metrics that run but with degraded inference.

        ``usable=True`` yet at least one ``warn_*``-tier
        :class:`Warning` attached (e.g. NW HAC SE unreliable at short
        ``n_periods``). They produce a value, but the caller should
        read the warnings before trusting the inference. Disjoint from
        :attr:`usable` and :attr:`unusable`.
        """
        return MetricApplicabilityGroup(
            m for m in self.metrics if m.usable and m.warnings
        )

    @property
    def unusable(self) -> MetricApplicabilityGroup:
        """Metrics that cannot run on this data.

        ``usable=False`` — blocked by cell mismatch or a ``min_*``
        floor violation; :attr:`MetricApplicability.blockers` carries
        the concrete reasons. Disjoint from :attr:`usable` and
        :attr:`degraded`.
        """
        return MetricApplicabilityGroup(m for m in self.metrics if not m.usable)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict view.

        Layout (top-level keys, stable order):

        - ``properties``: ``{scope, density, structure, n_assets, n_periods,
          n_pairs, sparse_ratio}`` — enum fields rendered as their
          ``.value`` string; ``sparse_ratio`` ``NaN`` emitted as
          ``None``.
        - ``reasoning``: ``{scope, density, structure}``.
        - ``metrics``: list of per-spec dicts
          ``{name, cell, usable, warnings, blockers}`` — same row
          shape suits ``pl.from_dicts`` for cross-data audit.
        - ``warnings``: data-level ``[{code, source, message}, ...]``.

        Mirrors :meth:`factrix.EvaluationResult.to_dict` shape — single
        ``to_dict`` convention across the public result-type group so
        log / parquet sinks treat them uniformly.
        """
        d = self.properties
        return {
            "properties": {
                "scope": d.scope.value,
                "density": d.density.value,
                "structure": d.structure.value,
                "n_assets": d.n_assets,
                "n_periods": d.n_periods,
                "n_pairs": d.n_pairs,
                "sparse_ratio": (
                    None if math.isnan(d.sparse_ratio) else d.sparse_ratio
                ),
            },
            "reasoning": {
                "scope": d.scope_reason,
                "density": d.density_reason,
                "structure": d.structure_reason,
            },
            "metrics": [
                {
                    "name": m.spec.name,
                    "cell": m.spec.cell.raw,
                    "usable": m.usable,
                    "warnings": [
                        {
                            "code": w.code.value,
                            "source": w.source,
                            "message": w.message,
                        }
                        for w in m.warnings
                    ],
                    "blockers": list(m.blockers),
                }
                for m in self.metrics
            ],
            "warnings": [
                {
                    "code": w.code.value,
                    "source": w.source,
                    "message": w.message,
                }
                for w in self.warnings
            ],
        }

    def _repr_html_(self) -> str:
        d = self.properties
        header_rows = [
            ("scope", d.scope.value),
            ("density", d.density.value),
            ("structure", d.structure.value),
            ("n_assets", d.n_assets),
            ("n_periods", d.n_periods),
            ("n_pairs", d.n_pairs),
            (
                "sparse_ratio",
                f"{d.sparse_ratio:.3f}" if not math.isnan(d.sparse_ratio) else "nan",
            ),
        ]
        header_html = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in header_rows
        )

        reasoning_rows = "".join(
            f"<tr><th style='text-align:left'>{axis}</th>"
            f"<td>{html.escape(reason)}</td></tr>"
            for axis, reason in (
                ("scope", d.scope_reason),
                ("density", d.density_reason),
                ("structure", d.structure_reason),
            )
        )

        n_usable = sum(1 for m in self.metrics if m.usable)
        metric_rows = "".join(
            f"<tr><td>{html.escape(m.spec.name)}</td>"
            f"<td>{'usable' if m.usable else 'unusable'}</td>"
            f"<td>{html.escape(m.spec.cell.raw)}</td>"
            f"<td>{html.escape('; '.join(m.blockers) if m.blockers else '')}</td>"
            f"<td>{html.escape('; '.join(w.code.value for w in m.warnings))}</td></tr>"
            for m in self.metrics
        )
        metric_table = (
            f"<table><caption>metrics ({n_usable}/{len(self.metrics)} usable)"
            "</caption>"
            "<thead><tr><th>metric</th><th>status</th><th>cell</th>"
            "<th>blockers</th><th>warnings</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table>"
        )

        warnings_block = ""
        if self.warnings:
            w_rows = "".join(
                f"<tr><td>{html.escape(w.code.value)}</td>"
                f"<td>{html.escape(w.message)}</td></tr>"
                for w in self.warnings
            )
            warnings_block = (
                "<details open><summary>data-level warnings "
                f"({len(self.warnings)})</summary>"
                "<table><thead><tr><th>code</th><th>message</th>"
                "</tr></thead>"
                f"<tbody>{w_rows}</tbody></table></details>"
            )

        return (
            "<div class='factrix-data-inspection'>"
            "<table><caption>DataInspection — properties</caption>"
            f"<tbody>{header_html}</tbody></table>"
            "<table><caption>reasoning</caption>"
            f"<tbody>{reasoning_rows}</tbody></table>"
            f"{metric_table}{warnings_block}"
            "</div>"
        )


def inspect_data(data: Any, factor_cols: Sequence[str] | None = None) -> DataInspection:
    """Inspect data and return typed dispatch-axis + per-metric verdict.

    Pre-flight introspection: typed detection plus, for every
    ``role=SpecRole.METRIC`` :class:`MetricSpec` in the registry, a
    :class:`MetricApplicability` verdict combining (a) cell-match
    against the detected ``(scope, density, structure)`` and (b) the
    spec's optional :class:`SampleThreshold` against the data's
    ``n_periods`` / ``n_assets`` / ``n_pairs`` / ``n_events``.

    The ``n_events`` floor is pre-flighted against the non-zero factor
    count for event-driven metrics; a metric with a dynamic event floor
    (``caar``) is pre-flighted at its default config, and its in-body
    short-circuit on the actual run-time params stays authoritative.

    A third, content-based gate beyond cell and sample shape: a metric
    declaring ``requires_continuous_magnitude`` (e.g. ``event_ic``) is
    blocked on a discrete ±k signal (``|factor|`` constant across events,
    such as a ternary ``{-1, 0, +1}`` indicator), matching its run-time
    ``not_applicable_discrete_signal`` short-circuit.

    Multi-factor input: the inspected :class:`DataProperties` and every
    per-metric verdict are computed from the **first** factor column only
    (deterministic first-detected). When columns disagree on
    ``FactorDensity`` or ``FactorScope`` a ``CROSS_FACTOR_*_MISMATCH``
    data-level warning is emitted directing the caller to re-run
    ``inspect_data(data, factor_cols=[col])`` for a column-specific verdict.
    Other per-column properties (e.g. signal magnitude, ``n_events``) are
    likewise first-column-based; for a heterogeneous panel prefer the
    per-column call.

    Args:
        data: Long-format factor data with the canonical columns.
            The baseline columns ``date`` / ``asset_id`` /
            ``forward_return`` / ``price`` are reserved and not treated as factors.
        factor_cols: Optional list of factor columns to check. When
            ``None`` (default), auto-detects all columns except the
            reserved columns as factor candidates.

    Returns:
        :class:`DataInspection`.

    Examples:
        >>> import factrix as fx
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> info = fx.inspect_data(raw)
        >>> all(m.spec.cell.matches(info.properties.scope, info.properties.density, info.properties.structure) for m in info.usable)
        True
    """
    if isinstance(factor_cols, str):
        raise TypeError(
            f"factor_cols must be a list of column names, not a str; "
            f"did you mean factor_cols=[{factor_cols!r}]?"
        )
    if factor_cols is None:
        cols = [c for c in data.columns if c not in _INSPECT_RESERVED]
    else:
        cols = list(factor_cols)
    if not cols:
        raise ValueError(
            "No factor columns found. Pass factor_cols= explicitly or ensure "
            "data has columns beyond the reserved set "
            f"({sorted(_INSPECT_RESERVED)})."
        )

    first_col = cols[0]

    # Project raw data to "factor" column name to reuse standard detectors
    def _detect_col_density(col: str) -> tuple[FactorDensity, str, float]:
        temp = data.select("date", "asset_id", pl.col(col).alias("factor"))
        return _detect_density(temp)

    def _detect_col_scope(col: str) -> tuple[FactorScope, str]:
        temp = data.select("date", "asset_id", pl.col(col).alias("factor"))
        return _detect_scope(temp)

    # First factor column drives the returned properties (deterministic first-detected)
    density, density_reason, sparse_ratio = _detect_col_density(first_col)
    scope, scope_reason = _detect_col_scope(first_col)
    structure = _detect_structure(data)
    n_assets = int(data["asset_id"].n_unique())
    n_periods = int(data["date"].n_unique())
    n_pairs = int(data.drop_nulls(first_col).height)
    # Event sample: non-zero factor cells (nulls compare false, so excluded),
    # matching the ``factor != 0`` filter the event-driven metrics apply.
    n_events = int(data.filter(pl.col(first_col) != 0).height)
    ic_stage1_profile = _compute_ic_stage1_profile(data, first_col)
    fm_stage1_profile = _compute_fm_stage1_profile(data, first_col)

    structure_reason = (
        f"n_assets={n_assets} → "
        f"{'TIMESERIES (single-asset data)' if structure is DataStructure.TIMESERIES else 'PANEL'}"
    )

    properties = DataProperties(
        scope=scope,
        scope_reason=scope_reason,
        density=density,
        density_reason=density_reason,
        structure=structure,
        structure_reason=structure_reason,
        n_assets=n_assets,
        n_periods=n_periods,
        n_pairs=n_pairs,
        n_events=n_events,
        sparse_ratio=sparse_ratio,
    )

    data_warnings = _data_level_warnings(properties)

    # Cross-factor consistency checks
    if len(cols) > 1:
        densities = [density] + [_detect_col_density(c)[0] for c in cols[1:]]
        scopes = [scope] + [_detect_col_scope(c)[0] for c in cols[1:]]

        if len(set(densities)) > 1:
            detail = ", ".join(
                f"'{c}': {d.value}" for c, d in zip(cols, densities, strict=True)
            )
            data_warnings.append(
                Warning(
                    code=WarningCode.CROSS_FACTOR_DENSITY_MISMATCH,
                    source=None,
                    message=(
                        f"Factor columns carry inconsistent FactorDensity: {{{detail}}}. "
                        f"Metric applicability is based on '{first_col}'; call "
                        f"inspect_data(data, factor_cols=[<col>]) for a verdict "
                        f"specific to that column."
                    ),
                )
            )

        if len(set(scopes)) > 1:
            detail = ", ".join(
                f"'{c}': {s.value}" for c, s in zip(cols, scopes, strict=True)
            )
            data_warnings.append(
                Warning(
                    code=WarningCode.CROSS_FACTOR_SCOPE_MISMATCH,
                    source=None,
                    message=(
                        f"Factor columns carry inconsistent FactorScope: {{{detail}}}. "
                        f"Metric applicability is based on '{first_col}'; call "
                        f"inspect_data(data, factor_cols=[<col>]) for a verdict "
                        f"specific to that column."
                    ),
                )
            )

    # Signal magnitude is a content gate beyond cell/sample shape: a discrete
    # ±k factor (e.g. ternary {-1, 0, +1}) makes magnitude-dependent metrics
    # (event_ic) undefined. Computed on the first column — the verdict basis
    # for multi-factor input (see docstring) — using the same predicate the
    # metric short-circuits on at run time.
    from factrix.metrics._helpers import _event_signal_is_discrete

    signal_discrete = _event_signal_is_discrete(data, first_col)

    metrics = [
        _evaluate_applicability(
            spec,
            properties,
            signal_discrete,
            ic_stage1_profile=ic_stage1_profile,
            fm_stage1_profile=fm_stage1_profile,
        )
        for _, spec in public_specs()
    ]
    data_warnings.extend(_single_asset_event_warning(properties, metrics))

    return DataInspection(
        properties=properties,
        metrics=metrics,
        warnings=data_warnings,
    )


def _single_asset_event_warning(
    properties: DataProperties, metrics: list[MetricApplicability]
) -> list[Warning]:
    """Explain why a cross-sectional event metric is missing from `usable` on
    single-asset event data.

    On TIMESERIES + SPARSE data (n_assets=1) the event-axis metrics run over the
    event cross-section and are usable; only a metric that still needs the asset
    cross-section (``cell.structure=PANEL``, e.g. ``clustering_hhi``, whose
    same-date event clustering is degenerate when a single name has at most one
    event per date) stays blocked. Name those so their absence is explained
    rather than silent. Fires only when such a metric is actually present and
    unusable — dynamic on the verdicts, not on shape alone.
    """
    if not (
        properties.structure is DataStructure.TIMESERIES
        and properties.density is FactorDensity.SPARSE
    ):
        return []
    cross_sectional = sorted(
        m.name
        for m in metrics
        if m.spec.cell.density is FactorDensity.SPARSE
        and m.spec.cell.structure is DataStructure.PANEL
        and not m.usable
    )
    if not cross_sectional:
        return []
    names = ", ".join(cross_sectional)
    message = (
        f"Single-asset event data (n_assets=1): event-axis metrics run over the "
        f"event cross-section and are usable. Metrics that need the asset "
        f"cross-section ({names}) need n_assets>=2 and are unavailable here."
    )
    return [
        Warning(code=WarningCode.SINGLE_ASSET_EVENT_DATA, source=None, message=message)
    ]


def _evaluate_applicability(
    spec: MetricSpec,
    properties: DataProperties,
    signal_discrete: bool,
    ic_stage1_profile: _ICStage1Profile | None = None,
    fm_stage1_profile: _FMStage1Profile | None = None,
) -> MetricApplicability:
    from factrix.metrics._registry import REGISTRY

    blockers: list[str] = []
    warnings: list[Warning] = []
    uses_compute_ic = _requires_compute_ic(spec)
    uses_compute_fm_betas = _requires_compute_fm_betas(spec)

    if not spec.cell.matches(
        properties.scope, properties.density, properties.structure
    ):
        blockers.append(
            f"cell mismatch: spec={spec.cell.raw}, panel="
            f"({properties.scope.value.upper()}, "
            f"{properties.density.value.upper()}, "
            f"*, {properties.structure.value.upper()})"
        )

    if spec.requires_continuous_magnitude and signal_discrete:
        blockers.append(
            "discrete signal: |factor| has no magnitude variance over events "
            "(e.g. a ternary {-1, 0, +1} indicator); this metric needs a "
            "continuous magnitude and short-circuits "
            "not_applicable_discrete_signal at run time"
        )

    threshold_properties = properties
    skip_period_floor = False
    if uses_compute_ic and ic_stage1_profile is not None:
        if ic_stage1_profile.max_assets_per_period < MIN_IC_ASSETS_HARD:
            blockers.append(
                f"n_assets_per_period_max={ic_stage1_profile.max_assets_per_period} "
                f"< MIN_IC_ASSETS_HARD={MIN_IC_ASSETS_HARD}: compute_ic would drop every "
                "date before this metric runs"
            )
            skip_period_floor = True
        else:
            threshold_properties = replace(
                properties, n_periods=ic_stage1_profile.n_periods
            )
            if ic_stage1_profile.min_assets_per_period < MIN_IC_ASSETS_WARN:
                warnings.append(
                    Warning(
                        code=WarningCode.FEW_ASSETS,
                        source=spec.name,
                        message=(
                            "n_assets_per_period_min="
                            f"{ic_stage1_profile.min_assets_per_period} "
                            f"< MIN_IC_ASSETS_WARN={MIN_IC_ASSETS_WARN}: "
                            "IC cross-sections are computable but thin"
                        ),
                    )
                )

    if uses_compute_fm_betas and fm_stage1_profile is not None:
        if fm_stage1_profile.max_assets_per_period < MIN_FM_ASSETS_HARD:
            blockers.append(
                f"n_assets_per_period_max={fm_stage1_profile.max_assets_per_period} "
                f"< MIN_FM_ASSETS_HARD={MIN_FM_ASSETS_HARD}: compute_fm_betas would "
                "drop every date before this metric runs"
            )
            skip_period_floor = True
        elif fm_stage1_profile.n_periods == 0:
            blockers.append(
                "no non-degenerate per-date FM cross-section survived "
                "compute_fm_betas before this metric runs"
            )
            skip_period_floor = True
        else:
            threshold_properties = replace(
                threshold_properties, n_periods=fm_stage1_profile.n_periods
            )
            if fm_stage1_profile.min_assets_per_period < MIN_FM_ASSETS_WARN:
                warnings.append(
                    Warning(
                        code=WarningCode.FEW_ASSETS,
                        source=spec.name,
                        message=(
                            "n_assets_per_period_min="
                            f"{fm_stage1_profile.min_assets_per_period} "
                            f"< MIN_FM_ASSETS_WARN={MIN_FM_ASSETS_WARN}: "
                            "FM cross-sections are computable but thin"
                        ),
                    )
                )

    floor = spec.sample_threshold
    for av in floor.iter_verdicts(threshold_properties):
        if skip_period_floor and av.axis == "periods":
            continue
        if av.tier is Tier.UNUSABLE:
            blockers.append(f"n_{av.axis}={av.n} < min_{av.axis}={av.floor}")
        elif av.tier is Tier.DEGRADED:
            # The assets axis gates DEGRADED on the spec's warn_assets, but the
            # emitted warning is the inference-stage FEW_ASSETS code, keyed
            # on the global MIN_ASSETS_WARN constant — so it can be None here
            # even though the axis verdict is DEGRADED.
            if av.axis == "assets":
                code = cross_section_tier(properties.n_assets)
                if code is not None:
                    warnings.append(
                        Warning(code=code, source=spec.name, message=code.description)
                    )
            elif av.axis == "events":
                # Event-thin degraded tier maps to the event-specific code,
                # not the periods-flavored UNRELIABLE_SE_SHORT_PERIODS.
                warnings.append(
                    Warning(
                        code=WarningCode.FEW_EVENTS,
                        source=spec.name,
                        message=f"n_{av.axis}={av.n} < warn_{av.axis}={av.warn}: inference degraded",
                    )
                )
            else:
                warnings.append(
                    Warning(
                        code=WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
                        source=spec.name,
                        message=f"n_{av.axis}={av.n} < warn_{av.axis}={av.warn}: inference degraded",
                    )
                )

    return MetricApplicability(
        metric=REGISTRY[spec.name],
        name=spec.name,
        spec=spec,
        usable=not blockers,
        warnings=warnings,
        blockers=blockers,
    )


def _requires_compute_ic(spec: MetricSpec) -> bool:
    """True when a metric consumes the ``compute_ic`` stage-1 output."""
    return any(
        getattr(producer, "__name__", "") == "compute_ic"
        and getattr(producer, "__module__", "") == "factrix.metrics._primitives._ic"
        for producer in spec.requires.values()
    )


def _requires_compute_fm_betas(spec: MetricSpec) -> bool:
    """True when a metric consumes the ``compute_fm_betas`` stage-1 output."""
    return any(
        getattr(producer, "__name__", "") == "compute_fm_betas"
        and getattr(producer, "__module__", "")
        == "factrix.metrics._primitives._fm_betas"
        for producer in spec.requires.values()
    )


def _compute_ic_stage1_profile(data: Any, factor_col: str) -> _ICStage1Profile | None:
    """Mirror ``compute_ic``'s per-date pairwise-complete asset gate."""
    if "forward_return" not in data.columns:
        return None
    if data.is_empty():
        return _ICStage1Profile(
            n_periods=0,
            min_assets_per_period=0,
            max_assets_per_period=0,
        )

    valid_pair = (
        pl.col(factor_col).is_not_null() & pl.col("forward_return").is_not_null()
    )
    per_date = data.group_by("date").agg(valid_pair.sum().alias("n_assets"))
    if per_date.is_empty():
        return _ICStage1Profile(
            n_periods=0,
            min_assets_per_period=0,
            max_assets_per_period=0,
        )

    counts = per_date["n_assets"]
    max_assets = counts.max()
    surviving_counts = counts.filter(counts >= MIN_IC_ASSETS_HARD)
    min_surviving_assets = surviving_counts.min()
    return _ICStage1Profile(
        n_periods=int((counts >= MIN_IC_ASSETS_HARD).sum()),
        min_assets_per_period=(
            0 if min_surviving_assets is None else int(min_surviving_assets)
        ),
        max_assets_per_period=0 if max_assets is None else int(max_assets),
    )


def _compute_fm_stage1_profile(data: Any, factor_col: str) -> _FMStage1Profile | None:
    """Mirror ``compute_fm_betas``'s per-date OLS asset and variance gates."""
    if "forward_return" not in data.columns:
        return None
    if data.is_empty():
        return _FMStage1Profile(
            n_periods=0,
            min_assets_per_period=0,
            max_assets_per_period=0,
        )

    valid_pair = (
        pl.col(factor_col).is_not_null() & pl.col("forward_return").is_not_null()
    )
    per_date = data.group_by("date").agg(
        valid_pair.sum().alias("n_assets"),
        pl.col(factor_col).filter(valid_pair).var().alias("factor_var"),
    )
    if per_date.is_empty():
        return _FMStage1Profile(
            n_periods=0,
            min_assets_per_period=0,
            max_assets_per_period=0,
        )

    counts = per_date["n_assets"]
    max_assets = counts.max()
    surviving = per_date.filter(
        (pl.col("n_assets") >= MIN_FM_ASSETS_HARD) & (pl.col("factor_var") > 0)
    )
    surviving_counts = surviving["n_assets"]
    min_surviving_assets = surviving_counts.min()
    return _FMStage1Profile(
        n_periods=surviving.height,
        min_assets_per_period=(
            0 if min_surviving_assets is None else int(min_surviving_assets)
        ),
        max_assets_per_period=0 if max_assets is None else int(max_assets),
    )


def _data_level_warnings(properties: DataProperties) -> list[Warning]:
    """Sample-shape warnings the user can act on before running anything.

    Data-level (``source=None``) — not attributable to any single
    metric. Per-metric degraded warnings live on each
    :class:`MetricApplicability`.
    """
    from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN

    warnings: list[Warning] = []
    if (
        properties.structure is DataStructure.TIMESERIES
        and MIN_PERIODS_HARD <= properties.n_periods < MIN_PERIODS_WARN
    ):
        code = WarningCode.UNRELIABLE_SE_SHORT_PERIODS
        warnings.append(Warning(code=code, source=None, message=code.description))
    if properties.structure is DataStructure.PANEL:
        tier = cross_section_tier(properties.n_assets)
        if tier is not None:
            warnings.append(Warning(code=tier, source=None, message=tier.description))
    return warnings
