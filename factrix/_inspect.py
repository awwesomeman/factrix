"""``fx.inspect_panel`` — typed panel introspection with per-metric verdict (#443).

Two-stage applicability model:

1. **Cell match** — ``(scope, density, structure)`` axes on
   :class:`MetricSpec.cell` must agree with the inspected panel's
   :class:`PanelProperties`. DataStructure is integral to the match — a metric
   whose cell declares ``structure=DataStructure.PANEL`` (e.g. IC, which has no
   cross-section in TIMESERIES) is unusable when the panel is
   single-asset, not just "degraded".
2. **Sample floor** — :class:`MetricSpec.sample_floor` declares the
   panel-shape thresholds below which the metric short-circuits
   (``min_*``) or runs with a documented bias warning (``warn_*``).
   :func:`inspect_panel` evaluates these against
   :class:`PanelProperties`.

Each public spec receives one :class:`MetricApplicability` verdict
exposing ``usable`` / ``warnings`` / ``blockers``. The flat
``list[MetricApplicability]`` on :class:`PanelInspection.metrics` is
the single source of truth; the ``usable`` / ``degraded`` /
``unusable`` properties expose it as a mutually exclusive partition.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from factrix._axis import FactorScope, FactorDensity, DataStructure
from factrix._codes import WarningCode, cross_section_tier
from factrix._metric_index import MetricSpec, public_specs
from factrix._results import Warning

_SPARSITY_THRESHOLD: float = 0.5


def _detect_structure(panel: Any) -> DataStructure:
    """Return ``TIMESERIES`` if the panel has a single asset, else ``PANEL``."""
    return (
        DataStructure.TIMESERIES if panel["asset_id"].n_unique() <= 1 else DataStructure.PANEL
    )


def _detect_density(raw: Any) -> tuple[FactorDensity, str, float]:
    """Sparsity ratio in ``factor`` ≥ 0.5 → SPARSE, else DENSE.

    Returns ``(density, reason, sparsity)`` where ``sparsity`` is the
    zero-ratio in the factor column.
    """
    n = len(raw)
    if n == 0:
        return (
            FactorDensity.DENSE,
            "factor column empty: defaulting to DENSE",
            math.nan,
        )
    n_zero = int((raw["factor"] == 0).sum())
    sparsity = n_zero / n
    density = (
        FactorDensity.SPARSE
        if sparsity >= _SPARSITY_THRESHOLD
        else FactorDensity.DENSE
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
class PanelProperties:
    """Inspected panel properties driving cell dispatch.

    Carries both the dispatch axes (``scope`` / ``density`` / ``structure``
    as typed enums) and the panel-shape numerics the user typically
    wants next to them (``n_assets`` / ``n_periods`` / ``n_pairs`` /
    ``sparse_ratio``). Named ``Properties`` rather than ``Axes``
    because the numeric fields are not axes — they are observations
    supporting the axis decisions.

    Attributes:
        scope: Detected :class:`FactorScope`.
        density: Detected :class:`FactorDensity`.
        structure: Detected :class:`DataStructure` — ``TIMESERIES`` iff
            ``n_assets == 1`` (single-asset panel), ``PANEL`` otherwise.
        n_assets: Unique ``asset_id`` count under any-non-null union.
        n_periods: Unique ``date`` count under any-non-null union.
        n_pairs: Non-null ``(date, asset_id)`` factor observation
            count — the upper bound on usable sample size for any
            pair-counting metric (IC, rank-IC, FM cross-section).
            Equals ``panel.height`` for a dense panel; smaller when
            the factor column has nulls.
        sparse_ratio: Zero-ratio in the ``factor`` column (denominator
            is non-null cell count). ``math.nan`` for an empty panel.
    """

    scope: FactorScope
    density: FactorDensity
    structure: DataStructure
    n_assets: int
    n_periods: int
    n_pairs: int
    sparse_ratio: float


@dataclass(frozen=True, slots=True)
class PanelReasoning:
    """Per-axis human-readable rationale for the panel's detection.

    Three fields parallel the three axis enums on
    :class:`PanelProperties` (``scope`` / ``density`` / ``structure``).
    """

    scope_reason: str
    density_reason: str
    structure_reason: str


@dataclass(frozen=True, slots=True)
class MetricApplicability:
    """Per-metric pre-flight verdict against one panel's properties.

    Attributes:
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

    spec: MetricSpec
    usable: bool
    warnings: list[Warning] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PanelInspection:
    """Result of :func:`inspect_panel`.

    Pure data — no execution methods.

    Attributes:
        detected: :class:`PanelProperties` with typed enum axes and
            shape numerics.
        reasoning: :class:`PanelReasoning` carrying per-axis prose
            (scope / density / structure).
        metrics: Flat ``list[MetricApplicability]`` — one verdict
            per ``visibility=PUBLIC`` spec the inspector considered.
            Single source of truth; the :attr:`usable` /
            :attr:`degraded` / :attr:`unusable` properties expose it
            as a mutually exclusive partition.
        warnings: Panel-level sample-shape diagnostics (NW HAC SE
            unreliable, cross-asset df low). ``source=None`` on
            every entry because these are panel-level, not
            per-metric. Per-metric degraded warnings live inside
            each :class:`MetricApplicability`.
    """

    detected: PanelProperties
    reasoning: PanelReasoning
    metrics: list[MetricApplicability]
    warnings: list[Warning] = field(default_factory=list)

    @property
    def usable(self) -> list[MetricApplicability]:
        """Metrics that are applicable AND carry no degraded warning.

        The production-safe set: every verdict here passed cell match
        and every ``min_*`` floor with zero ``warn_*`` violations.

        ``usable`` / :attr:`degraded` / :attr:`unusable` form a
        **mutually exclusive** partition of :attr:`metrics` — a
        verdict appears in exactly one. ``usable`` deliberately
        *excludes* degraded verdicts so it can be the single safe set
        a bulk discovery flow runs without re-filtering. The
        "usable but warned" verdicts live in :attr:`degraded`.
        """
        return [m for m in self.metrics if m.usable and not m.warnings]

    @property
    def degraded(self) -> list[MetricApplicability]:
        """Applicable metrics that run but with degraded inference.

        ``usable=True`` yet at least one ``warn_*``-tier
        :class:`Warning` attached (e.g. NW HAC SE unreliable at short
        ``n_periods``). They produce a value, but the caller should
        read the warnings before trusting the inference. Disjoint from
        :attr:`usable` and :attr:`unusable`.
        """
        return [m for m in self.metrics if m.usable and m.warnings]

    @property
    def unusable(self) -> list[MetricApplicability]:
        """Metrics that cannot run on this panel.

        ``usable=False`` — blocked by cell mismatch or a ``min_*``
        floor violation; :attr:`MetricApplicability.blockers` carries
        the concrete reasons. Disjoint from :attr:`usable` and
        :attr:`degraded`.
        """
        return [m for m in self.metrics if not m.usable]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict view.

        Layout (top-level keys, stable order):

        - ``detected``: ``{scope, density, structure, n_assets, n_periods,
          n_pairs, sparse_ratio}`` — enum fields rendered as their
          ``.value`` string; ``sparse_ratio`` ``NaN`` emitted as
          ``None``.
        - ``reasoning``: ``{scope, density, structure}``.
        - ``metrics``: list of per-spec dicts
          ``{name, cell, usable, warnings, blockers}`` — same row
          shape suits ``pl.from_dicts`` for cross-panel audit.
        - ``warnings``: panel-level ``[{code, source, message}, ...]``.

        Mirrors :meth:`factrix.EvaluationResult.to_dict` shape — single
        ``to_dict`` convention across the public result-type group so
        log / parquet sinks treat them uniformly.
        """
        d = self.detected
        return {
            "detected": {
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
                "scope": self.reasoning.scope_reason,
                "density": self.reasoning.density_reason,
                "structure": self.reasoning.structure_reason,
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
        d = self.detected
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
                ("scope", self.reasoning.scope_reason),
                ("density", self.reasoning.density_reason),
                ("structure", self.reasoning.structure_reason),
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
                "<details open><summary>panel-level warnings "
                f"({len(self.warnings)})</summary>"
                "<table><thead><tr><th>code</th><th>message</th>"
                "</tr></thead>"
                f"<tbody>{w_rows}</tbody></table></details>"
            )

        return (
            "<div class='factrix-panel-inspection'>"
            "<table><caption>PanelInspection — detected</caption>"
            f"<tbody>{header_html}</tbody></table>"
            "<table><caption>reasoning</caption>"
            f"<tbody>{reasoning_rows}</tbody></table>"
            f"{metric_table}{warnings_block}"
            "</div>"
        )


def inspect_panel(panel: Any) -> PanelInspection:
    """Inspect a panel and return typed dispatch-axis + per-metric verdict.

    Pre-flight introspection: typed detection plus, for every
    ``visibility=PUBLIC`` :class:`MetricSpec` in the registry, a
    :class:`MetricApplicability` verdict combining (a) cell-match
    against the detected ``(scope, density, structure)`` and (b) the
    spec's optional :class:`SampleThreshold` against the panel's
    ``n_periods`` / ``n_assets``.

    Sample-floor checks against ``n_events`` (factor-column-dependent)
    are out of scope — those surface at evaluate time on the metric's
    own short-circuit path.

    Args:
        panel: Long-format panel with the canonical columns
            ``date`` / ``asset_id`` / ``factor``.

    Returns:
        :class:`PanelInspection`.

    Examples:
        >>> import factrix as fx
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> info = fx.inspect_panel(raw)
        >>> all(m.spec.cell.matches(info.detected.scope, info.detected.density, info.detected.structure) for m in info.usable)
        True
    """
    density, density_reason, sparse_ratio = _detect_density(panel)
    scope, scope_reason = _detect_scope(panel)
    structure = _detect_structure(panel)
    n_assets = int(panel["asset_id"].n_unique())
    n_periods = int(panel["date"].n_unique())
    n_pairs = int(panel.drop_nulls("factor").height)

    structure_reason = (
        f"n_assets={n_assets} → "
        f"{'TIMESERIES (single-asset panel)' if structure is DataStructure.TIMESERIES else 'PANEL'}"
    )

    properties = PanelProperties(
        scope=scope,
        density=density,
        structure=structure,
        n_assets=n_assets,
        n_periods=n_periods,
        n_pairs=n_pairs,
        sparse_ratio=sparse_ratio,
    )
    reasoning = PanelReasoning(
        scope_reason=scope_reason,
        density_reason=density_reason,
        structure_reason=structure_reason,
    )

    panel_warnings = _panel_level_warnings(properties)
    metrics = [_evaluate_applicability(spec, properties) for _, spec in public_specs()]

    return PanelInspection(
        detected=properties,
        reasoning=reasoning,
        metrics=metrics,
        warnings=panel_warnings,
    )


def _evaluate_applicability(
    spec: MetricSpec, properties: PanelProperties
) -> MetricApplicability:
    blockers: list[str] = []
    warnings: list[Warning] = []

    if not spec.cell.matches(properties.scope, properties.density, properties.structure):
        blockers.append(
            f"cell mismatch: spec={spec.cell.raw}, panel="
            f"({properties.scope.value.upper()}, "
            f"{properties.density.value.upper()}, "
            f"*, {properties.structure.value.upper()})"
        )

    floor = spec.sample_threshold
    for axis, n_actual in (
        ("periods", properties.n_periods),
        ("pairs", properties.n_pairs),
    ):
        min_v = getattr(floor, f"min_{axis}")
        warn_v = getattr(floor, f"warn_{axis}")
        if min_v is not None and n_actual < min_v:
            blockers.append(f"n_{axis}={n_actual} < min_{axis}={min_v}")
        elif warn_v is not None and n_actual < warn_v:
            warnings.append(
                Warning(
                    code=WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
                    source=spec.name,
                    message=f"n_{axis}={n_actual} < warn_{axis}={warn_v}: inference degraded",
                )
            )
    if floor.min_assets is not None and properties.n_assets < floor.min_assets:
        blockers.append(
            f"n_assets={properties.n_assets} < min_assets={floor.min_assets}"
        )
    elif floor.warn_assets is not None and properties.n_assets < floor.warn_assets:
        tier = cross_section_tier(properties.n_assets)
        if tier is not None:
            warnings.append(
                Warning(code=tier, source=spec.name, message=tier.description)
            )

    return MetricApplicability(
        spec=spec,
        usable=not blockers,
        warnings=warnings,
        blockers=blockers,
    )


def _panel_level_warnings(properties: PanelProperties) -> list[Warning]:
    """Sample-shape warnings the user can act on before running anything.

    Panel-level (``source=None``) — not attributable to any single
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
