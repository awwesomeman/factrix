"""``fx.inspect_panel`` — typed panel introspection with per-metric verdict (#443).

Two-stage applicability model:

1. **Cell match** — ``(scope, signal, mode)`` axes on
   :class:`MetricSpec.cell` must agree with the inspected panel's
   :class:`PanelProperties`. Mode is integral to the match — a metric
   whose cell declares ``mode=Mode.PANEL`` (e.g. IC, which has no
   cross-section in TIMESERIES) is unusable when the panel is
   single-asset, not just "degraded".
2. **Sample floor** — :class:`MetricSpec.sample_floor` declares the
   panel-shape thresholds below which the metric short-circuits
   (``min_*``) or runs with a documented bias warning (``warn_*``).
   :func:`inspect_panel` evaluates these against
   :class:`PanelProperties`.

Each public spec receives one :class:`MetricApplicability` verdict
exposing ``usable`` / ``warnings`` / ``blockers``. The flat
``list[MetricApplicability]`` on :class:`PanelInspection` is the
single source of truth; partitioning by ``.usable`` is a one-line
list comprehension at the call site.

``suggest_config`` retires as part of #438 unification; this module
shares its detection helpers (``_detect_signal`` / ``_detect_scope``)
via direct import from :mod:`factrix._describe` to avoid duplication
during the overlap window.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any

from factrix._axis import FactorScope, Mode, Signal
from factrix._codes import WarningCode, cross_section_tier
from factrix._describe import _detect_scope, _detect_signal
from factrix._evaluate import _derive_mode
from factrix._metric_index import MetricSpec, public_specs
from factrix._results import Warning


@dataclass(frozen=True, slots=True)
class PanelProperties:
    """Inspected panel properties driving cell dispatch.

    Carries both the dispatch axes (``scope`` / ``signal`` / ``mode``
    as typed enums) and the panel-shape numerics the user typically
    wants next to them (``n_assets`` / ``n_periods`` / ``sparsity``).
    Named ``Properties`` rather than ``Axes`` because the numeric
    fields are not axes — they are observations supporting the axis
    decisions.

    Attributes:
        scope: Detected :class:`FactorScope`.
        signal: Detected :class:`Signal`.
        mode: Derived :class:`Mode` — ``TIMESERIES`` iff
            ``n_assets == 1`` (single-asset panel), ``PANEL`` otherwise.
        n_assets: Unique ``asset_id`` count under any-non-null union.
        n_periods: Unique ``date`` count under any-non-null union.
        sparsity: Zero-ratio in the ``factor`` column.
            ``math.nan`` for an empty panel.
    """

    scope: FactorScope
    signal: Signal
    mode: Mode
    n_assets: int
    n_periods: int
    sparsity: float


@dataclass(frozen=True, slots=True)
class PanelReasoning:
    """Per-axis human-readable rationale for the panel's detection.

    Three fields parallel the three axis enums on
    :class:`PanelProperties` (``scope`` / ``signal`` / ``mode``).
    """

    scope_reason: str
    signal_reason: str
    mode_reason: str


@dataclass(frozen=True, slots=True)
class MetricApplicability:
    """Per-metric pre-flight verdict against one panel's properties.

    Attributes:
        spec: The :class:`MetricSpec` being evaluated.
        usable: ``True`` iff the metric passes cell match AND every
            ``min_*`` floor on its :class:`SampleFloor`. ``warn_*``
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
            (scope / signal / mode).
        metrics: Flat ``list[MetricApplicability]`` — one verdict
            per ``visibility=PUBLIC`` spec the inspector considered.
            Caller partitions via list comprehension:
            ``usable = [m for m in info.metrics if m.usable]``.
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

    def _repr_html_(self) -> str:
        d = self.detected
        header_rows = [
            ("scope", d.scope.value),
            ("signal", d.signal.value),
            ("mode", d.mode.value),
            ("n_assets", d.n_assets),
            ("n_periods", d.n_periods),
            (
                "sparsity",
                f"{d.sparsity:.3f}" if not math.isnan(d.sparsity) else "nan",
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
                ("signal", self.reasoning.signal_reason),
                ("mode", self.reasoning.mode_reason),
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
    against the detected ``(scope, signal, mode)`` and (b) the
    spec's optional :class:`SampleFloor` against the panel's
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
        >>> usable = [m for m in info.metrics if m.usable]
        >>> all(m.spec.cell.matches(info.detected.scope, info.detected.signal, info.detected.mode) for m in usable)
        True
    """
    signal, signal_reason, sparsity = _detect_signal(panel)
    scope, scope_reason = _detect_scope(panel)
    mode = _derive_mode(panel)
    n_assets = int(panel["asset_id"].n_unique())
    n_periods = int(panel["date"].n_unique())

    mode_reason = (
        f"n_assets={n_assets} → "
        f"{'TIMESERIES (single-asset panel)' if mode is Mode.TIMESERIES else 'PANEL'}"
    )

    properties = PanelProperties(
        scope=scope,
        signal=signal,
        mode=mode,
        n_assets=n_assets,
        n_periods=n_periods,
        sparsity=sparsity,
    )
    reasoning = PanelReasoning(
        scope_reason=scope_reason,
        signal_reason=signal_reason,
        mode_reason=mode_reason,
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

    if not spec.cell.matches(properties.scope, properties.signal, properties.mode):
        blockers.append(
            f"cell mismatch: spec={spec.cell.raw}, panel="
            f"({properties.scope.value.upper()}, "
            f"{properties.signal.value.upper()}, "
            f"*, {properties.mode.value.upper()})"
        )

    floor = spec.sample_floor
    if floor is not None:
        if floor.min_periods is not None and properties.n_periods < floor.min_periods:
            blockers.append(
                f"n_periods={properties.n_periods} < min_periods={floor.min_periods}"
            )
        elif (
            floor.warn_periods is not None and properties.n_periods < floor.warn_periods
        ):
            warnings.append(
                Warning(
                    code=WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
                    source=spec.name,
                    message=(
                        f"n_periods={properties.n_periods} < warn_periods="
                        f"{floor.warn_periods}: inference degraded"
                    ),
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
        properties.mode is Mode.TIMESERIES
        and MIN_PERIODS_HARD <= properties.n_periods < MIN_PERIODS_WARN
    ):
        code = WarningCode.UNRELIABLE_SE_SHORT_PERIODS
        warnings.append(Warning(code=code, source=None, message=code.description))
    if properties.mode is Mode.PANEL:
        tier = cross_section_tier(properties.n_assets)
        if tier is not None:
            warnings.append(Warning(code=tier, source=None, message=tier.description))
    return warnings
