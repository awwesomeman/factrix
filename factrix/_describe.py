"""v0.5 introspection helpers — `describe_analysis_modes` + `suggest_config`.

Both helpers reverse-query the registry SSOT (§4.4 A1) so the
"which cells exist" knowledge stays in one place. Plan §7.1 / §7.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import WarningCode
from factrix._registry import (
    _DISPATCH_REGISTRY,
    _SCOPE_COLLAPSED,
    _DispatchKey,
    _ScopeCollapsedSentinel,
)
from factrix._stats.constants import MIN_T_HARD, MIN_T_RELIABLE


# Sparsity threshold above which `factor` is treated as an event series.
_SPARSITY_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# describe_analysis_modes
# ---------------------------------------------------------------------------


def _user_facing_axis_tuples() -> list[tuple[FactorScope, Signal, Metric | None]]:
    """Distinct ``(scope, signal, metric)`` triples drawn from the registry.

    The sentinel-keyed entry expands to both ``INDIVIDUAL`` and
    ``COMMON`` because ``individual_sparse()`` and ``common_sparse()``
    are both legal user-facing constructions that converge at N=1
    (§5.4.1).
    """
    seen: set[tuple[FactorScope, Signal, Metric | None]] = set()
    for entry in _DISPATCH_REGISTRY.values():
        scope = entry.key.scope
        if isinstance(scope, _ScopeCollapsedSentinel):
            scopes: list[FactorScope] = [
                FactorScope.INDIVIDUAL, FactorScope.COMMON,
            ]
        else:
            scopes = [scope]
        for s in scopes:
            seen.add((s, entry.key.signal, entry.key.metric))

    def _sort_key(t: tuple[FactorScope, Signal, Metric | None]) -> tuple:
        return (
            t[0].value,
            t[1].value,
            t[2].value if t[2] is not None else "",
        )

    return sorted(seen, key=_sort_key)


def _entry_for(
    scope: FactorScope, signal: Signal, metric: Metric | None, mode: Mode,
) -> Any:
    """Return the registry entry routing this user-facing axis at ``mode``.

    Sparse signals at TIMESERIES route through the ``_SCOPE_COLLAPSED``
    sentinel (§5.4.1); other cells route on the user scope directly.
    """
    if signal is Signal.SPARSE and mode is Mode.TIMESERIES:
        key = _DispatchKey(_SCOPE_COLLAPSED, signal, metric, mode)
    else:
        key = _DispatchKey(scope, signal, metric, mode)
    return _DISPATCH_REGISTRY.get(key)


def _row_for_tuple(
    scope: FactorScope, signal: Signal, metric: Metric | None,
) -> dict[str, Any]:
    panel = _entry_for(scope, signal, metric, Mode.PANEL)
    timeseries = _entry_for(scope, signal, metric, Mode.TIMESERIES)
    return {
        "scope": scope.value,
        "signal": signal.value,
        "metric": metric.value if metric is not None else None,
        "mode_a_panel": (
            {
                "use_case": panel.canonical_use_case,
                "references": list(panel.references),
            }
            if panel is not None else None
        ),
        "mode_b_timeseries": (
            {
                "use_case": timeseries.canonical_use_case,
                "references": list(timeseries.references),
                "scope_collapsed": (
                    signal is Signal.SPARSE
                ),
            }
            if timeseries is not None
            else "raises ModeAxisError; see _FALLBACK_MAP"
        ),
    }


def _render_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        header = (
            f"({row['scope']}, {row['signal']}, {row['metric']})"
        )
        lines.append(f"Cell: {header}")
        panel = row["mode_a_panel"]
        if panel is None:
            lines.append("  Mode A (panel): not registered")
        else:
            lines.append(f"  Mode A (panel): {panel['use_case']}")
            if panel["references"]:
                lines.append(f"    refs: {'; '.join(panel['references'])}")
        ts = row["mode_b_timeseries"]
        if isinstance(ts, str):
            lines.append(f"  Mode B (timeseries): {ts}")
        else:
            note = (
                " — scope axis collapsed at N=1"
                if ts.get("scope_collapsed") else ""
            )
            lines.append(
                f"  Mode B (timeseries): {ts['use_case']}{note}",
            )
            if ts["references"]:
                lines.append(f"    refs: {'; '.join(ts['references'])}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def describe_analysis_modes(
    *, format: Literal["text", "json"] = "text",
) -> str | list[dict[str, Any]]:
    """Enumerate the legal analysis cells with Mode A/B routing notes.

    Iterates ``_DISPATCH_REGISTRY`` (single source of truth for "which
    cells exist") and groups by user-facing ``(scope, signal, metric)``
    so each row carries both Mode A (PANEL) and Mode B (TIMESERIES)
    information when registered. Plan §7.1.
    """
    rows = [_row_for_tuple(*t) for t in _user_facing_axis_tuples()]
    if format == "json":
        return rows
    return _render_text(rows)


# ---------------------------------------------------------------------------
# suggest_config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuggestConfigResult:
    """Auto-detected ``AnalysisConfig`` plus per-axis reasoning.

    ``reasoning`` is a structured dict with the four invariant keys
    ``scope`` / ``signal`` / ``metric`` / ``mode`` (plan §7.2 I4) so AI
    agents can pattern-match. ``warnings`` is a list of ``WarningCode``
    enums (I2) — pre-computed evaluate-time warnings the user can act
    on before running the suggested config.
    """

    suggested: AnalysisConfig
    reasoning: dict[str, str]
    warnings: list[WarningCode] = field(default_factory=list)


def _detect_signal(raw: Any) -> tuple[Signal, str]:
    """Sparsity ratio in ``factor`` ≥ 0.5 → SPARSE, else CONTINUOUS."""
    n = len(raw)
    if n == 0:
        return Signal.CONTINUOUS, "factor column empty: defaulting to CONTINUOUS"
    n_zero = int((raw["factor"] == 0).sum())
    sparsity = n_zero / n
    decision = "SPARSE" if sparsity >= _SPARSITY_THRESHOLD else "CONTINUOUS"
    return (
        Signal.SPARSE if decision == "SPARSE" else Signal.CONTINUOUS,
        f"sparsity ratio = {sparsity:.2f} "
        f"(threshold {_SPARSITY_THRESHOLD}): → {decision}",
    )


def _detect_scope(raw: Any) -> tuple[FactorScope, str]:
    """COMMON if factor is constant per date across assets, else INDIVIDUAL."""
    import polars as pl

    n_assets = int(raw["asset_id"].n_unique())
    if n_assets <= 1:
        return (
            FactorScope.COMMON,
            f"n_assets = {n_assets}: scope axis trivially COMMON at N=1",
        )
    per_date_unique = (
        raw.group_by("date")
        .agg(pl.col("factor").n_unique().alias("n_unique_per_date"))
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


def _build_suggested(
    scope: FactorScope, signal: Signal, *, forward_periods: int,
) -> AnalysisConfig:
    if signal is Signal.SPARSE:
        if scope is FactorScope.INDIVIDUAL:
            return AnalysisConfig.individual_sparse(
                forward_periods=forward_periods,
            )
        return AnalysisConfig.common_sparse(forward_periods=forward_periods)
    if scope is FactorScope.INDIVIDUAL:
        return AnalysisConfig.individual_continuous(
            metric=Metric.IC, forward_periods=forward_periods,
        )
    return AnalysisConfig.common_continuous(forward_periods=forward_periods)


def suggest_config(
    raw: Any, *, forward_periods: int = 5,
) -> SuggestConfigResult:
    """Inspect ``raw`` and propose an ``AnalysisConfig`` with reasoning.

    Plan §7.2: this is a *suggestion* — never auto-applied. The
    caller (or an AI agent) reads ``reasoning`` and ``warnings`` to
    decide whether to override.
    """
    signal, signal_reason = _detect_signal(raw)
    scope, scope_reason = _detect_scope(raw)

    n_assets = int(raw["asset_id"].n_unique())
    mode = Mode.TIMESERIES if n_assets <= 1 else Mode.PANEL
    mode_reason = (
        f"n_assets = {n_assets} detected → Mode "
        f"{'B (timeseries)' if mode is Mode.TIMESERIES else 'A (panel)'}"
    )

    suggested = _build_suggested(scope, signal, forward_periods=forward_periods)

    if suggested.metric is None:
        if scope is FactorScope.COMMON:
            metric_reason = "scope=COMMON: metric axis collapsed (no IC/FM choice)"
        else:
            metric_reason = "signal=SPARSE: metric axis collapsed (no IC/FM choice)"
    else:
        metric_reason = (
            f"scope=INDIVIDUAL × signal=CONTINUOUS: defaulting metric="
            f"{suggested.metric.value.upper()} (rank predictive ordering)"
        )

    reasoning = {
        "scope": scope_reason,
        "signal": signal_reason,
        "metric": metric_reason,
        "mode": mode_reason,
    }

    warnings: list[WarningCode] = []
    if mode is Mode.TIMESERIES:
        T = len(raw)
        if MIN_T_HARD <= T < MIN_T_RELIABLE:
            warnings.append(WarningCode.UNRELIABLE_SE_SHORT_SERIES)

    return SuggestConfigResult(
        suggested=suggested,
        reasoning=reasoning,
        warnings=warnings,
    )
