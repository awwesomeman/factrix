"""v0.5 introspection helpers — `describe_analysis_modes` + `suggest_config`.

Both helpers reverse-query the registry SSOT (§4.4 A1) so the
"which cells exist" knowledge stays in one place. Plan §7.1 / §7.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import WarningCode, cross_section_tier
from factrix._evaluate import _derive_mode
from factrix._registry import (
    _DISPATCH_REGISTRY,
    _DispatchKey,
    _ScopeCollapsedSentinel,
    _route_scope,
)
from factrix._stats.constants import (
    MIN_ASSETS_RELIABLE,
    MIN_PERIODS_HARD,
    MIN_PERIODS_RELIABLE,
)


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
    """Return the registry entry routing this user-facing axis at ``mode``."""
    routed_scope = _route_scope(scope, signal, mode)
    return _DISPATCH_REGISTRY.get(
        _DispatchKey(routed_scope, signal, metric, mode),
    )


def _row_for_tuple(
    scope: FactorScope, signal: Signal, metric: Metric | None,
) -> dict[str, Any]:
    panel = _entry_for(scope, signal, metric, Mode.PANEL)
    timeseries = _entry_for(scope, signal, metric, Mode.TIMESERIES)
    return {
        "scope": scope.value,
        "signal": signal.value,
        "metric": metric.value if metric is not None else None,
        "panel": (
            {
                "use_case": panel.canonical_use_case,
                "references": list(panel.references),
            }
            if panel is not None else None
        ),
        "timeseries": (
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


def _factory_call_for(
    scope: str, signal: str, metric: str | None,
) -> str:
    """Render the AnalysisConfig factory call for a `(scope, signal, metric)` row.

    UX-5 from review: ``describe_analysis_modes`` text output should
    answer "which factory do I call?" without forcing the reader to
    cross-reference the README factory table.
    """
    if scope == "individual" and signal == "continuous":
        m = "Metric.IC" if metric == "ic" else "Metric.FM"
        return f"AnalysisConfig.individual_continuous(metric={m})"
    if scope == "individual" and signal == "sparse":
        return "AnalysisConfig.individual_sparse()"
    if scope == "common" and signal == "continuous":
        return "AnalysisConfig.common_continuous()"
    if scope == "common" and signal == "sparse":
        return "AnalysisConfig.common_sparse()"
    return f"AnalysisConfig({scope=}, {signal=}, {metric=})"  # pragma: no cover


def _render_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        header = (
            f"({row['scope']}, {row['signal']}, {row['metric']})"
        )
        lines.append(f"Cell: {header}")
        lines.append(
            f"  Factory: {_factory_call_for(row['scope'], row['signal'], row['metric'])}",
        )
        panel = row["panel"]
        if panel is None:
            lines.append("  PANEL: not registered")
        else:
            lines.append(f"  PANEL: {panel['use_case']}")
            if panel["references"]:
                lines.append(f"    refs: {'; '.join(panel['references'])}")
        ts = row["timeseries"]
        if isinstance(ts, str):
            lines.append(f"  TIMESERIES: {ts}")
        else:
            # A-8 from review: name what the TIMESERIES null actually tests
            # rather than implying parity with PANEL's cross-asset null.
            # Sparse TIMESERIES genuinely collapses the scope axis (sentinel
            # routes both INDIVIDUAL and COMMON sparse to one procedure);
            # continuous TIMESERIES is a single-series β whose null is
            # distinct from the cross-asset E[β]=0 of PANEL.
            if ts.get("scope_collapsed"):
                note = " — scope axis collapsed at N=1"
            else:
                note = " — single-series test (null differs from PANEL)"
            lines.append(
                f"  TIMESERIES: {ts['use_case']}{note}",
            )
            if ts["references"]:
                lines.append(f"    refs: {'; '.join(ts['references'])}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def describe_analysis_modes(
    *, format: Literal["text", "json"] = "text",
) -> str | list[dict[str, Any]]:
    """Enumerate the legal analysis cells with PANEL / TIMESERIES routing notes.

    Iterates ``_DISPATCH_REGISTRY`` (single source of truth for "which
    cells exist") and groups by user-facing ``(scope, signal, metric)``
    so each row carries both ``PANEL`` and ``TIMESERIES`` information
    when registered. Plan §7.1.
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


def _detect_signal(raw: Any) -> tuple[Signal, str, bool]:
    """Sparsity ratio in ``factor`` ≥ 0.5 → SPARSE, else CONTINUOUS.

    The third return value is ``magnitude_dropped``: ``True`` iff the
    detected signal is SPARSE *and* the non-zero values are not strictly
    ternary {-1, +1}. Sparse procedures coerce via ``.sign()`` so any
    non-±1 magnitude is silently dropped — the flag lets ``suggest_config``
    surface ``WarningCode.SPARSE_MAGNITUDE_DROPPED``.
    """
    import polars as pl

    n = len(raw)
    if n == 0:
        return (
            Signal.CONTINUOUS,
            "factor column empty: defaulting to CONTINUOUS",
            False,
        )
    n_zero, all_ternary = raw.select(
        n_zero=(pl.col("factor") == 0).sum(),
        all_ternary=(
            (pl.col("factor") == 0) | (pl.col("factor").abs() == 1)
        ).all(),
    ).row(0)
    sparsity = n_zero / n
    signal = (
        Signal.SPARSE if sparsity >= _SPARSITY_THRESHOLD else Signal.CONTINUOUS
    )
    reason = (
        f"sparsity ratio = {sparsity:.2f} "
        f"(threshold {_SPARSITY_THRESHOLD}): → {signal.value.upper()}"
    )
    magnitude_dropped = signal is Signal.SPARSE and not bool(all_ternary)
    if magnitude_dropped:
        reason += " (non-±1 magnitudes present; .sign() coercion will drop them)"
    return signal, reason, magnitude_dropped


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
    signal, signal_reason, magnitude_dropped = _detect_signal(raw)
    scope, scope_reason = _detect_scope(raw)

    n_assets = int(raw["asset_id"].n_unique())
    mode = _derive_mode(raw)
    mode_reason = (
        f"n_assets = {n_assets} detected → "
        f"{'TIMESERIES' if mode is Mode.TIMESERIES else 'PANEL'}"
    )
    n_tier = cross_section_tier(n_assets) if mode is Mode.PANEL else None
    if n_tier is not None:
        mode_reason += (
            f" (n_assets < MIN_ASSETS_RELIABLE = {MIN_ASSETS_RELIABLE} → "
            f"cross-asset df low, see WarningCode.{n_tier.name})"
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
        n_periods = len(raw)
        if MIN_PERIODS_HARD <= n_periods < MIN_PERIODS_RELIABLE:
            warnings.append(WarningCode.UNRELIABLE_SE_SHORT_SERIES)
    if n_tier is not None:
        warnings.append(n_tier)
    if magnitude_dropped:
        warnings.append(WarningCode.SPARSE_MAGNITUDE_DROPPED)

    return SuggestConfigResult(
        suggested=suggested,
        reasoning=reasoning,
        warnings=warnings,
    )
