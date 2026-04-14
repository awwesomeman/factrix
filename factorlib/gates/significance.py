"""Gate 1: Statistical significance.

Passes if IC_IR t-stat OR Q1-Q5 spread t-stat >= threshold.
Default threshold = 2.0 (95% confidence).

Customize via ``functools.partial(significance_gate, threshold=3.0)``.
"""

from __future__ import annotations

from factorlib.gates._protocol import Artifacts, GateResult
from factorlib.tools.panel.ic import non_overlapping_ic_tstat
from factorlib.tools.series.significance import t_stat_from_array


def significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """Gate 1: IC_IR t-stat OR Q1-Q5 spread t-stat >= threshold.

    Either path passing is sufficient — some factors have strong IC
    but noisy spread (low N per quantile), or vice versa.

    Args:
        artifacts: Pre-computed pipeline artifacts.
        threshold: Minimum |t-stat| for significance (default 2.0).

    Returns:
        GateResult with detail containing both t-stats and which path(s) passed.
    """
    ic_tstat = non_overlapping_ic_tstat(
        artifacts.ic_series, artifacts.config.forward_periods,
    )
    spread_arr = artifacts.spread_series["spread"].drop_nulls().to_numpy()
    spread_tstat = t_stat_from_array(spread_arr)

    via: list[str] = []
    if abs(ic_tstat) >= threshold:
        via.append("IC_IR")
    if abs(spread_tstat) >= threshold:
        via.append("Q1-Q5_spread")

    return GateResult(
        name="significance",
        status="PASS" if via else "FAILED",
        detail={
            "ic_tstat": ic_tstat,
            "spread_tstat": spread_tstat,
            "threshold": threshold,
            "via": via,
        },
    )
