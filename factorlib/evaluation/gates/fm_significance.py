"""Gate: Fama-MacBeth significance for macro_panel.

Passes if FM β t-stat OR Pooled β t-stat >= threshold.
"""

from __future__ import annotations

from factorlib.evaluation._protocol import Artifacts, GateResult
from factorlib.metrics.fama_macbeth import fama_macbeth, pooled_ols


def fm_significance_gate(
    artifacts: Artifacts,
    *,
    threshold: float = 2.0,
) -> GateResult:
    """FM β t-stat OR Pooled β t-stat >= threshold.

    Either estimator passing is sufficient (robustness: both should
    agree in direction, but significance from either counts).
    """
    beta_series = artifacts.get("beta_series")
    fm = fama_macbeth(beta_series)
    pooled = pooled_ols(artifacts.prepared)

    fm_t = fm.stat or 0.0
    pooled_t = pooled.stat or 0.0

    via: list[str] = []
    if abs(fm_t) >= threshold:
        via.append("FM_beta")
    if abs(pooled_t) >= threshold:
        via.append("Pooled_beta")

    return GateResult(
        name="fm_significance",
        status="PASS" if via else "FAILED",
        detail={
            "fm_tstat": fm_t,
            "pooled_tstat": pooled_t,
            "threshold": threshold,
            "via": via,
        },
    )
