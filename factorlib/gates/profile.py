"""Factor profile computation.

After all gates pass, ``compute_profile`` runs the full suite of
Phase 1 tools to produce raw metric outputs — no scores, no mapping.

Reliability: IC, IC_IR, Hit_Rate, IC_Trend, Monotonicity, OOS_Decay.
Profitability: Q1-Q5 Spread, Long/Short Alpha, Turnover, Breakeven Cost,
    Net Spread, Q1 Concentration.
"""

from __future__ import annotations

from factorlib.gates._protocol import Artifacts, FactorProfile
from factorlib.tools._typing import MetricOutput
from factorlib.tools.panel import ic, quantile, monotonicity, concentration
from factorlib.tools.series import hit_rate, trend, oos
from factorlib.tools.cost import tradability


def compute_profile(artifacts: Artifacts) -> FactorProfile:
    """Compute the full factor profile from pre-built artifacts.

    Args:
        artifacts: Pre-computed pipeline artifacts (ic_series, spread_series, etc.).

    Returns:
        FactorProfile with reliability and profitability metric lists.
    """
    cfg = artifacts.config
    fp = cfg.forward_periods

    # --- Reliability ---
    ic_sig = ic.ic(artifacts.ic_series, forward_periods=fp)
    ic_ir = ic.ic_ir(artifacts.ic_series)
    hit = hit_rate.compute_hit_rate(artifacts.ic_values, forward_periods=fp)
    ic_trend = trend.theil_sen_slope(artifacts.ic_values)
    mono = monotonicity.compute_monotonicity(
        artifacts.prepared, forward_periods=fp, n_groups=cfg.n_groups,
    )

    oos_result = oos.multi_split_oos_decay(artifacts.ic_values)
    oos_metric = MetricOutput(
        name="OOS_Decay",
        value=oos_result.decay_ratio,
        metadata={
            "sign_flipped": oos_result.sign_flipped,
            "status": oos_result.status,
            "per_split": [
                {"is_ratio": s.is_ratio, "mean_is": s.mean_is,
                 "mean_oos": s.mean_oos, "decay_ratio": s.decay_ratio}
                for s in oos_result.per_split
            ],
        },
    )

    reliability = [ic_sig, ic_ir, hit, ic_trend, mono, oos_metric]

    # --- Profitability ---
    # WHY: reuse artifacts.spread_series to avoid recomputing quantile groups
    spread = quantile.quantile_spread(
        artifacts.prepared,
        forward_periods=fp,
        n_groups=cfg.n_groups,
        _precomputed_series=artifacts.spread_series,
    )
    ls_alpha = quantile.long_short_alpha(
        artifacts.prepared,
        forward_periods=fp,
        n_groups=cfg.n_groups,
        _precomputed_series=artifacts.spread_series,
    )
    turn = tradability.compute_turnover(artifacts.prepared)
    be = tradability.breakeven_cost(spread.value, turn.value)
    ns = tradability.net_spread(
        spread.value, turn.value, cfg.estimated_cost_bps,
    )
    conc = concentration.q1_concentration(
        artifacts.prepared, forward_periods=fp, q_top=cfg.q_top,
    )

    profitability = [spread, ls_alpha, turn, be, ns, conc]

    return FactorProfile(reliability=reliability, profitability=profitability)
