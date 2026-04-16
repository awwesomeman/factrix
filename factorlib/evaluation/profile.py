"""Factor profile computation.

After all gates pass, ``compute_profile`` runs the full suite of
Phase 1 tools to produce raw metric outputs — no scores, no mapping.

Reliability: IC, IC_IR, Hit_Rate, IC_Trend, Monotonicity, OOS_Decay.
Profitability: Q1-Q5 Spread (with long/short decomposition in metadata),
    Turnover, Breakeven Cost, Net Spread, Q1 Concentration.
"""

from __future__ import annotations

from factorlib.evaluation._protocol import Artifacts, FactorProfile
from factorlib._types import MetricOutput
from factorlib.metrics.ic import ic as ic_metric, ic_ir as ic_ir_metric
from factorlib.metrics.quantile import quantile_spread
from factorlib.metrics.monotonicity import monotonicity
from factorlib.metrics.concentration import q1_concentration
from factorlib.metrics.hit_rate import hit_rate
from factorlib.metrics.trend import ic_trend
from factorlib.metrics.oos import multi_split_oos_decay
from factorlib.metrics.tradability import turnover, breakeven_cost, net_spread


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
    ic_sig = ic_metric(artifacts.ic_series, forward_periods=fp)
    ic_ir = ic_ir_metric(artifacts.ic_series)
    hit = hit_rate(artifacts.ic_values, forward_periods=fp)
    ic_trn = ic_trend(artifacts.ic_values)
    mono = monotonicity(
        artifacts.prepared, forward_periods=fp, n_groups=cfg.n_groups,
    )

    oos_result = multi_split_oos_decay(artifacts.ic_values)
    oos_metric = MetricOutput(
        name="oos_decay",
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

    reliability = [ic_sig, ic_ir, hit, ic_trn, mono, oos_metric]

    # --- Profitability ---
    spread = quantile_spread(
        artifacts.prepared,
        forward_periods=fp,
        n_groups=cfg.n_groups,
        _precomputed_series=artifacts.spread_series,
    )
    turn = turnover(artifacts.prepared)
    be = breakeven_cost(spread.value, turn.value)
    ns = net_spread(
        spread.value, turn.value, cfg.estimated_cost_bps,
    )
    conc = q1_concentration(
        artifacts.prepared, forward_periods=fp, q_top=cfg.q_top,
    )

    profitability = [spread, turn, be, ns, conc]

    return FactorProfile(reliability=reliability, profitability=profitability)
