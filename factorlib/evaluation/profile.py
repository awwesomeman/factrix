"""Factor profile computation.

After all gates pass, ``compute_profile`` runs the full suite of
metrics to produce raw MetricOutput values — no scores, no mapping.

Cross-sectional metrics:
    IC, IC_IR, Hit_Rate, IC_Trend, Monotonicity, OOS_Decay,
    Q1-Q5 Spread, Turnover, Breakeven Cost, Net Spread, Q1 Concentration.
"""

from __future__ import annotations

from factorlib.config import BaseConfig, CrossSectionalConfig
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
    """Compute the full factor profile from pre-built artifacts."""
    match artifacts.config:
        case CrossSectionalConfig():
            return _cs_profile(artifacts, artifacts.config)
        case _:
            ft = type(artifacts.config).factor_type
            raise NotImplementedError(
                f"compute_profile not yet implemented for {ft}"
            )


def _cs_profile(
    artifacts: Artifacts, config: CrossSectionalConfig,
) -> FactorProfile:
    fp = config.forward_periods
    ic_series = artifacts.get("ic_series")
    ic_values = artifacts.get("ic_values")
    spread_series = artifacts.get("spread_series")

    ic_sig = ic_metric(ic_series, forward_periods=fp)
    ic_ir = ic_ir_metric(ic_series)
    hit = hit_rate(ic_values, forward_periods=fp)
    ic_trn = ic_trend(ic_values)
    mono = monotonicity(
        artifacts.prepared, forward_periods=fp, n_groups=config.n_groups,
    )

    oos_result = multi_split_oos_decay(ic_values)
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

    spread = quantile_spread(
        artifacts.prepared,
        forward_periods=fp,
        n_groups=config.n_groups,
        _precomputed_series=spread_series,
    )
    turn = turnover(artifacts.prepared)
    be = breakeven_cost(spread.value, turn.value)
    ns = net_spread(
        spread.value, turn.value, config.estimated_cost_bps,
    )
    conc = q1_concentration(
        artifacts.prepared, forward_periods=fp, q_top=config.q_top,
    )

    return FactorProfile(
        metrics=[ic_sig, ic_ir, hit, ic_trn, mono, oos_metric,
                 spread, turn, be, ns, conc],
    )
