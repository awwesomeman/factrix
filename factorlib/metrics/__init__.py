"""factorlib.metrics — Independent factor evaluation metrics.

All metrics return MetricOutput and can be used standalone.

Cross-Sectional:
    compute_ic, ic, ic_ir, hit_rate, ic_trend,
    multi_split_oos_decay, quantile_spread, monotonicity,
    q1_concentration, turnover, breakeven_cost, net_spread

Event Signal:
    compute_caar, caar, bmp_test, event_hit_rate, event_ic,
    compute_mfe_mae, mfe_mae_summary, profit_factor, event_skewness,
    clustering_diagnostic, corrado_rank_test

Macro Panel:
    compute_fm_betas, fama_macbeth, pooled_ols,
    beta_sign_consistency

Factor Attribution:
    spanning_alpha, greedy_forward_selection
"""

from factorlib.metrics.ic import (
    compute_ic,
    ic,
    ic_ir,
    regime_ic,
    multi_horizon_ic,
)
from factorlib.metrics.hit_rate import hit_rate
from factorlib.metrics.trend import ic_trend
from factorlib.metrics.oos import multi_split_oos_decay
from factorlib.metrics.quantile import (
    compute_spread_series,
    quantile_spread,
    quantile_spread_vw,
    compute_group_returns,
)
from factorlib.metrics.monotonicity import monotonicity
from factorlib.metrics.concentration import q1_concentration
from factorlib.metrics.tradability import turnover, breakeven_cost, net_spread
from factorlib.metrics.spanning import spanning_alpha, greedy_forward_selection
from factorlib.metrics.caar import (
    compute_caar,
    caar,
    bmp_test,
)
from factorlib.metrics.event_quality import (
    event_hit_rate,
    event_ic,
    profit_factor,
    event_skewness,
)
from factorlib.metrics.mfe_mae import (
    compute_mfe_mae,
    mfe_mae_summary,
)
from factorlib.metrics.clustering import clustering_diagnostic
from factorlib.metrics.corrado import corrado_rank_test
from factorlib.metrics.fama_macbeth import (
    compute_fm_betas,
    fama_macbeth,
    pooled_ols,
    beta_sign_consistency,
)
from factorlib.metrics.ts_beta import (
    compute_ts_betas,
    ts_beta,
    mean_r_squared,
    compute_rolling_mean_beta,
    ts_beta_sign_consistency,
)

__all__ = [
    "compute_ic", "ic", "ic_ir", "regime_ic", "multi_horizon_ic",
    "hit_rate", "ic_trend", "multi_split_oos_decay",
    "compute_spread_series", "quantile_spread", "quantile_spread_vw",
    "compute_group_returns", "monotonicity", "q1_concentration",
    "turnover", "breakeven_cost", "net_spread",
    "spanning_alpha", "greedy_forward_selection",
    "compute_caar", "caar", "bmp_test", "event_hit_rate", "event_ic",
    "compute_mfe_mae", "mfe_mae_summary", "profit_factor", "event_skewness",
    "clustering_diagnostic", "corrado_rank_test",
    "compute_fm_betas", "fama_macbeth", "pooled_ols",
    "beta_sign_consistency",
    "compute_ts_betas", "ts_beta", "mean_r_squared",
    "compute_rolling_mean_beta", "ts_beta_sign_consistency",
]
