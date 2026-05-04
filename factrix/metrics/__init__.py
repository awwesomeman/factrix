"""factrix.metrics — Independent factor evaluation metrics.

All metrics return MetricOutput and can be used standalone.

Cross-Sectional:
    compute_ic, ic, ic_ir, hit_rate, ic_trend,
    multi_split_oos_decay, quantile_spread, monotonicity,
    top_concentration, turnover, notional_turnover,
    breakeven_cost, net_spread

Event Signal:
    compute_caar, caar, bmp_test, event_hit_rate, event_ic,
    compute_mfe_mae, mfe_mae_summary, profit_factor, event_skewness,
    compute_event_returns, event_around_return, multi_horizon_hit_rate,
    signal_density, clustering_diagnostic, corrado_rank_test

Macro Panel:
    compute_fm_betas, fama_macbeth, pooled_ols,
    beta_sign_consistency

Time-Series / Standalone Diagnostic:
    compute_ts_betas, ts_beta, mean_r_squared,
    compute_rolling_mean_beta, ts_beta_sign_consistency,
    ts_quantile_spread, ts_asymmetry

    Single-series or per-asset diagnostics that supplement the four
    PANEL/TIMESERIES procedures registered in ``_DISPATCH_REGISTRY``.
    Useful for ts_beta-family applicability checks (linearity, symmetry,
    distinctness) before trusting a primary β estimate.

Factor Attribution:
    spanning_alpha, greedy_forward_selection
"""

from factrix.metrics.ic import (
    compute_ic,
    ic,
    ic_ir,
    regime_ic,
    multi_horizon_ic,
)
from factrix.metrics.hit_rate import hit_rate
from factrix.metrics.trend import ic_trend
from factrix.metrics.oos import multi_split_oos_decay
from factrix.metrics.quantile import (
    compute_spread_series,
    quantile_spread,
    quantile_spread_vw,
    compute_group_returns,
)
from factrix.metrics.monotonicity import monotonicity
from factrix.metrics.concentration import top_concentration
from factrix.metrics.tradability import (
    turnover, notional_turnover, breakeven_cost, net_spread,
)
from factrix.metrics.spanning import spanning_alpha, greedy_forward_selection
from factrix.metrics.caar import (
    compute_caar,
    caar,
    bmp_test,
)
from factrix.metrics.event_quality import (
    event_hit_rate,
    event_ic,
    profit_factor,
    event_skewness,
    signal_density,
)
from factrix.metrics.event_horizon import (
    compute_event_returns,
    event_around_return,
    multi_horizon_hit_rate,
)
from factrix.metrics.mfe_mae import (
    compute_mfe_mae,
    mfe_mae_summary,
)
from factrix.metrics.clustering import clustering_diagnostic
from factrix.metrics.corrado import corrado_rank_test
from factrix.metrics.fama_macbeth import (
    compute_fm_betas,
    fama_macbeth,
    pooled_ols,
    beta_sign_consistency,
)
from factrix.metrics.ts_beta import (
    compute_ts_betas,
    ts_beta,
    mean_r_squared,
    compute_rolling_mean_beta,
    ts_beta_sign_consistency,
)
from factrix.metrics.ts_quantile import ts_quantile_spread
from factrix.metrics.ts_asymmetry import ts_asymmetry

__all__ = [
    "compute_ic", "ic", "ic_ir", "regime_ic", "multi_horizon_ic",
    "hit_rate", "ic_trend", "multi_split_oos_decay",
    "compute_spread_series", "quantile_spread", "quantile_spread_vw",
    "compute_group_returns", "monotonicity", "top_concentration",
    "turnover", "notional_turnover", "breakeven_cost", "net_spread",
    "spanning_alpha", "greedy_forward_selection",
    "compute_caar", "caar", "bmp_test", "event_hit_rate", "event_ic",
    "compute_mfe_mae", "mfe_mae_summary", "profit_factor", "event_skewness",
    "compute_event_returns", "event_around_return", "multi_horizon_hit_rate",
    "signal_density", "clustering_diagnostic", "corrado_rank_test",
    "compute_fm_betas", "fama_macbeth", "pooled_ols",
    "beta_sign_consistency",
    "compute_ts_betas", "ts_beta", "mean_r_squared",
    "compute_rolling_mean_beta", "ts_beta_sign_consistency",
    "ts_quantile_spread", "ts_asymmetry",
]
