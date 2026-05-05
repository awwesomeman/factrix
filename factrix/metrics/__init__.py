"""factrix.metrics — Independent factor evaluation metrics.

All metrics return ``MetricOutput`` and can be used standalone.

Grouping below follows the ``Scope × Signal`` cells defined in
``factrix._axis`` — *not* ``Mode`` (which is a derived sample regime,
``PANEL`` for ``N ≥ 2`` and ``TIMESERIES`` for ``N == 1``). Every
cell can run in either Mode; the dispatch registry handles that.
Series tools are axis-agnostic and operate on any ``(date, value)``
series produced upstream.

Individual × Continuous:
    compute_ic, ic, ic_ir, regime_ic, multi_horizon_ic,
    compute_spread_series, quantile_spread, quantile_spread_vw,
    compute_group_returns, monotonicity, top_concentration,
    turnover, notional_turnover, breakeven_cost, net_spread,
    compute_fm_betas, fama_macbeth, pooled_ols, beta_sign_consistency,
    spanning_alpha, greedy_forward_selection

Individual × Sparse (the ``Common × Sparse`` cell has its own
broadcast-dummy procedure but reuses these helper metrics — there is
no separate Common-sparse module set):
    compute_caar, caar, bmp_test, event_hit_rate, event_ic,
    compute_mfe_mae, mfe_mae_summary, profit_factor, event_skewness,
    compute_event_returns, event_around_return, multi_horizon_hit_rate,
    signal_density, clustering_diagnostic, corrado_rank_test

Common × Continuous:
    compute_ts_betas, ts_beta, mean_r_squared,
    compute_rolling_mean_beta, ts_beta_sign_consistency,
    ts_quantile_spread, ts_asymmetry

Series tools — axis-agnostic on ``(date, value)``:
    hit_rate, ic_trend, multi_split_oos_decay
"""

from factrix.metrics.caar import (
    bmp_test,
    caar,
    compute_caar,
)
from factrix.metrics.clustering import clustering_diagnostic
from factrix.metrics.concentration import top_concentration
from factrix.metrics.corrado import corrado_rank_test
from factrix.metrics.event_horizon import (
    compute_event_returns,
    event_around_return,
    multi_horizon_hit_rate,
)
from factrix.metrics.event_quality import (
    event_hit_rate,
    event_ic,
    event_skewness,
    profit_factor,
    signal_density,
)
from factrix.metrics.fama_macbeth import (
    beta_sign_consistency,
    compute_fm_betas,
    fama_macbeth,
    pooled_ols,
)
from factrix.metrics.hit_rate import hit_rate
from factrix.metrics.ic import (
    compute_ic,
    ic,
    ic_ir,
    multi_horizon_ic,
    regime_ic,
)
from factrix.metrics.mfe_mae import (
    compute_mfe_mae,
    mfe_mae_summary,
)
from factrix.metrics.monotonicity import monotonicity
from factrix.metrics.oos import multi_split_oos_decay
from factrix.metrics.quantile import (
    compute_group_returns,
    compute_spread_series,
    quantile_spread,
    quantile_spread_vw,
)
from factrix.metrics.spanning import greedy_forward_selection, spanning_alpha
from factrix.metrics.tradability import (
    breakeven_cost,
    net_spread,
    notional_turnover,
    turnover,
)
from factrix.metrics.trend import ic_trend
from factrix.metrics.ts_asymmetry import ts_asymmetry
from factrix.metrics.ts_beta import (
    compute_rolling_mean_beta,
    compute_ts_betas,
    mean_r_squared,
    ts_beta,
    ts_beta_sign_consistency,
)
from factrix.metrics.ts_quantile import ts_quantile_spread

__all__ = [
    "beta_sign_consistency",
    "bmp_test",
    "breakeven_cost",
    "caar",
    "clustering_diagnostic",
    "compute_caar",
    "compute_event_returns",
    "compute_fm_betas",
    "compute_group_returns",
    "compute_ic",
    "compute_mfe_mae",
    "compute_rolling_mean_beta",
    "compute_spread_series",
    "compute_ts_betas",
    "corrado_rank_test",
    "event_around_return",
    "event_hit_rate",
    "event_ic",
    "event_skewness",
    "fama_macbeth",
    "greedy_forward_selection",
    "hit_rate",
    "ic",
    "ic_ir",
    "ic_trend",
    "mean_r_squared",
    "mfe_mae_summary",
    "monotonicity",
    "multi_horizon_hit_rate",
    "multi_horizon_ic",
    "multi_split_oos_decay",
    "net_spread",
    "notional_turnover",
    "pooled_ols",
    "profit_factor",
    "quantile_spread",
    "quantile_spread_vw",
    "regime_ic",
    "signal_density",
    "spanning_alpha",
    "top_concentration",
    "ts_asymmetry",
    "ts_beta",
    "ts_beta_sign_consistency",
    "ts_quantile_spread",
    "turnover",
]
