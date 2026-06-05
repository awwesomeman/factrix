"""factrix.metrics — Independent factor evaluation metrics.

All metrics return ``MetricResult`` and can be used standalone.

Grouping below follows the ``Scope × FactorDensity`` cells defined in
``factrix._axis`` — *not* ``DataStructure`` (which is a derived sample regime,
``PANEL`` for ``N ≥ 2`` and ``TIMESERIES`` for ``N == 1``). Every
cell can run in either DataStructure; the dispatch registry handles that.
Series diagnostics are axis-agnostic and operate on any ``(date, value)``
series produced upstream.

Individual × Continuous:
    compute_ic, ic, ic_ir,
    compute_spread_series, quantile_spread, quantile_spread_vw,
    compute_group_returns, monotonicity, top_concentration,
    turnover, notional_turnover, breakeven_cost, net_spread,
    compute_fm_betas, fm_beta, pooled_beta, beta_sign_consistency,
    spanning_alpha, greedy_forward_selection

Individual × Sparse (the ``Common × Sparse`` cell has its own
broadcast-dummy procedure but reuses these helper metrics — there is
no separate Common-sparse module set):
    compute_caar, caar, bmp_test, event_hit_rate, event_ic,
    compute_mfe_mae, mfe_mae_summary, profit_factor, event_skewness,
    compute_event_returns, event_around_return,
    signal_density, clustering_hhi, corrado_rank

Common × Continuous:
    compute_ts_betas, ts_beta, mean_r_squared,
    rolling_mean_beta, ts_beta_sign_consistency,
    ts_quantile_spread, ts_asymmetry

Series diagnostics — axis-agnostic on ``(date, value)``:
    hit_rate, ic_trend, oos_decay
"""

from factrix._metric_index import metric_spec, register
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics.caar import (
    bmp_test,
    caar,
)
from factrix.metrics.clustering_hhi import clustering_hhi
from factrix.metrics.concentration import top_concentration
from factrix.metrics.corrado_rank import corrado_rank
from factrix.metrics.event_horizon import (
    event_around_return,
)
from factrix.metrics.event_quality import (
    event_hit_rate,
    event_ic,
    event_skewness,
    profit_factor,
    signal_density,
)
from factrix.metrics.fm_beta import (
    beta_sign_consistency,
    fm_beta,
    pooled_beta,
)
from factrix.metrics.hit_rate import hit_rate
from factrix.metrics.ic import (
    ic,
    ic_ir,
    ic_newey_west,
)
from factrix.metrics.mfe_mae import (
    mfe_mae_summary,
)
from factrix.metrics.monotonicity import monotonicity
from factrix.metrics.oos_decay import oos_decay
from factrix.metrics.quantile import (
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
    rolling_mean_beta,
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
    "clustering_hhi",
    "rolling_mean_beta",
    "corrado_rank",
    "event_around_return",
    "event_hit_rate",
    "event_ic",
    "event_skewness",
    "fm_beta",
    "greedy_forward_selection",
    "hit_rate",
    "ic",
    "ic_ir",
    "ic_newey_west",
    "ic_trend",
    "mean_r_squared",
    "mfe_mae_summary",
    "monotonicity",
    "oos_decay",
    "net_spread",
    "notional_turnover",
    "pooled_beta",
    "profit_factor",
    "quantile_spread",
    "quantile_spread_vw",
    "signal_density",
    "spanning_alpha",
    "top_concentration",
    "ts_asymmetry",
    "ts_beta",
    "ts_beta_sign_consistency",
    "ts_quantile_spread",
    "turnover",
    # Third-party registration
    "metric_spec",
    "register",
    "MetricBase",
    "metric",
]
