"""factrix.metrics -- Independent factor evaluation metrics.

All metrics return ``MetricResult`` and can be used standalone.

Grouping below follows the ``Scope x FactorDensity`` cells defined in
``factrix._axis`` -- *not* ``DataStructure`` (which is a derived sample regime,
``PANEL`` for ``n_assets >= 2`` and ``TIMESERIES`` for ``n_assets == 1``). Metrics whose
registered cell pins ``DataStructure`` only run in that sample regime; the dispatch
registry rejects cell mismatches before execution.
Series diagnostics are axis-agnostic and operate on any ``(date, value)``
series produced upstream.

Individual x Continuous:
    compute_ic, ic, ic_ir,
    compute_spread_series, quantile_spread, quantile_spread_vw,
    compute_group_returns, monotonicity, top_concentration,
    rank_turnover, notional_turnover, breakeven_cost, net_spread,
    compute_fm_betas, fm_beta, pooled_beta, fm_beta_sign_consistency,
    spanning_alpha, greedy_forward_selection, k_spread

Individual x Sparse / Common x Sparse (scope-agnostic sparse event metrics;
there is no separate Common-sparse module set):
    compute_caar, caar, bmp_z, event_hit_rate, event_ic,
    compute_mfe_mae, mfe_mae, profit_factor, event_skewness,
    compute_event_returns, event_around_return,
    signal_density, clustering_hhi, corrado_rank

Common x Continuous:
    compute_common_betas, common_beta, common_beta_r_squared,
    compute_rolling_common_beta, common_beta_sign_consistency,
    common_beta_profile,
    common_quantile_spread, common_asymmetry

Single-asset x Continuous:
    predictive_beta -- TIMESERIES dense predictive regression with NW HAC

Series diagnostics -- axis-agnostic on ``(date, value)``:
    positive_rate, ic_trend, oos_decay

Scope-agnostic (run in either scope; ``cell`` scope is ``None``):
    directional_hit_rate -- small-N robust, Pesaran-Timmermann directional
    sibling of ``positive_rate`` (consumes a ``date, asset_id, factor,
    forward_return`` panel rather than a pre-aggregated series)

Individual x Dense x Panel diagnostics:
    directional_pair_accuracy -- descriptive small-N pairwise ordering
    accuracy across same-date asset pairs
"""

from factrix._metric_index import metric_spec, register
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics.caar import (
    bmp_z,
    caar,
)
from factrix.metrics.clustering_hhi import clustering_hhi
from factrix.metrics.common_asymmetry import common_asymmetry
from factrix.metrics.common_beta import (
    common_beta,
    common_beta_profile,
    common_beta_r_squared,
    common_beta_sign_consistency,
)
from factrix.metrics.common_quantile import common_quantile_spread
from factrix.metrics.concentration import top_concentration
from factrix.metrics.corrado_rank import corrado_rank
from factrix.metrics.directional_hit_rate import directional_hit_rate
from factrix.metrics.directional_pair_accuracy import directional_pair_accuracy
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
    fm_beta,
    fm_beta_sign_consistency,
    pooled_beta,
)
from factrix.metrics.ic import (
    ic,
    ic_ir,
)
from factrix.metrics.k_spread import k_spread
from factrix.metrics.mfe_mae import (
    mfe_mae,
)
from factrix.metrics.monotonicity import monotonicity
from factrix.metrics.oos_decay import oos_decay
from factrix.metrics.positive_rate import positive_rate
from factrix.metrics.predictive_beta import predictive_beta
from factrix.metrics.quantile import (
    quantile_spread,
    quantile_spread_vw,
)
from factrix.metrics.spanning import greedy_forward_selection, spanning_alpha
from factrix.metrics.tradability import (
    breakeven_cost,
    net_spread,
    notional_turnover,
    rank_turnover,
)
from factrix.metrics.trend import ic_trend

__all__ = [
    "fm_beta_sign_consistency",
    "bmp_z",
    "breakeven_cost",
    "caar",
    "clustering_hhi",
    "corrado_rank",
    "directional_hit_rate",
    "directional_pair_accuracy",
    "event_around_return",
    "event_hit_rate",
    "event_ic",
    "event_skewness",
    "fm_beta",
    "greedy_forward_selection",
    "positive_rate",
    "ic",
    "ic_ir",
    "ic_trend",
    "k_spread",
    "common_beta_r_squared",
    "mfe_mae",
    "monotonicity",
    "oos_decay",
    "net_spread",
    "notional_turnover",
    "pooled_beta",
    "profit_factor",
    "predictive_beta",
    "quantile_spread",
    "quantile_spread_vw",
    "signal_density",
    "spanning_alpha",
    "top_concentration",
    "common_asymmetry",
    "common_beta",
    "common_beta_profile",
    "common_beta_sign_consistency",
    "common_quantile_spread",
    "rank_turnover",
    # Third-party registration
    "metric_spec",
    "register",
    "MetricBase",
    "metric",
]
