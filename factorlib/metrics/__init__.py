"""factorlib.metrics — Independent factor evaluation metrics.

All metrics return MetricOutput and can be used standalone.

Signal Quality:
    compute_ic, ic, ic_ir, hit_rate, ic_trend,
    multi_split_oos_decay

Portfolio Performance:
    quantile_spread, monotonicity, q1_concentration,
    turnover, breakeven_cost, net_spread

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

__all__ = [
    "compute_ic", "ic", "ic_ir", "regime_ic", "multi_horizon_ic",
    "hit_rate", "ic_trend", "multi_split_oos_decay",
    "compute_spread_series", "quantile_spread", "quantile_spread_vw",
    "compute_group_returns", "monotonicity", "q1_concentration",
    "turnover", "breakeven_cost", "net_spread",
    "spanning_alpha", "greedy_forward_selection",
]
