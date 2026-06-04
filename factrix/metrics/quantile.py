"""Quantile analysis for cross-sectional panels.

All spread series are time-indexed (``date, value``) and can be fed
into any ``series/`` tool.

Notes:
    **Pipeline.** Per-date long-short spread on quantile groups
    (cross-section step), then non-overlapping t on the spread series.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Spread series, long/short alpha decomposition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    SEMethod,
    TestMethod,
)
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat, _p_value_from_t, _significance_marker
from factrix._types import (
    DDOF,
    MIN_PORTFOLIO_PERIODS_HARD,
)
from factrix.metrics import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups,
    _compute_tie_ratio,
    _lag_within_asset,
    _sample_non_overlapping,
    _short_circuit_output,
    _warn_high_tie_ratio,
)
from factrix.metrics._primitives import (
    compute_group_returns,
    compute_spread_series,
)

__all__ = [  # noqa: RUF022 (teaching order, see #322 SSOT note)
    "compute_spread_series",
    "compute_group_returns",
    "quantile_spread",
    "quantile_spread_vw",
]

_Q_CELL = cell(
    FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
)


@metric(
    cell=_Q_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.OLS,
    batchable=True,
)
def quantile_spread(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_cols: Sequence[str] = ("factor",),
    tie_policy: str = "ordinal",
    *,
    _precomputed_series: dict[str, pl.DataFrame] | None = None,
) -> dict[str, MetricResult]:
    """long-short spread (per-period mean).

    Args:
        _precomputed_series: If provided, skip recomputing ``compute_spread_series``.
        tie_policy: Bucketing tie-break policy, see ``_assign_quantile_groups``.
            When ``_precomputed_series`` is passed, this only affects the
            ``tie_ratio`` diagnostic — the series itself was already built.

    Returns:
        MetricResult with per-period mean spread, t-stat from non-overlapping periods.

    Notes:
        ``t = mean(spread) / (std(spread) / sqrt(n))`` on the non-overlap
        spread series. H0: ``E[spread] = 0``. Long/short alpha decomposition
        runs the same t-test on ``top_return - universe_return`` and
        ``universe_return - bottom_return`` so callers can attribute the
        spread to long-side vs short-side excess.

        factrix performs the t-test on the non-overlap series rather than
        applying Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) on an overlapping series; the two approaches are
        sibling routes — overlap variants live alongside ``ic_newey_west``.

    References:
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: overlapping-return
        autocorrelation, motivating the non-overlap stride.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import quantile_spread
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = quantile_spread(panel, forward_periods=5, n_groups=5)
        >>> result["factor"].name == ""
        True
    """
    cols = list(factor_cols)
    if not cols:
        raise ValueError("factor_cols must be non-empty")
    if _precomputed_series is not None and set(_precomputed_series) != set(cols):
        raise ValueError(
            "_precomputed_series keys must match factor_cols "
            f"(got {sorted(_precomputed_series)} vs {sorted(cols)})"
        )

    # Sample once across all factors; bucketing tie_ratio is computed
    # on the sampled subset (what bucketing actually sees) rather than
    # the full panel — ~N/forward_periods smaller scan.
    sampled = _sample_non_overlapping(df, forward_periods)
    series_by_factor = (
        _precomputed_series
        if _precomputed_series is not None
        else compute_spread_series(
            df,
            forward_periods,
            n_groups,
            factor_cols=cols,
            tie_policy=tie_policy,
        )
    )
    return {
        f: _quantile_spread_from_series(
            series=series_by_factor[f],
            sampled=sampled,
            factor_col=f,
            tie_policy=tie_policy,
        )
        for f in cols
    }


def _quantile_spread_from_series(
    *,
    series: pl.DataFrame,
    sampled: pl.DataFrame,
    factor_col: str,
    tie_policy: str,
) -> MetricResult:
    """Per-factor t-test pipeline shared by single and batch paths.

    ``sampled`` is the (already non-overlap-sampled) panel; ``series``
    is this factor's spread DataFrame. Splitting this out lets the
    batch path share the sample step across every factor while the
    single-factor path stays a one-liner.
    """
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread", tie_policy)
    spread_vals = series["spread"].drop_nulls()
    n = len(spread_vals)
    if n < MIN_PORTFOLIO_PERIODS_HARD:
        return _short_circuit_output(
            "quantile_spread",
            "insufficient_portfolio_periods",
            n_obs=n,
            min_required=MIN_PORTFOLIO_PERIODS_HARD,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
        )

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = _calc_t_stat(mean_spread, std_spread, n)

    p = _p_value_from_t(t, n)

    # Long/short decomposition (spread = long_alpha + short_alpha)
    long_excess = (series["top_return"] - series["universe_return"]).drop_nulls()
    short_excess = (series["universe_return"] - series["bottom_return"]).drop_nulls()

    long_arr = long_excess.to_numpy()
    short_arr = short_excess.to_numpy()

    mean_long = float(np.mean(long_arr))
    std_long = float(np.std(long_arr, ddof=DDOF))
    t_long = _calc_t_stat(mean_long, std_long, len(long_arr))
    p_long = _p_value_from_t(t_long, len(long_arr))

    mean_short = float(np.mean(short_arr))
    std_short = float(np.std(short_arr, ddof=DDOF))
    t_short = _calc_t_stat(mean_short, std_short, len(short_arr))
    p_short = _p_value_from_t(t_short, len(short_arr))

    return MetricResult(
        p=p,
        value=mean_spread,
        stat=t,
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "non-overlapping t-test",
            "long_alpha": mean_long,
            "short_alpha": mean_short,
            "long_stat": t_long,
            "long_p_value": p_long,
            "short_stat": t_short,
            "short_p_value": p_short,
            "short_significance": _significance_marker(p_short),
            "tie_ratio": tie_ratio,
            "tie_policy": tie_policy,
        },
    )


@metric(
    cell=_Q_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.OLS,
)
def quantile_spread_vw(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_col: str = "market_cap",
    tie_policy: str = "ordinal",
    lag_weights: bool = True,
) -> MetricResult:
    r"""Value-weighted long-short spread — alpha concentration diagnostic.

    Formula (per non-overlapping date $t$):

    $$
    \begin{aligned}
    \text{vw}_b[t] &= \frac{\sum_{i \in b} w_{i,t-1} \cdot \text{return}_{i, t \to t+h}}{\sum_{i \in b} w_{i,t-1}}, \quad b \in \{\text{bottom}, \text{top}\} \\
    \text{spread}[t] &= \text{vw}_{\text{top}}[t] - \text{vw}_{\text{bottom}}[t] \\
    \text{value} &= \mathrm{mean}_t\, \text{spread}[t], \quad t = \sqrt{n} \cdot \text{value} / \mathrm{std}(\text{spread}), \quad \text{DDOF}=1
    \end{aligned}
    $$

    Weights are **lagged by one sampled period per asset** by default
    (``lag_weights=True``): a portfolio rebalanced at date t uses the
    market-cap observed at the previous rebalance, not at t. Pairing
    contemporaneous ``market_cap[t]`` with ``forward_return[t→t+h]`` is
    a classic look-ahead trap — market cap measured on date t embeds
    news that the t→t+h return has not yet realized.

    Pass ``lag_weights=False`` only when the caller has **already**
    supplied a lagged weight column (e.g., prior-month-end cap) and
    wants the function to treat it as observed at t.

    Compare with equal-weighted ``quantile_spread``: if VW spread much
    smaller (e.g., < 1/3 of EW), the alpha is driven by small-cap assets
    and may not survive capacity / liquidity constraints.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return,
            market_cap`` (or whatever ``weight_col`` names).
        weight_col: Column for value weighting (default ``market_cap``).
        lag_weights: When True (default), shift ``weight_col`` by 1
            period per asset (on the non-overlap-sampled frame) before
            weighting. When False, use weights as supplied.

    Returns:
        MetricResult with per-period mean VW spread, t-stat, and p-value.
        Short-circuits if ``weight_col`` is missing or post-sampling n <
        ``MIN_PORTFOLIO_PERIODS_HARD``.

    Notes:
        Per non-overlapping date ``t``, per bucket ``b in {bot, top}``::

            vw_b[t] = sum_{i in b} w[i, t-1] * return[i, t -> t+h]
                      / sum_{i in b} w[i, t-1]
            spread[t] = vw_top[t] - vw_bot[t]
            value = mean_t spread[t];  t = sqrt(n) * value / std(spread)

        factrix lags weights by one **sampled** period per asset by default
        (not one raw bar) so the lag aligns with the rebalance stride;
        contemporaneous ``weight × forward_return`` would embed look-ahead
        bias from market-cap moves that the forward return has not yet
        realized.

    References:
        [Hou-Xue-Zhang (2020)][hou-xue-zhang-2020]: ~65% of anomalies
        fail $|t| \geq 1.96$ once microcaps are mitigated via NYSE
        breakpoints and value weighting jointly.

    Examples:
        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import quantile_spread_vw
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... ).with_columns(pl.lit(1e6).alias("market_cap"))
        >>> result = quantile_spread_vw(panel, forward_periods=5, n_groups=5)
        >>> result.name == ""
        True
    """
    if weight_col not in df.columns:
        return _short_circuit_output(
            "quantile_spread_vw",
            "no_weight_column",
            missing_column=weight_col,
        )

    sampled = _sample_non_overlapping(df, forward_periods)
    if lag_weights:
        sampled = _lag_within_asset(sampled, weight_col)
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread_vw", tie_policy)

    grouped = _assign_quantile_groups(
        sampled,
        factor_col,
        n_groups,
        tie_policy=tie_policy,
    )

    top_group = n_groups - 1
    bottom_group = 0

    # WHY: per-date weighted mean for top and bottom buckets
    vw_series = (
        grouped.with_columns(
            (pl.col(return_col) * pl.col(weight_col)).alias("_wr"),
        )
        .group_by("date")
        .agg(
            (
                pl.col("_wr").filter(pl.col("_group") == top_group).sum()
                / pl.col(weight_col).filter(pl.col("_group") == top_group).sum()
            ).alias("top_return_vw"),
            (
                pl.col("_wr").filter(pl.col("_group") == bottom_group).sum()
                / pl.col(weight_col).filter(pl.col("_group") == bottom_group).sum()
            ).alias("bottom_return_vw"),
        )
        .with_columns(
            (pl.col("top_return_vw") - pl.col("bottom_return_vw")).alias("spread_vw"),
        )
        .sort("date")
    )

    spread_vals = vw_series["spread_vw"].drop_nulls()
    n = len(spread_vals)
    if n < MIN_PORTFOLIO_PERIODS_HARD:
        return _short_circuit_output(
            "quantile_spread_vw",
            "insufficient_portfolio_periods",
            n_obs=n,
            min_required=MIN_PORTFOLIO_PERIODS_HARD,
            tie_ratio=tie_ratio,
            tie_policy=tie_policy,
        )

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = _calc_t_stat(mean_spread, std_spread, n)

    p = _p_value_from_t(t, n)
    return MetricResult(
        p=p,
        value=mean_spread,
        stat=t,
        metadata={
            "n_periods": n,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "tie_ratio": tie_ratio,
            "tie_policy": tie_policy,
            "weights_lagged": lag_weights,
        },
    )
