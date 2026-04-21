"""Quantile analysis for cross-sectional panels.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
Output: spread series, long/short alpha decomposition.

All spread series are time-indexed (``date, value``) and can be fed
into any ``series/`` tool.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from factorlib._types import (
    DDOF,
    MIN_PORTFOLIO_PERIODS,
    MetricOutput,
)
from factorlib.metrics._helpers import (
    _assign_quantile_groups,
    _compute_tie_ratio,
    _median_universe_size,
    _sample_non_overlapping,
    _short_circuit_output,
    _warn_high_tie_ratio,
)
from factorlib._stats import _calc_t_stat, _p_value_from_t, _significance_marker


def compute_spread_series(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    tie_policy: str = "ordinal",
) -> pl.DataFrame:
    """Per-date long-short spread series (non-overlapping).

    Top bucket = highest factor rank; bottom bucket = lowest. Labels use
    ``top_return`` / ``bottom_return`` rather than ``q1_return`` /
    ``q5_return`` because the bucket width depends on ``n_groups`` — at
    ``n_groups=10`` the bottom is Q10, not Q5.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return``.
        n_groups: Number of quantile groups.
        tie_policy: See ``_assign_quantile_groups``. ``"ordinal"`` (default)
            keeps balanced bucket sizes; ``"average"`` keeps tied assets
            in the same bucket — prefer for low-cardinality factors.

    Returns:
        DataFrame with ``date, spread, top_return, bottom_return, universe_return``.
    """
    sampled = _sample_non_overlapping(df, forward_periods)

    median_n = _median_universe_size(sampled)
    per_group = median_n // n_groups if n_groups > 0 else 0
    if per_group < 5:
        warnings.warn(
            f"Median {per_group} assets per group (N={median_n}, "
            f"n_groups={n_groups}). Spread may be dominated by "
            f"individual assets. Consider reducing n_groups.",
            UserWarning,
            stacklevel=2,
        )

    grouped = _assign_quantile_groups(
        sampled, factor_col, n_groups, tie_policy=tie_policy,
    )

    top_group = n_groups - 1
    bottom_group = 0

    return (
        grouped.group_by("date")
        .agg(
            pl.col(return_col)
            .filter(pl.col("_group") == top_group)
            .mean()
            .alias("top_return"),
            pl.col(return_col)
            .filter(pl.col("_group") == bottom_group)
            .mean()
            .alias("bottom_return"),
            pl.col(return_col).mean().alias("universe_return"),
        )
        .with_columns(
            (pl.col("top_return") - pl.col("bottom_return")).alias("spread"),
        )
        .sort("date")
    )


def quantile_spread(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    _precomputed_series: pl.DataFrame | None = None,
    tie_policy: str = "ordinal",
) -> MetricOutput:
    """long-short spread (per-period mean).

    Args:
        _precomputed_series: If provided, skip recomputing ``compute_spread_series``.
        tie_policy: Bucketing tie-break policy, see ``_assign_quantile_groups``.
            When ``_precomputed_series`` is passed, this only affects the
            ``tie_ratio`` diagnostic — the series itself was already built.

    Returns:
        MetricOutput with per-period mean spread, t-stat from non-overlapping periods.
    """
    # Compute tie_ratio on the sampled subset (what bucketing actually sees)
    # rather than the full panel — ~N/forward_periods smaller scan.
    sampled = _sample_non_overlapping(df, forward_periods)
    tie_ratio = _compute_tie_ratio(sampled)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread", tie_policy)

    series = (
        _precomputed_series
        if _precomputed_series is not None
        else compute_spread_series(df, forward_periods, n_groups, tie_policy=tie_policy)
    )
    spread_vals = series["spread"].drop_nulls()
    n = len(spread_vals)
    if n < MIN_PORTFOLIO_PERIODS:
        return _short_circuit_output(
            "quantile_spread", "insufficient_portfolio_periods",
            n_observed=n, min_required=MIN_PORTFOLIO_PERIODS,
            tie_ratio=tie_ratio, tie_policy=tie_policy,
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

    return MetricOutput(
        name="quantile_spread",
        value=mean_spread,
        stat=t,
        significance=_significance_marker(p),
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


def quantile_spread_vw(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_col: str = "market_cap",
    tie_policy: str = "ordinal",
) -> MetricOutput:
    """Value-weighted long-short spread — alpha concentration diagnostic.

    Formula (per non-overlapping date t):
        For bucket b ∈ {bottom, top}:
            vw_b[t] = Σ_{i∈b} weight[i,t] · return[i,t] / Σ_{i∈b} weight[i,t]
        spread[t] = vw_top[t] − vw_bottom[t]
        value = mean_t spread[t];  t-stat = √n · value / std(spread);  DDOF=1

    Compare with equal-weighted ``quantile_spread``: if VW spread much
    smaller (e.g., < 1/3 of EW), the alpha is driven by small-cap assets
    and may not survive capacity / liquidity constraints.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return, market_cap``.
        weight_col: Column for value weighting (default ``market_cap``).

    Returns:
        MetricOutput with per-period mean VW spread, t-stat, and p-value.
        Short-circuits if ``weight_col`` is missing or post-sampling n <
        ``MIN_PORTFOLIO_PERIODS``.

    References:
        Hou, Xue & Zhang (2020): ~65% of factors disappear under VW.
    """
    if weight_col not in df.columns:
        return _short_circuit_output(
            "quantile_spread_vw", "no_weight_column",
            missing_column=weight_col,
        )

    sampled = _sample_non_overlapping(df, forward_periods)
    tie_ratio = _compute_tie_ratio(sampled, factor_col)
    _warn_high_tie_ratio(tie_ratio, "quantile_spread_vw", tie_policy)

    grouped = _assign_quantile_groups(
        sampled, factor_col, n_groups, tie_policy=tie_policy,
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
    if n < MIN_PORTFOLIO_PERIODS:
        return _short_circuit_output(
            "quantile_spread_vw", "insufficient_portfolio_periods",
            n_observed=n, min_required=MIN_PORTFOLIO_PERIODS,
            tie_ratio=tie_ratio, tie_policy=tie_policy,
        )

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = _calc_t_stat(mean_spread, std_spread, n)

    p = _p_value_from_t(t, n)
    return MetricOutput(
        name="quantile_spread_vw",
        value=mean_spread,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "n_periods": n, "p_value": p, "stat_type": "t", "h0": "mu=0",
            "tie_ratio": tie_ratio, "tie_policy": tie_policy,
        },
    )


def compute_group_returns(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    tie_policy: str = "ordinal",
) -> pl.DataFrame:
    """Mean forward return per quantile bucket (for monotonicity charts).

    Formula:
        1. Sample dates every ``forward_periods`` rows (non-overlapping).
        2. Per sampled date, assign each asset to a quantile group
           0..n_groups-1 by ``factor`` (see ``_assign_quantile_groups``
           for tie_policy semantics).
        3. For each group g:
              mean_return[g] = mean across (date, asset) where _group=g
                               of ``return_col``
        (Equal-weighted across all obs in the bucket, not per-date then
         averaged — use ``compute_spread_series`` if you want the latter.)

    Returns:
        DataFrame with ``group, mean_return`` sorted ascending by group.
        Group 0 = lowest factor rank, n_groups-1 = highest.
    """
    sampled = _sample_non_overlapping(df, forward_periods)
    grouped = _assign_quantile_groups(
        sampled, factor_col, n_groups, tie_policy=tie_policy,
    )

    return (
        grouped.group_by("_group")
        .agg(pl.col(return_col).mean().alias("mean_return"))
        .sort("_group")
        .rename({"_group": "group"})
    )
