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
    _median_universe_size,
    _sample_non_overlapping,
)
from factorlib._stats import _calc_t_stat, _p_value_from_t, _significance_marker


def compute_spread_series(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-date Q1-Q5 spread series (non-overlapping).

    Q1 = top quantile (highest factor), Q5 = bottom quantile.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return``.
        n_groups: Number of quantile groups.

    Returns:
        DataFrame with ``date, spread, q1_return, q5_return, universe_return``.
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

    grouped = _assign_quantile_groups(sampled, factor_col, n_groups)

    top_group = n_groups - 1
    bottom_group = 0

    return (
        grouped.group_by("date")
        .agg(
            pl.col(return_col)
            .filter(pl.col("_group") == top_group)
            .mean()
            .alias("q1_return"),
            pl.col(return_col)
            .filter(pl.col("_group") == bottom_group)
            .mean()
            .alias("q5_return"),
            pl.col(return_col).mean().alias("universe_return"),
        )
        .with_columns(
            (pl.col("q1_return") - pl.col("q5_return")).alias("spread"),
        )
        .sort("date")
    )


def quantile_spread(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    _precomputed_series: pl.DataFrame | None = None,
) -> MetricOutput:
    """Q1-Q5 spread (per-period mean).

    Args:
        _precomputed_series: If provided, skip recomputing ``compute_spread_series``.

    Returns:
        MetricOutput with per-period mean spread, t-stat from non-overlapping periods.
    """
    series = _precomputed_series if _precomputed_series is not None else compute_spread_series(df, forward_periods, n_groups)
    spread_vals = series["spread"].drop_nulls()
    n = len(spread_vals)
    if n < MIN_PORTFOLIO_PERIODS:
        return MetricOutput(
            name="q1_q5_spread", value=0.0, stat=0.0, significance="",
            metadata={
                "reason": "insufficient_portfolio_periods",
                "n_observed": n,
                "min_required": MIN_PORTFOLIO_PERIODS,
            },
        )

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = _calc_t_stat(mean_spread, std_spread, n)

    p = _p_value_from_t(t, n)

    # Long/short decomposition (spread = long_alpha + short_alpha)
    long_excess = (series["q1_return"] - series["universe_return"]).drop_nulls()
    short_excess = (series["universe_return"] - series["q5_return"]).drop_nulls()

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
        name="q1_q5_spread",
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
        },
    )


def quantile_spread_vw(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_col: str = "market_cap",
) -> MetricOutput:
    """Value-weighted Q1-Q5 spread (per-period mean).

    Compares with equal-weighted spread to detect small-cap concentration.
    If VW spread << EW spread, alpha is driven by small stocks.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return, market_cap``.
        weight_col: Column for value weighting (default ``market_cap``).

    Returns:
        MetricOutput with per-period mean VW spread.

    References:
        Hou, Xue & Zhang (2020): ~65% of factors disappear under VW.
    """
    if weight_col not in df.columns:
        return MetricOutput(
            name="q1_q5_spread_vw", value=0.0, stat=0.0, significance="",
            metadata={
                "reason": "missing_weight_column",
                "missing_column": weight_col,
            },
        )

    sampled = _sample_non_overlapping(df, forward_periods)
    grouped = _assign_quantile_groups(sampled, factor_col, n_groups)

    top_group = n_groups - 1
    bottom_group = 0

    # WHY: per-date weighted mean for Q1 and Q5
    vw_series = (
        grouped.with_columns(
            (pl.col(return_col) * pl.col(weight_col)).alias("_wr"),
        )
        .group_by("date")
        .agg(
            (
                pl.col("_wr").filter(pl.col("_group") == top_group).sum()
                / pl.col(weight_col).filter(pl.col("_group") == top_group).sum()
            ).alias("q1_return_vw"),
            (
                pl.col("_wr").filter(pl.col("_group") == bottom_group).sum()
                / pl.col(weight_col).filter(pl.col("_group") == bottom_group).sum()
            ).alias("q5_return_vw"),
        )
        .with_columns(
            (pl.col("q1_return_vw") - pl.col("q5_return_vw")).alias("spread_vw"),
        )
        .sort("date")
    )

    spread_vals = vw_series["spread_vw"].drop_nulls()
    n = len(spread_vals)
    if n < MIN_PORTFOLIO_PERIODS:
        return MetricOutput(
            name="q1_q5_spread_vw", value=0.0, stat=0.0, significance="",
            metadata={
                "reason": "insufficient_portfolio_periods",
                "n_observed": n,
                "min_required": MIN_PORTFOLIO_PERIODS,
            },
        )

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = _calc_t_stat(mean_spread, std_spread, n)

    p = _p_value_from_t(t, n)
    return MetricOutput(
        name="q1_q5_spread_vw",
        value=mean_spread,
        stat=t,
        significance=_significance_marker(p),
        metadata={"n_periods": n, "p_value": p, "stat_type": "t", "h0": "mu=0"},
    )


def compute_group_returns(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Mean return per quantile group (for monotonicity charts).

    Returns:
        DataFrame with ``group, mean_return`` averaged across non-overlapping dates.
    """
    sampled = _sample_non_overlapping(df, forward_periods)
    grouped = _assign_quantile_groups(sampled, factor_col, n_groups)

    return (
        grouped.group_by("_group")
        .agg(pl.col(return_col).mean().alias("mean_return"))
        .sort("_group")
        .rename({"_group": "group"})
    )
