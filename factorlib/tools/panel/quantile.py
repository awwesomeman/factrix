"""Quantile analysis for cross-sectional panels.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
Output: spread series, long/short alpha decomposition.

All spread series are time-indexed (``date, value``) and can be fed
into any ``series/`` tool.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib.tools._typing import (
    DDOF,
    MIN_PORTFOLIO_PERIODS,
    MetricOutput,
)
from factorlib.tools._helpers import (
    annualize_return,
    assign_quantile_groups,
    sample_non_overlapping,
)
from factorlib.tools.series.significance import calc_t_stat, significance_marker


def quantile_spread_series(
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
    sampled = sample_non_overlapping(df, forward_periods)
    grouped = assign_quantile_groups(sampled, factor_col, n_groups)

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
    """Q1-Q5 spread (annualized).

    Args:
        _precomputed_series: If provided, skip recomputing ``quantile_spread_series``.

    Returns:
        MetricOutput with annualized spread, t-stat from non-overlapping periods.
    """
    series = _precomputed_series if _precomputed_series is not None else quantile_spread_series(df, forward_periods, n_groups)
    spread_vals = series["spread"].drop_nulls()
    n = len(spread_vals)
    if n < MIN_PORTFOLIO_PERIODS:
        return MetricOutput(name="Q1_Q5_Spread", value=0.0, t_stat=0.0, significance="○")

    arr = spread_vals.to_numpy()
    mean_spread = float(np.mean(arr))
    std_spread = float(np.std(arr, ddof=DDOF))
    t = calc_t_stat(mean_spread, std_spread, n)

    ann = annualize_return(arr, series["date"])
    if ann is None:
        return MetricOutput(name="Q1_Q5_Spread", value=0.0, t_stat=0.0, significance="○")

    return MetricOutput(
        name="Q1_Q5_Spread",
        value=ann,
        t_stat=t,
        significance=significance_marker(t),
        metadata={"mean_per_period": mean_spread, "n_periods": n},
    )


def long_short_alpha(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 5,
    _precomputed_series: pl.DataFrame | None = None,
) -> MetricOutput:
    """Long Alpha (Q1 - Universe) and Short Alpha (Universe - Q5), annualized.

    Args:
        _precomputed_series: If provided, skip recomputing ``quantile_spread_series``.
            Caller can pass the same series used for ``quantile_spread`` to avoid
            duplicate work.

    Returns:
        MetricOutput with value=Long_Alpha (annualized),
        metadata containing short_alpha and per-period details.
    """
    series = _precomputed_series if _precomputed_series is not None else quantile_spread_series(df, forward_periods, n_groups)
    n = len(series)
    if n < MIN_PORTFOLIO_PERIODS:
        return MetricOutput(name="Long_Short_Alpha", value=0.0, t_stat=0.0, significance="○")

    long_excess = (series["q1_return"] - series["universe_return"]).drop_nulls()
    short_excess = (series["universe_return"] - series["q5_return"]).drop_nulls()

    long_arr = long_excess.to_numpy()
    short_arr = short_excess.to_numpy()

    mean_long = float(np.mean(long_arr))
    std_long = float(np.std(long_arr, ddof=DDOF))
    t_long = calc_t_stat(mean_long, std_long, len(long_arr))

    mean_short = float(np.mean(short_arr))
    std_short = float(np.std(short_arr, ddof=DDOF))
    t_short = calc_t_stat(mean_short, std_short, len(short_arr))

    ann_long = annualize_return(long_arr, series["date"])
    ann_short = annualize_return(short_arr, series["date"])
    if ann_long is None:
        return MetricOutput(name="Long_Short_Alpha", value=0.0, t_stat=0.0, significance="○")

    return MetricOutput(
        name="Long_Short_Alpha",
        value=ann_long,
        t_stat=t_long,
        significance=significance_marker(t_long),
        metadata={
            "long_alpha_ann": ann_long,
            "short_alpha_ann": ann_short or 0.0,
            "long_t_stat": t_long,
            "short_t_stat": t_short,
            "short_significance": significance_marker(t_short),
        },
    )


def quantile_group_returns(
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
    sampled = sample_non_overlapping(df, forward_periods)
    grouped = assign_quantile_groups(sampled, factor_col, n_groups)

    return (
        grouped.group_by("_group")
        .agg(pl.col(return_col).mean().alias("mean_return"))
        .sort("_group")
        .rename({"_group": "group"})
    )
