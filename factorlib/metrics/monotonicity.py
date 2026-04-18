"""Monotonicity test for cross-sectional panels.

Measures whether factor quantile groups exhibit monotonic return ordering.
Per-date: split into n_groups by factor rank, compute mean return per group,
Spearman corr between group index and return.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import DDOF, MIN_MONOTONICITY_PERIODS, MetricOutput
from factorlib.metrics._helpers import _assign_quantile_groups, _sample_non_overlapping
from factorlib._stats import _calc_t_stat, _p_value_from_t, _significance_marker


def monotonicity(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 10,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Quantile return monotonicity (Spearman correlation).

    ``value`` = mean |Spearman| — magnitude of monotonicity (always ≥ 0).
    ``t_stat`` = t-test on signed Spearman — whether direction is consistent.

    A high ``value`` with insignificant ``t_stat`` means the factor has
    strong monotonicity but the direction flips across dates.

    Args:
        df: Panel with ``date, asset_id, factor, forward_return``.
        n_groups: Number of quantile groups (default 10 for Taiwan ~2000 stocks).
            Use 5 for N < 1000, 3 for N < 200.

    Returns:
        MetricOutput with value = mean |Spearman(group_idx, group_return)|.
    """
    filtered = _sample_non_overlapping(df, forward_periods)
    grouped = _assign_quantile_groups(filtered, factor_col, n_groups)

    # Mean return per group per date
    group_returns = (
        grouped.group_by(["date", "_group"])
        .agg(pl.col(return_col).mean().alias("group_ret"))
        .sort(["date", "_group"])
    )

    # WHY: Spearman(group_index, returns) = Pearson(group_index, rank(returns))
    mono_df = (
        group_returns
        .filter(pl.col("group_ret").is_not_null() & pl.col("group_ret").is_not_nan())
        .with_columns(
            pl.col("group_ret").rank(method="average").over("date").alias("_ret_rank")
        )
        .group_by("date")
        .agg(
            pl.corr("_group", "_ret_rank").alias("mono"),
            pl.len().alias("n"),
        )
        .filter(
            (pl.col("n") == n_groups)
            & pl.col("mono").is_not_null()
            & pl.col("mono").is_not_nan()
        )
        .sort("date")
    )

    if len(mono_df) < MIN_MONOTONICITY_PERIODS:
        return MetricOutput(
            name="monotonicity", value=0.0, stat=0.0, significance="",
            metadata={
                "reason": "insufficient_monotonicity_periods",
                "n_observed": len(mono_df),
                "min_required": MIN_MONOTONICITY_PERIODS,
                "n_groups": n_groups,
            },
        )

    mono_arr = mono_df["mono"].to_numpy()
    avg_mono = float(np.mean(np.abs(mono_arr)))
    mean_mono = float(np.mean(mono_arr))
    std_mono = float(np.std(mono_arr, ddof=DDOF))
    t = _calc_t_stat(mean_mono, std_mono, len(mono_arr))

    p = _p_value_from_t(t, len(mono_arr))
    return MetricOutput(
        name="monotonicity",
        value=avg_mono,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "mean_signed": mean_mono,
            "n_valid_periods": len(mono_arr),
            "n_groups": n_groups,
        },
    )
