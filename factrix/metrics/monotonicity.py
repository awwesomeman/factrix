"""Monotonicity test for cross-sectional panels.

Aggregation: per-date Spearman corr between quantile index and group
mean return (cross-section step), then non-overlapping cross-asset t
on the per-date series.

Measures whether factor quantile groups exhibit monotonic return ordering.
Per-date: split into n_groups by factor rank, compute mean return per group,
Spearman corr between group index and return.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import (
    DDOF,
    MIN_MONOTONICITY_PERIODS,
    MetricOutput,
)
from factrix.metrics._helpers import (
    _assign_quantile_groups,
    _compute_tie_ratio,
    _sample_non_overlapping,
    _short_circuit_output,
    _warn_high_tie_ratio,
)
from factrix._stats import _calc_t_stat, _p_value_from_t, _significance_marker


def monotonicity(
    df: pl.DataFrame,
    forward_periods: int = 5,
    n_groups: int = 10,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    tie_policy: str = "ordinal",
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
        tie_policy: Bucketing tie-break policy, see ``_assign_quantile_groups``.

    Returns:
        MetricOutput with value = mean |Spearman(group_idx, group_return)|.
    """
    filtered = _sample_non_overlapping(df, forward_periods)
    tie_ratio = _compute_tie_ratio(filtered, factor_col)
    _warn_high_tie_ratio(tie_ratio, "monotonicity", tie_policy)

    grouped = _assign_quantile_groups(
        filtered, factor_col, n_groups, tie_policy=tie_policy,
    )

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
        return _short_circuit_output(
            "monotonicity", "insufficient_monotonicity_periods",
            n_observed=len(mono_df), min_required=MIN_MONOTONICITY_PERIODS,
            n_groups=n_groups, tie_ratio=tie_ratio, tie_policy=tie_policy,
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
            "tie_ratio": tie_ratio,
            "tie_policy": tie_policy,
        },
    )
