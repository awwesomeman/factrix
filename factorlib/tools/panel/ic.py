"""IC (Information Coefficient) computation for cross-sectional panels.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
Output: time-indexed IC series (``date, ic``) that can be fed into
any ``series/`` tool (oos, trend, significance, hit_rate).
"""

from __future__ import annotations

import polars as pl

from factorlib.tools._typing import (
    EPSILON,
    MIN_IC_PERIODS,
    MetricOutput,
)
from factorlib.tools.series.significance import calc_t_stat, significance_marker


def compute_ic(
    df: pl.DataFrame,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-date Spearman Rank IC.

    Args:
        df: Panel with ``date``, ``asset_id``, ``factor_col``, ``return_col``.

    Returns:
        DataFrame with columns ``date, ic`` sorted by date.
        Dates with fewer than ``MIN_IC_PERIODS`` assets are dropped.
    """
    ranked = df.with_columns(
        pl.col(factor_col).rank(method="average").over("date").alias("_rank_factor"),
        pl.col(return_col).rank(method="average").over("date").alias("_rank_return"),
    )
    return (
        ranked.group_by("date")
        .agg(
            pl.corr("_rank_factor", "_rank_return").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= MIN_IC_PERIODS)
        .sort("date")
        .select("date", "ic")
    )


def non_overlapping_ic_tstat(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> float:
    """T-stat from non-overlapping IC samples.

    Samples every ``forward_periods``-th date to eliminate autocorrelation
    from overlapping forward returns.

    Args:
        ic_df: Output of ``compute_ic()`` (``date, ic``), already sorted by date.
        forward_periods: Sampling interval.

    Returns:
        t-statistic of the mean IC, or 0.0 if insufficient data.
    """
    # WHY: ic_df is already sorted by compute_ic(), no need to re-sort
    sampled_dates = ic_df["date"].gather_every(forward_periods)
    sampled_ic = ic_df.filter(
        pl.col("date").is_in(sampled_dates.implode())
    )["ic"].drop_nulls()

    n = len(sampled_ic)
    if n < 2:
        return 0.0
    return calc_t_stat(
        float(sampled_ic.mean()),
        float(sampled_ic.std()),
        n,
    )


def ic_ir(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
) -> MetricOutput:
    """IC_IR = |mean(IC)| / std(IC).

    Uses all IC periods for the ratio; t-stat uses non-overlapping samples.

    Args:
        ic_df: Output of ``compute_ic()``.

    Returns:
        MetricOutput with value=IC_IR, t_stat from non-overlapping sampling.
    """
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    if n < MIN_IC_PERIODS:
        return MetricOutput(name="IC_IR", value=0.0, t_stat=0.0, significance="○")

    mean_ic = float(ic_vals.mean())
    std_ic = float(ic_vals.std())

    if std_ic < EPSILON:
        return MetricOutput(name="IC_IR", value=0.0, t_stat=0.0, significance="○")

    ratio = abs(mean_ic) / std_ic
    t = non_overlapping_ic_tstat(ic_df, forward_periods)

    return MetricOutput(
        name="IC_IR",
        value=ratio,
        t_stat=t,
        significance=significance_marker(t),
        metadata={"mean_ic": mean_ic, "std_ic": std_ic, "n_periods": n},
    )


def multi_horizon_ic(
    df: pl.DataFrame,
    asset_col: str = "asset_id",
    price_col: str = "close",
    factor_col: str = "factor",
    periods: list[int] | None = None,
) -> MetricOutput:
    """Compute mean IC at multiple forward horizons.

    Reuses ``compute_forward_return`` from preprocessing to ensure
    consistent return calculation across the codebase.

    Args:
        df: Raw panel with ``date``, ``asset_col``, ``price_col``, ``factor_col``.
            Must contain price data (not preprocessed forward_return).
        periods: List of forward periods (default [1, 5, 10, 20]).

    Returns:
        MetricOutput with value=mean IC at the default horizon,
        metadata containing per-horizon IC values.
    """
    if periods is None:
        periods = [1, 5, 10, 20]

    horizon_ics: dict[int, float] = {}

    # WHY: 一次算所有 horizon 的 forward return，避免重複排序
    sorted_df = df.sort([asset_col, "date"])
    all_returns = sorted_df.with_columns([
        (
            pl.col(price_col).shift(-p).over(asset_col)
            / pl.col(price_col)
            - 1
        ).alias(f"_fwd_ret_{p}")
        for p in periods
    ])

    for p in periods:
        ret_col = f"_fwd_ret_{p}"
        valid = all_returns.filter(pl.col(ret_col).is_not_null())

        ic_series = compute_ic(valid, factor_col=factor_col, return_col=ret_col)
        ic_vals = ic_series["ic"].drop_nulls()

        if len(ic_vals) >= MIN_IC_PERIODS:
            horizon_ics[p] = float(ic_vals.mean())
        else:
            horizon_ics[p] = float("nan")

    primary = horizon_ics.get(periods[0], float("nan"))
    return MetricOutput(
        name="Multi_Horizon_IC",
        value=primary,
        metadata={"horizon_ics": horizon_ics},
    )
