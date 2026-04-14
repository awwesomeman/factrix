"""Shared helpers used across multiple tool modules.

These are internal utilities — not part of the public API.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib.tools._typing import CALENDAR_DAYS_PER_YEAR


def sample_non_overlapping(
    df: pl.DataFrame,
    forward_periods: int,
) -> pl.DataFrame:
    """Keep every N-th date to avoid overlapping forward returns.

    Args:
        df: DataFrame with a ``date`` column.
        forward_periods: Sampling interval.

    Returns:
        Filtered DataFrame containing only sampled dates.
    """
    sampled = df["date"].unique().sort().gather_every(forward_periods)
    return df.filter(pl.col("date").is_in(sampled.implode()))


def assign_quantile_groups(
    df: pl.DataFrame,
    factor_col: str = "factor",
    n_groups: int = 5,
) -> pl.DataFrame:
    """Assign quantile group labels (0 = bottom, n_groups-1 = top) per date.

    Uses ``rank(method="ordinal")`` to break ties deterministically,
    ensuring balanced group sizes even when many assets share the same
    factor value. Tie-breaking order is arbitrary but consistent.

    Returns:
        DataFrame with ``_group`` column appended.
    """
    # WHY: "ordinal" assigns unique ranks even for tied values,
    # preventing multiple assets from clustering in the same group.
    # "average" would give tied assets the same rank → unbalanced groups.
    return (
        df.with_columns(
            pl.col(factor_col).rank(method="ordinal").over("date").alias("_rank"),
            pl.len().over("date").alias("_n"),
        )
        .with_columns(
            ((pl.col("_rank") - 1) * n_groups / pl.col("_n"))
            .cast(pl.Int32)
            .clip(0, n_groups - 1)
            .alias("_group")
        )
        .drop("_rank", "_n")
    )


def median_universe_size(df: pl.DataFrame) -> int:
    """Median number of unique assets per date."""
    return int(
        df.group_by("date")
        .agg(pl.col("asset_id").n_unique().alias("n"))
        ["n"].median()
    )


def annualize_return(arr: np.ndarray, dates: pl.Series) -> float | None:
    """Compound and annualize a per-period return series.

    Args:
        arr: 1-D array of per-period returns.
        dates: Corresponding date series (for computing date range).

    Returns:
        Annualized return, or None if date range < 0.1 years.
    """
    date_range = dates.drop_nulls()
    n_years = (date_range.max() - date_range.min()).days / CALENDAR_DAYS_PER_YEAR
    if n_years < 0.1:
        return None
    total = float(np.prod(1 + arr)) - 1
    return (1 + total) ** (1 / n_years) - 1
