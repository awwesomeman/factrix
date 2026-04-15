"""Preprocessing Step 1-3: forward return computation and adjustment.

Step 1 — Forward Return: price[t+N] / price[t] - 1
Step 2 — Winsorize Forward Return: per-date percentile clip
Step 3 — Abnormal Return: forward_return - cross-sectional mean

All functions expect canonical column names (date, asset_id, price).
Use ``adapt()`` to rename before calling.
"""

import polars as pl


def compute_forward_return(
    df: pl.DataFrame,
    forward_periods: int = 5,
) -> pl.DataFrame:
    """Step 1: Compute N-period forward return per asset.

    Args:
        df: Must contain ``date``, ``asset_id``, ``price``.
        forward_periods: Number of periods ahead (default 5).

    Returns:
        Input DataFrame with ``forward_return`` column appended.
        Rows where forward return is null (end of series) are dropped.
    """
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (
                pl.col("price").shift(-forward_periods).over("asset_id")
                / pl.col("price")
                - 1
            ).alias("forward_return")
        )
        .filter(pl.col("forward_return").is_not_null())
    )


def winsorize_forward_return(
    df: pl.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Step 2: Per-date percentile clip on forward returns.

    Args:
        lower: Lower quantile bound (default 0.01 = 1st percentile).
        upper: Upper quantile bound (default 0.99 = 99th percentile).
            Set to (0.0, 1.0) to disable.

    Returns:
        DataFrame with ``forward_return`` clipped in-place.
    """
    if lower <= 0.0 and upper >= 1.0:
        return df

    lb = pl.col("forward_return").quantile(lower).over("date")
    ub = pl.col("forward_return").quantile(upper).over("date")

    return df.with_columns(
        pl.col("forward_return").clip(lb, ub).alias("forward_return")
    )


def compute_abnormal_return(df: pl.DataFrame) -> pl.DataFrame:
    """Step 3: Cross-sectional abnormal return.

    ``abnormal_return = forward_return - mean(forward_return) per date``

    Returns:
        DataFrame with ``abnormal_return`` column appended.
    """
    return df.with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over("date"))
        .alias("abnormal_return")
    )
