"""Preprocessing Step 1-3: forward return computation and adjustment.

Step 1 — Forward Return: close[t+N] / close[t] - 1
Step 2 — Winsorize Forward Return: per-date percentile clip
Step 3 — Abnormal Return: forward_return - cross-sectional mean

All functions are stateless: DataFrame in → DataFrame out.
Each function is independently importable.
"""

import polars as pl


def compute_forward_return(
    df: pl.DataFrame,
    date_col: str = "datetime",
    asset_col: str = "ticker",
    price_col: str = "close",
    forward_periods: int = 5,
) -> pl.DataFrame:
    """Step 1: Compute N-period forward return per asset.

    Args:
        df: Must contain date_col, asset_col, price_col.
        forward_periods: Number of periods ahead (default 5).

    Returns:
        Input DataFrame with ``forward_return`` column appended.
        Rows where forward return is null (end of series) are dropped.
    """
    return (
        df.sort([asset_col, date_col])
        .with_columns(
            (
                pl.col(price_col).shift(-forward_periods).over(asset_col)
                / pl.col(price_col)
                - 1
            ).alias("forward_return")
        )
        .filter(pl.col("forward_return").is_not_null())
    )


def winsorize_forward_return(
    df: pl.DataFrame,
    date_col: str = "datetime",
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Step 2: Per-date percentile clip on forward returns.

    Caps extreme return observations at the (lower, upper) quantiles within
    each cross-section to reduce the influence of outliers on downstream
    aggregations that use ``.mean()`` (e.g. spread, Long_Alpha).

    Args:
        lower: Lower quantile bound (default 0.01 = 1st percentile).
        upper: Upper quantile bound (default 0.99 = 99th percentile).
            Set to (0.0, 1.0) to disable.

    Returns:
        DataFrame with ``forward_return`` clipped in-place.
    """
    if lower <= 0.0 and upper >= 1.0:
        return df

    lb = pl.col("forward_return").quantile(lower).over(date_col)
    ub = pl.col("forward_return").quantile(upper).over(date_col)

    return df.with_columns(
        pl.col("forward_return").clip(lb, ub).alias("forward_return")
    )


def compute_abnormal_return(
    df: pl.DataFrame,
    date_col: str = "datetime",
) -> pl.DataFrame:
    """Step 3: Cross-sectional abnormal return.

    ``abnormal_return = forward_return - mean(forward_return) per date``

    This is equal-weighted market-mean adjustment. For beta-adjusted
    abnormal return, see the P1 improvement in the v3 spec.

    Returns:
        DataFrame with ``abnormal_return`` column appended.
    """
    return df.with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over(date_col))
        .alias("abnormal_return")
    )
