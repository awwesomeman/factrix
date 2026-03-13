"""
Layer 1: Data Engine — Pure Polars preprocessing.
All functions are stateless: DataFrame in → DataFrame out.
"""

import polars as pl


def prepare_factor_data(
    df: pl.DataFrame,
    date_col: str = "datetime",
    asset_col: str = "ticker",
    factor_col: str = "factor",
    price_col: str = "close",
    forward_periods: int = 5,
) -> pl.DataFrame:
    """
    Full preprocessing pipeline:
    1. Forward return computation.
    2. Cross-sectional MAD-based z-score normalization.
    """
    out = compute_forward_return(df, date_col, asset_col, price_col, forward_periods)
    out = cross_sectional_zscore(out, date_col, factor_col)

    return out.select(
        pl.col(date_col).alias("date"),
        pl.col(asset_col).alias("asset_id"),
        pl.col(factor_col).alias("factor_raw"),
        pl.col("factor_zscore").alias("factor"),
        pl.col("forward_return"),
    )


def compute_forward_return(
    df: pl.DataFrame,
    date_col: str,
    asset_col: str,
    price_col: str,
    forward_periods: int,
) -> pl.DataFrame:
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


def cross_sectional_zscore(
    df: pl.DataFrame,
    date_col: str,
    factor_col: str,
) -> pl.DataFrame:
    """MAD-robust z-score within each cross-section (date)."""
    return df.with_columns(
        (
            (pl.col(factor_col) - pl.col(factor_col).median().over(date_col))
            / pl.col(factor_col).std().over(date_col)
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("factor_zscore")
    )
