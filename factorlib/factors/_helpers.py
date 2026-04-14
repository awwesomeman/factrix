"""Shared helpers for factor generators."""

import polars as pl


def compute_market_return(df: pl.DataFrame) -> pl.DataFrame:
    """Equal-weighted daily market return across all tickers."""
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("_ret")
        )
        .group_by("date")
        .agg(pl.col("_ret").mean().alias("mkt_ret"))
        .sort("date")
    )
