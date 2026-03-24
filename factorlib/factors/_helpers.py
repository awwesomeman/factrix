"""Shared helpers for factor generators."""

import polars as pl


def compute_market_return(df: pl.DataFrame) -> pl.DataFrame:
    """Equal-weighted daily market return across all tickers."""
    return (
        df.sort(["ticker", "datetime"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1)
            .alias("_ret")
        )
        .group_by("datetime")
        .agg(pl.col("_ret").mean().alias("mkt_ret"))
        .sort("datetime")
    )
