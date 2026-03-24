"""Momentum-based factor generators."""

import polars as pl


def generate_momentum(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """動量因子：過去 N 日報酬率。買贏家賣輸家。"""
    return (
        df.sort(["ticker", "datetime"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(lookback).over("ticker") - 1)
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null())
    )


def generate_momentum_60d(df: pl.DataFrame) -> pl.DataFrame:
    """中期動量因子：T-60 到 T-5 的累積報酬率。
    跳過最近 5 日是為了避免短期均值回歸污染動量信號。
    買 12-1 月動量最強的股票，賣最弱的。
    """
    return (
        df.sort(["ticker", "datetime"])
        .with_columns(
            (
                pl.col("close").shift(5).over("ticker")
                / pl.col("close").shift(60).over("ticker")
                - 1
            ).alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )
