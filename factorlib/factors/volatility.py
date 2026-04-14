"""Volatility-based factor generators."""

import polars as pl

from factorlib.factors._helpers import compute_market_return


def generate_volatility(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """低波動因子：過去 N 日收益率標準差的負值。
    低波動異常 (Ang et al., 2006)。
    """
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("daily_ret")
        )
        .with_columns(
            (-pl.col("daily_ret").rolling_std(window_size=lookback).over("asset_id"))
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )


def generate_idiosyncratic_vol(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """特異波動率因子（IVOL）：扣除市場報酬後的殘差標準差，取負值。
    IVOL 越低的股票，未來報酬率越高（IVOL puzzle, Ang et al., 2006）。
    近似做法：std(r_i - r_m) 作為特異波動率代理指標。
    # WHY: 直接用 OLS 殘差需要逐股逐窗滾動回歸，計算成本高；
    #      std(r_i - r_m) 是 IVOL 的合理近似，在截面排序上效果相當
    """
    market = compute_market_return(df)

    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("_ret")
        )
        .join(market, on="date", how="left")
        .with_columns(
            (pl.col("_ret") - pl.col("mkt_ret")).alias("_resid")
        )
        .with_columns(
            (
                -pl.col("_resid")
                .rolling_std(window_size=lookback)
                .over("asset_id")
            ).alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )
