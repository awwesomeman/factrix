"""Technical / price-action factor generators."""

import polars as pl

from factorlib.factors._helpers import compute_market_return


def generate_mean_reversion(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """短期反轉因子：負的短期報酬率。跌深反彈邏輯。"""
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (-(pl.col("price") / pl.col("price").shift(lookback).over("asset_id") - 1))
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null())
    )


def generate_52w_high_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """近高點因子：收盤價 / 過去 252 日最高價。
    接近歷史高點的股票傾向續漲 (George & Hwang, 2004)。
    """
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("high").rolling_max(window_size=252).over("asset_id"))
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )


def generate_overnight_return(df: pl.DataFrame) -> pl.DataFrame:
    """隔夜收益因子：開盤價 / 前日收盤 - 1。反映盤後資訊流入方向。"""
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("open") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )


def generate_intraday_range(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """日內振幅因子：負的 (High-Low)/Close 的 N 日均值。
    高振幅 = 高不確定性 = 負溢酬。
    """
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (
                -((pl.col("high") - pl.col("low")) / pl.col("price"))
                .rolling_mean(window_size=lookback)
                .over("asset_id")
            ).alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )


def generate_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """RSI 反轉因子：負的 RSI 值。超買超賣均值回歸。"""
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") - pl.col("price").shift(1).over("asset_id")).alias("change")
        )
        .with_columns(
            pl.when(pl.col("change") > 0).then(pl.col("change")).otherwise(0.0).alias("gain"),
            pl.when(pl.col("change") < 0).then(-pl.col("change")).otherwise(0.0).alias("loss"),
        )
        .with_columns(
            pl.col("gain").rolling_mean(window_size=period).over("asset_id").alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=period).over("asset_id").alias("avg_loss"),
        )
        .with_columns(
            pl.when(pl.col("avg_loss") > 1e-10)
            .then(-(100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))))
            .otherwise(None)
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )


def generate_volume_price_trend(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """量價趨勢因子：成交量加權報酬率。
    量增價漲 = 趨勢確認。
    """
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1).alias("ret"),
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=lookback).over("asset_id")).alias("vol_ratio"),
        )
        .with_columns(
            (pl.col("ret") * pl.col("vol_ratio"))
            .rolling_sum(window_size=lookback)
            .over("asset_id")
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )


def generate_market_beta(df: pl.DataFrame, lookback: int = 60) -> pl.DataFrame:
    """滾動市場 Beta 因子（取負值）。
    低 Beta 股票長期有超額報酬 (Black, 1972; Frazzini & Pedersen, 2014)。
    Beta = cov(r_i, r_m) / var(r_m)，用滾動均值公式向量化計算。
    """
    market = compute_market_return(df)

    with_ret = (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("_ret")
        )
    )

    joined = with_ret.join(market, on="date", how="left")

    # 滾動計算 cov(r_i, r_m) 和 var(r_m)
    # cov = E[r_i * r_m] - E[r_i] * E[r_m]
    # var = E[r_m^2] - E[r_m]^2
    result = (
        joined
        .with_columns(
            (pl.col("_ret") * pl.col("mkt_ret")).alias("_ret_mkt"),
            (pl.col("mkt_ret") ** 2).alias("_mkt_sq"),
        )
        .with_columns(
            pl.col("_ret_mkt").rolling_mean(window_size=lookback).over("asset_id").alias("_mean_cross"),
            pl.col("_ret").rolling_mean(window_size=lookback).over("asset_id").alias("_mean_ret"),
            pl.col("mkt_ret").rolling_mean(window_size=lookback).over("asset_id").alias("_mean_mkt"),
            pl.col("_mkt_sq").rolling_mean(window_size=lookback).over("asset_id").alias("_mean_mkt_sq"),
        )
        .with_columns(
            (pl.col("_mean_cross") - pl.col("_mean_ret") * pl.col("_mean_mkt")).alias("_cov"),
            (pl.col("_mean_mkt_sq") - pl.col("_mean_mkt") ** 2).alias("_var_mkt"),
        )
        .with_columns(
            pl.when(pl.col("_var_mkt") > 1e-10)
            .then(-pl.col("_cov") / pl.col("_var_mkt"))  # 負號：低 Beta 因子
            .otherwise(None)
            .alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )
    return result


def generate_max_effect(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """MAX 效應因子（Bali et al., 2011）：過去 N 日最大單日報酬率的負值。
    投資人偏愛彩票型股票（有機會暴漲），導致這類股票被高估、預期報酬偏低。
    # WHY: MAX 取負值使得低 MAX（穩健）的股票得高分，符合買穩健方向
    """
    return (
        df.sort(["asset_id", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("_ret")
        )
        .with_columns(
            (
                -pl.col("_ret")
                .rolling_max(window_size=lookback)
                .over("asset_id")
            ).alias("factor")
        )
        .filter(pl.col("factor").is_not_null() & pl.col("factor").is_not_nan())
    )
