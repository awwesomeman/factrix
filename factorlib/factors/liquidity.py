"""Liquidity-based factor generators."""

import polars as pl


def generate_amihud(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Amihud (2002) 非流動性因子：|日報酬率| / 成交金額（price x volume）的滾動均值。
    取負值：流動性越差，分數越低，避免買到難以交易的股票。
    # WHY: 台股成交金額 = 收盤價 x 成交量（張），用於標準化截面可比性
    """
    return (
        df.sort(["ticker", "datetime"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1)
            .alias("_ret")
        )
        .with_columns(
            (
                (pl.col("_ret").abs() / (pl.col("close") * pl.col("volume")))
                .rolling_mean(window_size=lookback)
                .over("ticker")
                * (-1e8)  # 縮放到合理數量級，負號 = 流動性溢酬
            ).alias("factor")
        )
        .filter(
            pl.col("factor").is_not_null()
            & pl.col("factor").is_not_nan()
            & pl.col("factor").is_finite()
        )
    )
