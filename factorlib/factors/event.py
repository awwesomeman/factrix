"""Event-based signal generators."""

import numpy as np
import polars as pl


def generate_event_signal_mock(df: pl.DataFrame) -> pl.DataFrame:
    """事件訊號因子 (Mock)：隨機標記 2% 買入與 2% 賣出訊號。"""
    rng = np.random.default_rng(42)
    return (
        df.sort(["ticker", "datetime"])
        .with_columns(
            pl.Series("rand", rng.uniform(0, 1, len(df)))
        )
        .with_columns(
            pl.when(pl.col("rand") > 0.98).then(1.0)
            .when(pl.col("rand") < 0.02).then(-1.0)
            .otherwise(0.0)
            .alias("factor")
        )
        .drop("rand")
    )
