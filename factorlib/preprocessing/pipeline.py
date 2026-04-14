"""Preprocessing orchestration: chain Step 1-5 into a single call.

Each step is independently importable from ``returns`` and ``normalize``.
This module only provides the convenience ``run_preprocessing()`` wrapper.
"""

import polars as pl

from factorlib.preprocessing.returns import (
    compute_abnormal_return,
    compute_forward_return,
    winsorize_forward_return,
)
from factorlib.preprocessing.normalize import (
    cross_sectional_zscore,
    mad_winsorize,
)


def run_preprocessing(
    df: pl.DataFrame,
    *,
    date_col: str = "datetime",
    asset_col: str = "ticker",
    factor_col: str = "factor",
    price_col: str = "close",
    forward_periods: int = 5,
    return_clip_pct: tuple[float, float] = (0.01, 0.99),
    mad_n: float = 3.0,
) -> pl.DataFrame:
    """Run the full Step 1-5 preprocessing pipeline.

    Steps:
        1. Forward return (``close[t+N] / close[t] - 1``).
        2. Forward return percentile winsorization.
        3. Abnormal return (cross-sectional de-mean).
        4. Factor MAD winsorization.
        5. Cross-sectional MAD z-score.

    Args:
        df: Raw data with date, asset, factor, and price columns.
        date_col: Date column name in the input DataFrame.
        asset_col: Asset identifier column name.
        factor_col: Raw factor value column name.
        price_col: Price column name for computing forward returns.
        forward_periods: Number of periods for forward return (default 5).
        return_clip_pct: (lower, upper) quantile bounds for return clipping.
        mad_n: Number of MAD units for factor winsorization (0 to disable).

    Returns:
        DataFrame with standardized column names:
        ``date, asset_id, factor_raw, factor, forward_return, abnormal_return``.
    """
    out = compute_forward_return(df, date_col, asset_col, price_col, forward_periods)
    out = winsorize_forward_return(out, date_col, lower=return_clip_pct[0], upper=return_clip_pct[1])
    out = compute_abnormal_return(out, date_col)

    # WHY: 保留原始因子值供後續 Profile 分析（如 Q1_Concentration 使用原始分佈）
    out = out.with_columns(pl.col(factor_col).alias("factor_raw"))

    out = mad_winsorize(out, date_col, factor_col, n_mad=mad_n)
    out = cross_sectional_zscore(out, date_col, factor_col)

    # WHY: 輸出統一欄位名稱，下游工具不需知道上游的原始欄位命名
    return out.select(
        pl.col(date_col).cast(pl.Datetime("ms")).alias("date"),
        pl.col(asset_col).alias("asset_id"),
        pl.col("factor_raw"),
        pl.col("factor_zscore").alias("factor"),
        pl.col("forward_return"),
        pl.col("abnormal_return"),
    )
