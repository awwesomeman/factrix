"""Preprocessing orchestration: chain Step 1-5 into a single call.

Each step is independently importable from ``returns`` and ``normalize``.
This module only provides the convenience ``preprocess_cs_factor()`` wrapper.

Expects canonical column names (date, asset_id, price, factor).
Use ``adapt()`` to rename before calling.
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


def preprocess_cs_factor(
    df: pl.DataFrame,
    *,
    forward_periods: int = 5,
    return_clip_pct: tuple[float, float] = (0.01, 0.99),
    mad_n: float = 3.0,
) -> pl.DataFrame:
    """Run the full Step 1-5 preprocessing pipeline for cross-sectional factors.

    Steps:
        1. Forward return (``price[t+N] / price[t] - 1``).
        2. Forward return percentile winsorization.
        3. Abnormal return (cross-sectional de-mean).
        4. Factor MAD winsorization.
        5. Cross-sectional MAD z-score.

    Args:
        df: Data with canonical columns ``date``, ``asset_id``, ``price``,
            ``factor``. Use ``adapt()`` to rename if needed.
        forward_periods: Number of periods for forward return (default 5).
        return_clip_pct: (lower, upper) quantile bounds for return clipping.
        mad_n: Number of MAD units for factor winsorization (0 to disable).

    Returns:
        DataFrame with columns:
        ``date, asset_id, factor_raw, factor, forward_return, abnormal_return, price``.
    """
    out = compute_forward_return(df, forward_periods)
    out = winsorize_forward_return(out, lower=return_clip_pct[0], upper=return_clip_pct[1])
    out = compute_abnormal_return(out)

    # WHY: 保留原始因子值供後續 Profile 分析（如 Q1_Concentration 使用原始分佈）
    out = out.with_columns(pl.col("factor").alias("factor_raw"))

    out = mad_winsorize(out, n_mad=mad_n)
    out = cross_sectional_zscore(out)

    return out.select(
        pl.col("date").cast(pl.Datetime("ms")),
        pl.col("asset_id"),
        pl.col("factor_raw"),
        pl.col("factor_zscore").alias("factor"),
        pl.col("forward_return"),
        pl.col("abnormal_return"),
        pl.col("price"),
    )
