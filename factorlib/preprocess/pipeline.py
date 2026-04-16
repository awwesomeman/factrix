"""Preprocessing orchestration.

Each step is independently importable from ``returns`` and ``normalize``.

Expects canonical column names (date, asset_id, price, factor).
Use ``adapt()`` to rename before calling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from factorlib.preprocess.returns import (
    compute_abnormal_return,
    compute_forward_return,
    winsorize_forward_return,
)
from factorlib.preprocess.normalize import (
    cross_sectional_zscore,
    mad_winsorize,
)

if TYPE_CHECKING:
    from factorlib.config import BaseConfig


def preprocess(
    df: pl.DataFrame,
    *,
    config: BaseConfig | None = None,
) -> pl.DataFrame:
    """Preprocess factor data based on config type.

    Dispatches to the appropriate type-specific preprocessor.
    Defaults to cross-sectional preprocessing.
    """
    from factorlib.config import CrossSectionalConfig

    if config is None:
        config = CrossSectionalConfig()

    match config:
        case CrossSectionalConfig():
            return preprocess_cs_factor(df, config=config)
        case _:
            ft = type(config).factor_type
            raise NotImplementedError(
                f"preprocess not yet implemented for {ft}"
            )


def preprocess_cs_factor(
    df: pl.DataFrame,
    *,
    config: CrossSectionalConfig | None = None,
    forward_periods: int = 5,
    return_clip_pct: tuple[float, float] = (0.01, 0.99),
    mad_n: float = 3.0,
) -> pl.DataFrame:
    """Run the full Step 1-5 preprocessing pipeline for cross-sectional factors.

    Steps:
        1. Forward return (``(price[t+N] / price[t] - 1) / N``, per-period).
        2. Forward return percentile winsorization.
        3. Abnormal return (cross-sectional de-mean).
        4. Factor MAD winsorization.
        5. Cross-sectional MAD z-score.

    Args:
        df: Data with canonical columns ``date``, ``asset_id``, ``price``,
            ``factor``. Use ``adapt()`` to rename if needed.
        config: If provided, ``forward_periods``, ``return_clip_pct``, and
            ``mad_n`` are taken from *config* (keyword args are ignored).
            This ensures a single source of truth with downstream tools.
        forward_periods: Number of periods for forward return (default 5).
        return_clip_pct: (lower, upper) quantile bounds for return clipping.
        mad_n: Number of MAD units for factor winsorization (0 to disable).

    Returns:
        DataFrame with columns:
        ``date, asset_id, factor_raw, factor, forward_return, abnormal_return, price``.
    """
    if config is not None:
        forward_periods = config.forward_periods
        return_clip_pct = config.return_clip_pct
        mad_n = config.mad_n

    out = compute_forward_return(df, forward_periods)
    out = winsorize_forward_return(out, lower=return_clip_pct[0], upper=return_clip_pct[1])
    out = compute_abnormal_return(out)

    # WHY: 保留原始因子值供後續 Profile 分析（如 Q1_Concentration 使用原始分佈）
    out = out.with_columns(pl.col("factor").alias("factor_raw"))

    out = mad_winsorize(out, n_mad=mad_n)
    out = cross_sectional_zscore(out)

    return out.select(
        pl.col("date"),
        pl.col("asset_id"),
        pl.col("factor_raw"),
        pl.col("factor_zscore").alias("factor"),
        pl.col("forward_return"),
        pl.col("abnormal_return"),
        pl.col("price"),
    )
