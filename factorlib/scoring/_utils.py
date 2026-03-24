"""Shared metric utilities, constants, and sample-size thresholds."""

import logging
from typing import NamedTuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# WHY: 統一除零保護門檻，避免 std ≈ 0 時 t-stat 膨脹至天文數字
# （sigmoid adaptive_weight 會將極端 t-stat 映射為權重 1.0，注入虛假信心）
EPSILON: float = 1e-9

# WHY: ddof=1 為樣本標準差（業界主流），全專案統一避免跨指標比較產生系統性偏差
# Polars .std() 預設 ddof=1，NumPy 需顯式傳入
DDOF: int = 1

# WHY: 日曆天年化因子，集中管理避免魔術數字散落各指標
CALENDAR_DAYS_PER_YEAR: float = 365.25

# ---------------------------------------------------------------------------
# Minimum sample thresholds (in number of IC periods)
# ---------------------------------------------------------------------------

MIN_IC_PERIODS = 10          # Rank_IC, IC_IR
MIN_STABILITY_PERIODS = 12   # IC_Stability (adaptive window = n//3, so need >=12 for window>=4)
MIN_OOS_PERIODS = 5          # OOS_Decay (OOS partition)
MIN_PORTFOLIO_PERIODS = 5    # Long_Alpha, MDD
MIN_EVENT_SAMPLE = 5         # Event_CAAR, Event_KS, Event_Hit_Rate, Profit_Factor
MIN_EVENT_DECAY_PERIODS = 15 # Event_Decay (IS/OOS split needs enough dates)
# WHY: 80/20 split 下需 ≥3 OOS dates（15×0.2=3）才能估計 std_oos 計算 t_stat
MIN_EVENT_MONTHS = 3         # Event_Stability (monthly CV)
MIN_EVENT_DATES = 3          # Event_CAR_Dispersion (cross-sectional dates)
MIN_SKEW_PERIODS = 30        # Event_Skewness: 偏態估計需要較大樣本才穩定


# ---------------------------------------------------------------------------
# Metric return type
# ---------------------------------------------------------------------------

class MetricResult(NamedTuple):
    """Unified return type for all scoring metrics.

    All registered metric functions must return ``MetricResult | None``.
    Returning ``None`` signals insufficient data — the scorer auto-skips
    and redistributes weight.
    """
    score: float
    t_stat: float | None = None
    raw_value: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calc_t_stat(mean: float, std: float, n: int) -> float:
    """Compute t-statistic with EPSILON guard against near-zero std."""
    if std > EPSILON and n > 0:
        return float(mean / (std / np.sqrt(n)))
    return 0.0


def _ic_series(df: pl.DataFrame) -> pl.DataFrame:
    """Per-date Spearman Rank IC (Pure Polars).
    Uses method='average' for correct Spearman with tied values.
    """
    ranked = df.with_columns(
        pl.col("factor").rank(method="average").over("date").alias("rank_factor"),
        pl.col("forward_return").rank(method="average").over("date").alias("rank_return"),
    )
    return (
        ranked.group_by("date")
        .agg(
            pl.corr("rank_factor", "rank_return").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= MIN_IC_PERIODS)
        .sort("date")
        .select("date", "ic")
    )


def _non_overlapping_ic_tstat(ic_df: pl.DataFrame, forward_periods: int) -> float:
    """T-stat from non-overlapping IC samples.

    WHY: non-overlapping sampling eliminates residual autocorrelation from
    overlapping forward returns, preventing t-stat inflation.
    """
    sampled_dates = ic_df["date"].sort().gather_every(forward_periods)
    sampled_ic = ic_df.filter(pl.col("date").is_in(sampled_dates.implode()))["ic"].drop_nulls()
    n_sampled = len(sampled_ic)
    if n_sampled < 2:
        return 0.0
    return calc_t_stat(float(sampled_ic.mean()), float(sampled_ic.std()), n_sampled)


def _non_overlapping_dates(df: pl.DataFrame, step: int) -> pl.Series:
    """Sample every N-th date to avoid overlapping forward returns."""
    return df["date"].unique().sort().gather_every(step)


def _signed_event_cars(
    df: pl.DataFrame,
    return_col: str = "forward_return",
) -> pl.Series | None:
    """signed_car = {return_col} × sign(factor_raw)。

    WHY: 使用 forward_return（非 abnormal_return）確保單標的與多標的通用。
    abnormal_return 在單標的時恆為 0（mean(1 個值) = 自身）。
    future: 可傳入 return_col="abnormal_return" 啟用橫截面語意。

    Buy(+1) 正確→正貢獻；Sell(-1) 正確→也正貢獻，方向不相消。
    回傳 drop_nulls 後的 Series，若欄位缺失或事件為空則回傳 None。
    """
    if return_col not in df.columns or "factor_raw" not in df.columns:
        return None
    events = df.filter(pl.col("factor_raw") != 0)
    if events.is_empty():
        return None
    return (
        events.with_columns(
            (pl.col(return_col) * pl.col("factor_raw").sign()).alias("signed_car")
        )["signed_car"].drop_nulls()
    )


def _rolling_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Create rolling windows via numpy stride tricks (zero-copy).

    Raises:
        ValueError: If array length < window (as_strided has no bounds check).
    """
    if arr.shape[0] < window:
        raise ValueError(
            f"Array length {arr.shape[0]} < window {window}; "
            "caller must guard with a minimum-sample check"
        )
    shape = (arr.shape[0] - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
