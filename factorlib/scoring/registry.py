"""Metric registry, shared utilities, and sample-size constants."""

import logging

import numpy as np
import polars as pl
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def map_linear(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100
    return float(np.clip(score, 0, 100))


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
        .filter(pl.col("n") >= 10)
        .sort("date")
        .select("date", "ic")
    )


def _non_overlapping_dates(df: pl.DataFrame, step: int) -> pl.Series:
    """Sample every N-th date to avoid overlapping forward returns."""
    return df["date"].unique().sort().gather_every(step)


def _rolling_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Create rolling windows via numpy stride tricks (zero-copy)."""
    shape = (arr.shape[0] - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


# Minimum sample thresholds (in number of IC periods)
MIN_IC_PERIODS = 10          # Rank_IC, IC_IR
MIN_STABILITY_PERIODS = 12   # IC_Stability (adaptive window = n//3, so need >=12 for window>=4)
MIN_OOS_PERIODS = 5          # Internal_OOS_Decay (OOS partition)
MIN_PORTFOLIO_PERIODS = 5    # Long_Only_Alpha, MDD
