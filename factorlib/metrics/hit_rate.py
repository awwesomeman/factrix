"""Hit rate computation for any time-indexed series.

Input: DataFrame with ``date, value`` or a 1-D array.
Output: proportion of periods where the value satisfies a condition
(default: value > 0).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import MIN_IC_PERIODS, MetricOutput
from factorlib.metrics._helpers import _sample_non_overlapping
from factorlib._stats import _p_value_from_z, _significance_marker


def hit_rate(
    series: pl.DataFrame,
    value_col: str = "value",
    forward_periods: int = 5,
) -> MetricOutput:
    """Hit rate = proportion of periods where value > 0.

    Uses non-overlapping sampling to avoid autocorrelation.
    t-stat is from a binomial test: ``(p - 0.5) / sqrt(0.25 / n)``.

    Args:
        series: DataFrame with ``date`` and ``value_col``.
        forward_periods: Sampling interval for non-overlapping dates.

    Returns:
        MetricOutput with value = hit rate (0.0-1.0).
    """
    sampled = _sample_non_overlapping(series, forward_periods)
    vals = sampled[value_col].drop_nulls()

    n = len(vals)
    if n < MIN_IC_PERIODS:
        return MetricOutput(name="hit_rate", value=0.0, stat=0.0, significance="")

    hits = int((vals > 0).sum())
    rate = hits / n

    # WHY: 二項分布 t-stat，H₀: p = 0.5（隨機猜測）
    # (rate - 0.5) / sqrt(0.25/n) = (rate - 0.5) * sqrt(n) / 0.5
    t = float((rate - 0.5) * np.sqrt(n) / 0.5)

    p = _p_value_from_z(t)
    return MetricOutput(
        name="hit_rate",
        value=rate,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "n_hits": hits,
            "n_total": n,
            "p_value": p,
            "stat_type": "z",
            "h0": "p=0.5",
            "method": "binomial score test",
        },
    )
