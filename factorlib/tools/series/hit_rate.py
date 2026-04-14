"""Hit rate computation for any time-indexed series.

Input: DataFrame with ``date, value`` or a 1-D array.
Output: proportion of periods where the value satisfies a condition
(default: value > 0).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib.tools._typing import MIN_IC_PERIODS, MetricOutput
from factorlib.tools._helpers import sample_non_overlapping
from factorlib.tools.series.significance import significance_marker


def compute_hit_rate(
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
    sampled = sample_non_overlapping(series, forward_periods)
    vals = sampled[value_col].drop_nulls()

    n = len(vals)
    if n < MIN_IC_PERIODS:
        return MetricOutput(name="Hit_Rate", value=0.0, t_stat=0.0, significance="")

    hits = int((vals > 0).sum())
    rate = hits / n

    # WHY: 二項分布 t-stat，H₀: p = 0.5（隨機猜測）
    # (rate - 0.5) / sqrt(0.25/n) = (rate - 0.5) * sqrt(n) / 0.5
    t = float((rate - 0.5) * np.sqrt(n) / 0.5)

    return MetricOutput(
        name="Hit_Rate",
        value=rate,
        t_stat=t,
        significance=significance_marker(t),
        metadata={"n_hits": hits, "n_total": n},
    )
