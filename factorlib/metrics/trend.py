"""IC trend analysis using Theil-Sen estimator.

Input: DataFrame with ``date, value`` (typically an IC series).
Output: slope + confidence interval for trend detection.

Theil-Sen is preferred over OLS because it has a breakdown point of 29.3%,
making it robust to outliers (e.g. COVID-era IC spikes).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factorlib._types import MetricOutput
from factorlib.metrics._helpers import _short_circuit_output
from factorlib._stats import _p_value_from_t, _significance_marker


def ic_trend(
    series: pl.DataFrame,
    value_col: str = "value",
    *,
    name: str = "ic_trend",
) -> MetricOutput:
    """Theil-Sen median slope of a time-indexed series.

    Answers "is this factor getting better or worse over time?"
    - slope ≈ 0: stable
    - slope significantly < 0: decaying (crowding / alpha erosion)
    - slope significantly > 0: improving

    Args:
        series: DataFrame with ``date`` and ``value_col``.
        name: MetricOutput.name for the returned output. Defaults to
            ``"ic_trend"``; EventFactor.caar_trend / MacroPanelFactor.
            beta_trend pass their own names so method / cache key /
            primitive name stay three-point unified.

    Returns:
        MetricOutput with value = slope, t_stat from Theil-Sen confidence interval.

    References:
        Sen (1968), "Estimates of the Regression Coefficient Based on Kendall's Tau."
        Lou & Polk (2022), "Comomentum" — factor crowding/decay framework.
    """
    sorted_s = series.sort("date").drop_nulls(subset=[value_col])
    vals = sorted_s[value_col].to_numpy()
    n = len(vals)

    if n < 10:
        return _short_circuit_output(
            name, "insufficient_trend_periods",
            n_observed=n, min_required=10,
        )

    # WHY: 使用序號而非日期差，因為非重疊取樣後日期間距可能不均
    x = np.arange(n, dtype=float)

    result = sp_stats.theilslopes(vals, x)
    slope = float(result.slope)
    # WHY: scipy theilslopes 回傳 (slope, intercept, low_slope, high_slope)
    low_slope = float(result.low_slope)
    high_slope = float(result.high_slope)

    # WHY: 如果 CI 不跨零，slope 顯著
    ci_excludes_zero = (low_slope > 0 and high_slope > 0) or (low_slope < 0 and high_slope < 0)

    # WHY: 從 CI 推導近似 t-stat 供顯著性標記使用
    # slope ± margin = CI → margin = (high - low) / 2 → SE ≈ margin / 1.96
    margin = (high_slope - low_slope) / 2
    if margin > 0:
        approx_t = slope / (margin / 1.96)
    else:
        approx_t = 0.0

    p = _p_value_from_t(approx_t, n)
    return MetricOutput(
        name=name,
        value=slope,
        stat=approx_t,
        significance=_significance_marker(p),
        metadata={
            "p_value": p,
            "stat_type": "t",
            "h0": "slope=0",
            "method": "theil-sen CI approximation",
            "n_periods": n,
            "ci_low": low_slope,
            "ci_high": high_slope,
            "ci_excludes_zero": ci_excludes_zero,
            "intercept": float(result.intercept),
        },
    )
