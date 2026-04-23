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

from factrix._types import MetricOutput
from factrix.metrics._helpers import _short_circuit_output
from factrix._stats import _adf, _p_value_from_t, _significance_marker

# ADF p above this threshold flags "can't reject unit root" → downstream
# slope / t-stat on an I(1) series is Stock-Watson (1988) spurious.
_ADF_UNIT_ROOT_P: float = 0.10


def ic_trend(
    series: pl.DataFrame,
    value_col: str = "value",
    *,
    name: str = "ic_trend",
    adf_check: bool = True,
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
        adf_check: When True (default), run an Augmented Dickey-Fuller
            test on the input series. A unit root (ADF p > 0.10) makes
            the slope t-stat **spurious** in the Stock-Watson (1988)
            sense — OLS / Theil-Sen on an I(1) series reject the null
            at inflated rates regardless of the true trend. When
            detected, metadata flags ``unit_root_suspected=True`` and
            records ``adf_stat`` / ``adf_p``; the slope value is still
            returned (caller decides) but significance should be read
            with scepticism.

    Returns:
        MetricOutput with value = slope, t_stat from Theil-Sen confidence interval.

    References:
        Sen (1968), "Estimates of the Regression Coefficient Based on Kendall's Tau."
        Lou & Polk (2022), "Comomentum" — factor crowding/decay framework.
        Stock & Watson (1988), "Variable Trends in Economic Time Series."
    """
    sorted_s = series.sort("date").drop_nulls(subset=[value_col])
    vals = sorted_s[value_col].to_numpy()
    # polars drop_nulls does not drop float NaN; an all-NaN IC series
    # (e.g. from a constant factor whose per-date rank correlation is
    # degenerate) would otherwise flow into theilslopes and _adf and
    # trip LAPACK DLASCL before any short-circuit could save us.
    vals = vals[np.isfinite(vals)]
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
    metadata: dict = {
        "p_value": p,
        "stat_type": "t",
        "h0": "slope=0",
        "method": "theil-sen CI approximation",
        "n_periods": n,
        "ci_low": low_slope,
        "ci_high": high_slope,
        "ci_excludes_zero": ci_excludes_zero,
        "intercept": float(result.intercept),
    }
    if adf_check:
        adf_stat, adf_p = _adf(vals)
        metadata["adf_stat"] = adf_stat
        metadata["adf_p"] = adf_p
        metadata["unit_root_suspected"] = adf_p > _ADF_UNIT_ROOT_P
    return MetricOutput(
        name=name,
        value=slope,
        stat=approx_t,
        significance=_significance_marker(p),
        metadata=metadata,
    )
