"""IC trend analysis using Theil-Sen estimator.

Aggregation: time-series only, Theil-Sen median pairwise slope on a
1-D series; CI from the rank-based pairwise slope distribution.

Input: DataFrame with ``date, value`` (typically an IC series).
Output: slope + confidence interval for trend detection.

Theil-Sen is preferred over OLS because it has a breakdown point of 29.3%,
making it robust to outliers (e.g. COVID-era IC spikes).

Matrix-row: ic_trend | (*, CONTINUOUS, *, TIMESERIES) | TS-only | Theil-Sen rank-based CI | _significance_marker, _short_circuit_output, _adf, _p_value_from_t
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._stats import _adf, _p_value_from_t, _significance_marker
from factrix._types import MetricOutput
from factrix.metrics._helpers import _short_circuit_output


def ic_trend(
    series: pl.DataFrame,
    value_col: str = "value",
    *,
    name: str = "ic_trend",
    adf_threshold: float | None = 0.10,
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
        adf_threshold: ADF p-value above which the input is flagged as
            unit-root suspect. Default ``0.10`` matches the conventional
            Stock-Watson (1988) cutoff: at p > 0.10 we cannot reject
            I(1), so OLS / Theil-Sen on the series reject the slope null
            at inflated rates regardless of the true trend. When
            ``None``, the ADF check is skipped entirely and no
            ``adf_stat`` / ``adf_p`` / ``unit_root_suspected`` keys are
            written. When a float is provided it must lie in (0, 1).
            Detected unit roots set ``unit_root_suspected=True`` in
            metadata; the slope value is still returned (caller decides)
            but significance should be read with scepticism.

    Returns:
        MetricOutput with value = slope, t_stat from Theil-Sen confidence interval.

    Notes:
        Theil-Sen median pairwise slope: ``slope = median{(y_j - y_i) /
        (j - i) : i < j}``. Approximate t-stat is reconstructed from the
        rank-based 95% confidence interval ``[low, high]``:
        ``SE ≈ (high - low) / 2 / 1.96`` and ``t ≈ slope / SE``. An ADF
        unit-root pre-check on the input flags series for which the slope
        null is rejected at inflated rates regardless of the true trend.

        factrix uses Theil-Sen rather than OLS because its 29.3% breakdown
        point absorbs IC outliers (e.g. COVID-era spikes) that would
        dominate an OLS slope; the trade-off is the SE recovered from the
        rank-CI is approximate, not asymptotically exact.

    References:
        [Sen 1968][sen-1968]: Theil-Sen median pairwise slope.
        [Lou-Polk 2022][lou-polk-2022]: factor-crowding decay
        mechanism that motivates a slope test on IC.
        [Stock-Watson 1988][stock-watson-1988]: practitioner
        unit-root concerns — backs the default `adf_threshold=0.10`.
        [Dickey-Fuller 1979][dickey-fuller-1979]: ADF persistence
        diagnostic on the input series.
        [MacKinnon 1996][mackinnon-1996]: ADF p-value response surface
        used by ``_adf_pvalue_interp``.
    """
    if adf_threshold is not None and not (0.0 < adf_threshold < 1.0):
        raise ValueError(
            f"adf_threshold must be a probability in (0, 1) or None, "
            f"got {adf_threshold!r}"
        )

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
            name,
            "insufficient_trend_periods",
            n_observed=n,
            min_required=10,
        )

    # WHY: 使用序號而非日期差，因為非重疊取樣後日期間距可能不均
    x = np.arange(n, dtype=float)

    result = sp_stats.theilslopes(vals, x)
    slope = float(result.slope)
    # WHY: scipy theilslopes 回傳 (slope, intercept, low_slope, high_slope)
    low_slope = float(result.low_slope)
    high_slope = float(result.high_slope)

    # WHY: 如果 CI 不跨零，slope 顯著
    ci_excludes_zero = (low_slope > 0 and high_slope > 0) or (
        low_slope < 0 and high_slope < 0
    )

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
    if adf_threshold is not None:
        adf_stat, adf_p = _adf(vals)
        metadata["adf_stat"] = adf_stat
        metadata["adf_p"] = adf_p
        metadata["unit_root_suspected"] = adf_p > adf_threshold
    return MetricOutput(
        name=name,
        value=slope,
        stat=approx_t,
        significance=_significance_marker(p),
        metadata=metadata,
    )
