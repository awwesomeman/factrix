"""Statistical significance tools for any numeric series.

Provides t-statistic computation, significance markers (``***``/``**``/``*``),
and BHY multiple testing correction.
Operates on raw numeric arrays — agnostic to what the series represents.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factorlib.tools._typing import EPSILON, DDOF


def _calc_t_stat(mean: float, std: float, n: int) -> float:
    """Compute t-statistic with EPSILON guard against near-zero std.

    Args:
        mean: Sample mean.
        std: Sample standard deviation (ddof=1).
        n: Sample size.

    Returns:
        t-statistic, or 0.0 if std is near-zero or n ≤ 0.
    """
    if std > EPSILON and n > 0:
        return float(mean / (std / np.sqrt(n)))
    return 0.0


def _t_stat_from_array(values: np.ndarray) -> float:
    """Convenience: compute t-stat directly from a 1-D array.

    Args:
        values: 1-D numeric array with at least 2 elements.

    Returns:
        t-statistic of the mean, or 0.0 if insufficient data.
    """
    if len(values) < 2:
        return 0.0
    return _calc_t_stat(
        float(np.mean(values)),
        float(np.std(values, ddof=DDOF)),
        len(values),
    )


def _p_value_from_t(
    t_stat: float,
    n: int,
    alternative: str = "two-sided",
) -> float:
    """P-value from t-statistic using t-distribution.

    Args:
        alternative: "two-sided" (default), "less" (left-tail), "greater" (right-tail).
    """
    if n <= 1:
        return 1.0
    dof = n - 1
    if alternative == "less":
        return float(sp_stats.t.cdf(t_stat, dof))
    if alternative == "greater":
        return float(sp_stats.t.sf(t_stat, dof))
    return float(2 * sp_stats.t.sf(abs(t_stat), dof))


def _p_value_from_z(z: float) -> float:
    """Two-sided p-value from z-statistic using normal distribution."""
    return float(2 * sp_stats.norm.sf(abs(z)))


def _t_test_summary(
    mean: float, std: float, n: int,
) -> tuple[float, float, str]:
    """Compute t-stat, p-value, and significance marker in one call."""
    t = _calc_t_stat(mean, std, n)
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p)


def _significance_marker(p_value: float | None) -> str:
    """Map p-value to academic significance marker.

    | Marker | Condition   | Meaning              |
    |:------:|-------------|----------------------|
    | ``***``| p < 0.01    | Highly significant   |
    | ``**`` | p < 0.05    | Significant          |
    | ``*``  | p < 0.10    | Weakly significant   |
    |        | p >= 0.10   | Not significant      |

    Returns:
        One of ``"***"``, ``"**"``, ``"*"``, ``""``.
    """
    if p_value is None:
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def bhy_threshold(
    t_stats: np.ndarray,
    fdr: float = 0.05,
    min_obs: int = 60,
) -> float:
    """Benjamini-Hochberg-Yekutieli adjusted significance threshold.

    Returns the BHY-adjusted t-stat threshold for a given FDR level
    across a set of simultaneous tests. Factors with |t| >= threshold
    are significant after controlling for multiple testing.

    Uses normal approximation for p-values. This is valid when each
    t-stat comes from a regression with >= ``min_obs`` observations
    (t distribution → N(0,1) as dof → inf). If any input t-stat comes
    from a low-N regression, the threshold will be too permissive.

    Args:
        t_stats: 1-D array of t-statistics from multiple factor tests.
        fdr: Target false discovery rate (default 0.05).
        min_obs: Minimum observations per test for normal approximation
            to be valid. Used only for documentation/logging — caller
            is responsible for ensuring adequate sample sizes.

    Returns:
        Adjusted t-stat threshold. If no test passes, returns inf.

    References:
        Harvey, Liu & Zhu (2016): t > 3.0 as rough heuristic.
        Harvey & Liu (2020): Bayesian multiple testing for correlated factors.
        Chordia, Goyal & Saretto (2020): BHY correction for anomalies.
    """
    n = len(t_stats)
    if n == 0:
        return float("inf")

    # WHY: normal approximation is valid when all underlying t-stats have
    # dof >= ~30. With typical factor analysis (>60 obs), t → N(0,1).
    # Caller must ensure adequate sample sizes; this function cannot verify.
    pvals = 2 * sp_stats.norm.sf(np.abs(t_stats))

    # BHY correction: accounts for arbitrary dependence between tests
    c_m = float(np.sum(1.0 / np.arange(1, n + 1)))

    sorted_p = np.sort(pvals)

    # Vectorized BHY step-up: find largest k where p_(k) <= k / (m * c(m)) * fdr
    k_vec = np.arange(1, n + 1)
    bhy_crits = k_vec / (n * c_m) * fdr
    passing = sorted_p <= bhy_crits

    if not np.any(passing):
        return float("inf")

    k_max = int(np.max(np.where(passing)[0]))
    threshold_p = bhy_crits[k_max]

    return float(sp_stats.norm.isf(threshold_p / 2))
