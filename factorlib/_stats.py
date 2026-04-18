"""Statistical significance tools for any numeric series.

Provides t-statistic computation and significance markers
(``***``/``**``/``*``). Operates on raw numeric arrays — agnostic
to what the series represents.

BHY multiple-testing lives in ``factorlib.stats.multiple_testing``;
it operates on *p-values* (profile-era) rather than the legacy
``bhy_threshold(t_stats)`` helper that was removed in the profile
migration.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factorlib._types import EPSILON, DDOF


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


def _newey_west_se(values: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC standard error for the mean of a time series.

    Uses Bartlett kernel weights: w_j = 1 - j/(L+1).

    Args:
        values: 1-D array of time series observations.
        lags: Number of lags. Defaults to floor(T^(1/3)).

    Returns:
        HAC-adjusted standard error of the mean.
    """
    n = len(values)
    if n < 2:
        return 0.0

    if lags is None:
        lags = int(np.floor(n ** (1 / 3)))
    lags = min(lags, n - 1)

    mean = float(np.mean(values))
    demeaned = values - mean

    # γ_0 = Var
    gamma_0 = float(np.dot(demeaned, demeaned)) / n

    # Weighted autocovariances: γ_j with Bartlett kernel
    weighted_sum = gamma_0
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weight = 1.0 - j / (lags + 1)
        weighted_sum += 2.0 * weight * gamma_j

    variance_of_mean = max(weighted_sum / n, 0.0)
    return float(np.sqrt(variance_of_mean))


def _newey_west_t_test(
    values: np.ndarray,
    lags: int | None = None,
) -> tuple[float, float, str]:
    """Newey-West t-test for H₀: mean = 0.

    Returns:
        (t_stat, p_value, significance_marker)
    """
    if len(values) < 3:
        return 0.0, 1.0, ""

    mean = float(np.mean(values))
    se = _newey_west_se(values, lags)
    if se < EPSILON:
        return 0.0, 1.0, ""

    t = mean / se
    p = _p_value_from_t(t, len(values))
    return t, p, _significance_marker(p)
