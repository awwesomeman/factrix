"""t-statistic, p-value, significance marker, and binomial test primitives.

Stateless helpers operating on raw numeric arrays — agnostic to what the
series represents. ``_p_value_from_t`` and ``_significance_marker`` are
the convert-to-p-value / attach-marker primitives shared with the
heteroskedasticity-and-autocorrelation-consistent (HAC) t-tests in ``factrix._stats.hac``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factrix._types import DDOF, EPSILON


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


# Below this sample count, the normal approximation to the binomial
# systematically mis-sizes the test (≈5% actual α at nominal 5% only
# kicks in around n≥20; smaller n is liberal). Use exact binomial CDF
# when we fall below.
_BINOMIAL_EXACT_CUTOFF: int = 20


def _binomial_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    """Two-sided binomial test p-value for ``H₀: p = p0``.

    Uses the exact binomial CDF for ``n < _BINOMIAL_EXACT_CUTOFF`` and
    the normal-approximation ``z = (p̂ − p0) / √(p0(1−p0)/n)`` for larger
    samples. For p0 = 0.5 the two tails are symmetric; otherwise scipy's
    ``binomtest`` handles the asymmetric two-sided convention.
    """
    if n <= 0:
        return 1.0
    if n < _BINOMIAL_EXACT_CUTOFF:
        return float(sp_stats.binomtest(hits, n, p0).pvalue)
    rate = hits / n
    denom = float(np.sqrt(p0 * (1.0 - p0) / n))
    if denom < EPSILON:
        return 1.0
    z = (rate - p0) / denom
    return _p_value_from_z(z)


def _binomial_test_method_name(n: int) -> str:
    """Human-readable test name mirroring the branch in ``_binomial_two_sided_p``."""
    return (
        "binomial exact test"
        if n < _BINOMIAL_EXACT_CUTOFF
        else "binomial score test (normal approximation)"
    )


def _t_test_summary(
    mean: float,
    std: float,
    n: int,
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
