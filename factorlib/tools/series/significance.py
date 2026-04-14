"""Statistical significance tools for any numeric series.

Provides t-statistic computation, significance markers (``***``/``**``/``*``),
and BHY multiple testing correction.
Operates on raw numeric arrays — agnostic to what the series represents.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factorlib.tools._typing import EPSILON, DDOF


def calc_t_stat(mean: float, std: float, n: int) -> float:
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


def t_stat_from_array(values: np.ndarray) -> float:
    """Convenience: compute t-stat directly from a 1-D array.

    Args:
        values: 1-D numeric array with at least 2 elements.

    Returns:
        t-statistic of the mean, or 0.0 if insufficient data.
    """
    if len(values) < 2:
        return 0.0
    return calc_t_stat(
        float(np.mean(values)),
        float(np.std(values, ddof=DDOF)),
        len(values),
    )


def significance_marker(t_stat: float | None) -> str:
    """Map t-stat to academic significance marker.

    | Marker | Condition     | Meaning                            |
    |:------:|---------------|------------------------------------|
    | ``***``| |t| >= 3.0    | Highly significant (Harvey-strict) |
    | ``**`` | |t| >= 2.0    | Significant (p < 0.05)             |
    | ``*``  | |t| >= 1.65   | Weakly significant (p < 0.10)      |
    |        | |t| < 1.65    | Not significant                    |

    Returns:
        One of ``"***"``, ``"**"``, ``"*"``, ``""``.
    """
    if t_stat is None:
        return ""
    abs_t = abs(t_stat)
    if abs_t >= 3.0:
        return "***"
    if abs_t >= 2.0:
        return "**"
    if abs_t >= 1.65:
        return "*"
    return ""


def bhy_threshold(
    t_stats: np.ndarray,
    fdr: float = 0.05,
) -> float:
    """Benjamini-Hochberg-Yekutieli adjusted significance threshold.

    Returns the BHY-adjusted t-stat threshold for a given FDR level
    across a set of simultaneous tests. Factors with |t| >= threshold
    are significant after controlling for multiple testing.

    Args:
        t_stats: 1-D array of t-statistics from multiple factor tests.
        fdr: Target false discovery rate (default 0.05).

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

    # WHY: use normal instead of t-distribution because the input t-stats come
    # from different regressions with varying (unknown) dof. With typical factor
    # analysis sample sizes (>60 observations), t → N(0,1) is a safe assumption.
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
