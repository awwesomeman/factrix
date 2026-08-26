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


def _degenerate_t_input(std: float, n: int) -> bool:
    """True when ``(std, n)`` cannot support a t-statistic.

    Zero (or EPSILON-small) dispersion and a non-positive sample size are
    the two regimes where ``mean / (std / √n)`` has no finite value.
    :func:`_calc_t_stat` returns NaN on both; callers that own a
    ``MetricResult`` recognise that NaN afterwards and withhold the test
    (``factrix.metrics._helpers._degenerate_test_fields``).
    """
    return not (std > EPSILON and n > 0)


def _calc_t_stat(mean: float, std: float, n: int) -> float:
    """Compute t-statistic, or NaN when the sample cannot support one.

    A near-zero ``std`` is **not** evidence of a null effect. When every
    observation is identical and non-zero the sample is degenerate in the
    *maximum*-evidence direction (``t → ±∞``), and when it is identical and
    zero the ratio is an undefined ``0/0``. An earlier version returned
    ``0.0`` for both, which downstream turned into ``p = 1`` — reporting
    "no predictive power" for a sample that carries either overwhelming or
    undefined evidence, never the null.

    Reference behaviour splits on the same input but never lands on
    ``t = 0``: ``scipy.stats.ttest_1samp`` propagates (``t ≈ 1.3e16, p ≈ 0``
    for a constant non-zero sample; ``nan`` for a constant zero one), while
    R's ``t.test`` refuses with "data are essentially constant". This
    returns NaN — the soft form of R's refusal, and the value that cannot
    be mistaken for a finding. Metric callers pass the NaN to
    ``factrix.metrics._helpers._degenerate_test_fields``, which keeps the
    point estimate and reports ``stat=None`` / ``p_value=None`` under a
    ``degenerate_variance`` warning code. ±∞ is
    deliberately not used: it would spread through serialization,
    aggregation and plotting as a legitimate extreme value.

    Args:
        mean: Sample mean.
        std: Sample standard deviation (ddof=1).
        n: Sample size.

    Returns:
        t-statistic, or NaN when :func:`_degenerate_t_input` holds.
    """
    if _degenerate_t_input(std, n):
        return float("nan")
    return float(mean / (std / np.sqrt(n)))


def _t_stat_from_array(values: np.ndarray) -> float:
    """Convenience: compute t-stat directly from a 1-D array.

    Args:
        values: 1-D numeric array with at least 2 elements.

    Returns:
        t-statistic of the mean, or NaN if the sample is too short (< 2)
        or degenerate — see :func:`_calc_t_stat` for why not 0.0.
    """
    if len(values) < 2:
        return float("nan")
    return _calc_t_stat(
        float(np.mean(values)),
        float(np.std(values, ddof=DDOF)),
        len(values),
    )


def _p_value_from_t(
    t_stat: float,
    n: int,
    alternative: str = "two-sided",
    *,
    dof: float | None = None,
) -> float:
    """P-value from t-statistic using t-distribution.

    Args:
        alternative: "two-sided" (default), "less" (left-tail), "greater" (right-tail).
        dof: Residual degrees of freedom. Defaults to ``n - 1`` (single-sample
            mean t-test). Pass an explicit value for a regression t-test where
            ``k > 1`` parameters are estimated (``dof = n - k``), or a
            fractional effective df for a HAR t-test
            (:func:`factrix._stats.hac._har_dof`). Returns 1.0 if the
            resolved ``dof`` is below 1.
    """
    dof = n - 1 if dof is None else dof
    if dof < 1:
        return 1.0
    if alternative == "less":
        return float(sp_stats.t.cdf(t_stat, dof))
    if alternative == "greater":
        return float(sp_stats.t.sf(t_stat, dof))
    return float(2 * sp_stats.t.sf(abs(t_stat), dof))


def _p_value_from_z(z: float) -> float:
    """Two-sided p-value from z-statistic using normal distribution."""
    return float(2 * sp_stats.norm.sf(abs(z)))


def _binomial_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test p-value for ``H₀: p = p0``.

    Delegates to :func:`scipy.stats.binomtest` at every ``n`` (the
    minimum-likelihood two-sided convention, which is also what R's
    ``binom.test`` and statsmodels use).

    Why exact at every ``n``: an earlier version switched to the
    uncorrected normal-approximation score test above ``n = 20``. Without
    a continuity correction that branch is anti-conservative — at
    ``n=20, hits=15`` it reports ``p=0.025`` where the exact test gives
    ``0.041``; at ``n=50, hits=32`` ``0.048`` vs ``0.065`` — so headline
    p-values in the 0.02–0.15 band were systematically too small, with a
    step discontinuity at the cutoff. The exact test is O(n) and
    negligible at any realistic series length, so there is no reason to
    approximate.
    """
    if n <= 0:
        return 1.0
    return float(sp_stats.binomtest(hits, n, p0).pvalue)


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
