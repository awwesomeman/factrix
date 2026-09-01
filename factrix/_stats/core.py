"""t-statistic, p-value, significance marker, and binomial test primitives.

Stateless helpers operating on raw numeric arrays — agnostic to what the
series represents. ``_p_value_from_t`` and ``_significance_marker`` are
the convert-to-p-value / attach-marker primitives shared with the
heteroskedasticity-and-autocorrelation-consistent (HAC) t-tests in ``factrix._stats.hac``.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
from scipy import stats as sp_stats

from factrix._errors import UserInputError
from factrix._types import DDOF, EPSILON, PValueAlternative

_P_VALUE_ALTERNATIVES: tuple[str, ...] = get_args(PValueAlternative)


def _validate_p_value_alternative(
    alternative: object,
    *,
    func_name: str,
) -> None:
    """Reject a runtime value outside :class:`PValueAlternative`."""
    if alternative not in _P_VALUE_ALTERNATIVES:
        raise UserInputError(
            func_name=func_name,
            field="alternative",
            value=alternative,
            candidates=_P_VALUE_ALTERNATIVES,
            docs_path="reference/statistical-methods",
        )


def _degenerate_t_input(std: float, n: int) -> bool:
    """True when ``(std, n)`` cannot support a t-statistic.

    This is the canonical guard for scalar standard errors and dispersions
    used in t-statistic construction. It deliberately uses a positive
    finite-value test: NaN is unordered, so a bare ``std < EPSILON`` would
    let it pass, while infinity cannot support meaningful inference either.
    Callers should use this helper instead of repeating a threshold comparison.

    :func:`_calc_t_stat` returns NaN for non-finite or EPSILON-small
    dispersion and for a non-positive sample size. Callers that own a
    ``MetricResult`` recognise that NaN afterwards and withhold the test
    (``factrix.metrics._helpers._degenerate_test_fields``).
    """
    return not (np.isfinite(std) and std > EPSILON and n > 0)


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
    alternative: PValueAlternative = "two-sided",
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
    _validate_p_value_alternative(alternative, func_name="_p_value_from_t")
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


def _mann_kendall_hamed_rao(values: np.ndarray) -> tuple[float, float, float]:
    """[Hamed-Rao (1998)][hamed-rao-1998] serial-correlation-corrected Mann-Kendall trend test.

    The classical Mann-Kendall test ([Mann (1945)][mann-1945],
    [Kendall (1975)][kendall-1975]) assumes serially independent
    observations: its null variance ``Var(S) = n(n-1)(2n+5)/18`` counts
    every one of the ``n(n-1)/2`` pairwise comparisons as an independent
    draw. On a persistent series that is far too small, and the test
    over-rejects badly — measured 34% at a nominal 5% on AR(0.6), 67% on
    AR(0.9), 37-68% on the overlapping IC series ``ic_trend`` consumes by
    default.

    Hamed-Rao inflate the variance by an effective-sample-size factor
    computed from the autocorrelation of the *detrended ranks*:

        ``n/n* = 1 + 2/(n(n-1)(n-2)) · Σ_s (n-s)(n-s-1)(n-s-2) · ρ_s``

    summed over lags whose ``|ρ_s|`` clears the ``2/√n`` significance
    bound (Hamed-Rao §3: insignificant lags contribute noise, not
    dependence). Detrending uses the Theil-Sen slope so the trend under
    test does not itself masquerade as persistence. The factor is floored
    at 1 — a correction that *shrinks* the null variance would make the
    test anti-conservative, which is the opposite of the point.

    Known limit, measured, disclosed: Hamed-Rao under-corrects at
    research sample sizes. It takes AR(0.9) at T=120 from 67% to 36% and
    the h=21 overlapping series from 68% to 27% — a large improvement,
    not a calibrated test. Callers that know their overlap horizon should
    sub-sample to non-overlapping observations *first* (which is exactly
    calibrated for that channel: 2.2-4.5% across the same grid) and use
    this correction for the residual persistence, which is what
    ``factrix.metrics.trend.ic_trend`` does.

    Args:
        values: 1-D series in time order, at least 3 finite observations.

    Returns:
        ``(tau, p_value, variance_inflation)`` — ``tau`` is Kendall's tau
        between the sequence index and the series (the effect size,
        unchanged by the correction), ``p_value`` the two-sided p from the
        continuity-corrected normal statistic
        ``z = (S - sign(S)) / √Var*(S)``, and ``variance_inflation`` the
        ``n/n*`` factor for metadata. A constant series admits no rank
        ordering: all three are NaN, and callers map that to a withheld
        test under ``DEGENERATE_VARIANCE``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    index = np.arange(n, dtype=float)
    tau = float(sp_stats.kendalltau(index, values).statistic)
    if not np.isfinite(tau):
        return float("nan"), float("nan"), float("nan")

    inflation = _hamed_rao_variance_inflation(values, index)

    # Mann-Kendall S and its tie-corrected iid null variance.
    s_stat = float(
        sum(np.sum(np.sign(values[i + 1 :] - values[i])) for i in range(n - 1))
    )
    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1].astype(float)
    var_s = (
        n * (n - 1) * (2 * n + 5) - float(np.sum(ties * (ties - 1) * (2 * ties + 5)))
    ) / 18.0
    var_s *= inflation
    if var_s <= EPSILON:
        return tau, float("nan"), inflation

    z = (s_stat - np.sign(s_stat)) / np.sqrt(var_s)
    return tau, float(2 * sp_stats.norm.sf(abs(z))), inflation


def _hamed_rao_variance_inflation(values: np.ndarray, index: np.ndarray) -> float:
    """``n/n*`` variance-inflation factor for :func:`_mann_kendall_hamed_rao`."""
    n = len(values)
    slope = float(sp_stats.theilslopes(values, index).slope)
    ranks = sp_stats.rankdata(values - slope * index)
    ranks = ranks - ranks.mean()
    denom = float(np.dot(ranks, ranks))
    if denom < EPSILON or n < 4:
        return 1.0
    # Hamed-Rao §3 keep only autocorrelations clearing the +/-2/sqrt(n)
    # bound; including insignificant lags adds estimation noise to the
    # correction without adding dependence.
    bound = 2.0 / np.sqrt(n)
    accumulated = 0.0
    for lag in range(1, n - 2):
        rho = float(np.dot(ranks[lag:], ranks[:-lag])) / denom
        if abs(rho) <= bound:
            continue
        accumulated += (n - lag) * (n - lag - 1) * (n - lag - 2) * rho
    inflation = 1.0 + 2.0 * accumulated / (n * (n - 1) * (n - 2))
    return max(inflation, 1.0)
