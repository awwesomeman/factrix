"""Heteroskedasticity- and autocorrelation-consistent (HAC) standard errors.

Newey-West (Bartlett kernel) and Hansen-Hodrick (rectangular kernel)
HAC SE / t-test for the mean of a (possibly overlapping) time series.
``_resolve_nw_lags`` is the shared bandwidth picker honouring the
forward-overlap horizon.
"""

from __future__ import annotations

import numpy as np

from factrix._stats.constants import auto_bartlett
from factrix._stats.core import _p_value_from_t, _significance_marker
from factrix._types import EPSILON


def _resolve_nw_lags(
    n: int,
    lags: int | None,
    forward_periods: int | None,
) -> int:
    """Pick Bartlett-kernel bandwidth, honoring the overlap horizon.

    ``max(auto_bartlett(T), forward_periods - 1)`` when ``forward_periods``
    is provided; the Newey-West (1994) auto rule supplies the default
    Bartlett bandwidth and the ``h - 1`` floor is required for consistency
    when input series carries an MA(h-1) structure from overlapping forward
    returns. Clipped to ``n - 1`` so the kernel stays inside the sample.
    """
    if n < 2:
        return 0
    base = auto_bartlett(n) if lags is None else lags
    if forward_periods is not None:
        base = max(base, max(forward_periods - 1, 0))
    return max(0, min(base, n - 1))


def _newey_west_se(
    values: np.ndarray,
    lags: int | None = None,
    forward_periods: int | None = None,
) -> float:
    """Newey-West HAC standard error for the mean of a time series.

    Uses Bartlett kernel weights: w_j = 1 - j/(L+1).

    Args:
        values: 1-D array of time series observations.
        lags: Number of lags. Defaults to ``auto_bartlett(T)``.
        forward_periods: Overlap horizon of the input series. When set,
            enforces ``lags >= forward_periods - 1`` — the minimum
            consistent bandwidth for overlapping h-period returns
            ([Hansen-Hodrick (1980)][hansen-hodrick-1980] MA(h-1) structure).

    Returns:
        HAC-adjusted standard error of the mean.
    """
    n = len(values)
    if n < 2:
        return 0.0

    lags = _resolve_nw_lags(n, lags, forward_periods)

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
    forward_periods: int | None = None,
) -> tuple[float, float, str]:
    """Newey-West t-test for H₀: mean = 0.

    Args:
        values: 1-D array of time series observations.
        lags: Optional explicit Bartlett-kernel bandwidth. ``None`` uses
            the Newey-West (1994) ``auto_bartlett(T)`` default.
        forward_periods: Overlap horizon of the series. When set,
            bandwidth is floored at ``forward_periods - 1`` to stay
            consistent under the MA(h-1) overlap structure.

    Returns:
        (t_stat, p_value, significance_marker)
    """
    from factrix._logging import get_metrics_logger

    n = len(values)
    if n < 3:
        return 0.0, 1.0, ""

    effective_lags = _resolve_nw_lags(n, lags, forward_periods)
    logger = get_metrics_logger()
    logger.debug("newey_west_t_test: n=%d lags=%d", n, effective_lags)
    # WARNING: NW kernel needs enough samples per lag to estimate
    # autocovariances; a crude but standard rule is T >= 5 * lags.
    if effective_lags > 0 and n < 5 * effective_lags:
        logger.warning(
            "newey_west_t_test: n=%d < 5 * lags=%d — HAC estimate may be "
            "poorly conditioned. Consider smaller lags or more data.",
            n,
            effective_lags,
        )

    mean = float(np.mean(values))
    se = _newey_west_se(values, lags, forward_periods=forward_periods)
    if se < EPSILON:
        return 0.0, 1.0, ""

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p)


def _hansen_hodrick_se(
    values: np.ndarray,
    forward_periods: int,
) -> tuple[float, bool]:
    """[Hansen-Hodrick (1980)][hansen-hodrick-1980] rectangular-kernel HAC SE for a sample mean.

    Closed-form variance under the textbook MA(h-1) overlap structure
    induced by h-period forward returns:

        Var(mean) = (γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n,    h = forward_periods

    Unlike the Bartlett kernel used by ``_newey_west_se``, weights are
    flat (1.0) inside ``j ≤ h-1`` and zero beyond. The estimator carries
    no PSD guarantee ([Andrews (1991)][andrews-1991] §3): on short / mildly anti-correlated
    samples the parenthesised sum can come out negative. Callers may map
    ``clamped=True`` to a degenerate-sample warning.

    Args:
        values: 1-D array of the overlapping series whose mean is tested.
        forward_periods: Overlap horizon ``h``. Must be ≥ 1; ``h = 1``
            collapses to the iid SE (no autocovariance terms).

    Returns:
        ``(se, clamped)`` — clamped variance √max(., 0); ``clamped`` is
        ``True`` iff the raw variance estimate was < 0.
    """
    n = len(values)
    if n < 2 or forward_periods < 1:
        return 0.0, False

    mean = float(np.mean(values))
    demeaned = values - mean

    gamma_0 = float(np.dot(demeaned, demeaned)) / n
    weighted_sum = gamma_0
    lags = min(forward_periods - 1, n - 1)
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weighted_sum += 2.0 * gamma_j

    variance_of_mean = weighted_sum / n
    clamped = variance_of_mean < 0.0
    return float(np.sqrt(max(variance_of_mean, 0.0))), clamped


def _bartlett_lrcov(scores_per_period: np.ndarray, lags: int) -> np.ndarray:
    r"""Newey-West (Bartlett-kernel) long-run covariance of a ``(T, K)`` vector sequence.

    For a chronologically ordered sequence $h_1, \dots, h_T$ of
    $K$-vectors:

        $$
        S = \Omega_0 + \sum_{j=1}^{L}\Bigl(1 - \tfrac{j}{L+1}\Bigr)
            (\Omega_j + \Omega_j'),\qquad
        \Omega_j = \sum_{t=j+1}^{T} h_t\,h_{t-j}'.
        $$

    Returns the $K\times K$ matrix $S$ as a *sum* (not time-averaged), so
    a sandwich $(X'X)^{-1} S (X'X)^{-1}$ stays correctly scaled when
    $X'X$ is itself a sum over the same observations — the
    Driscoll-Kraay use below. The Bartlett weight keeps $S$ positive
    semi-definite ([Newey-West (1987)][newey-west-1987]).

    Args:
        scores_per_period: ``(T, K)`` array; row ``t`` is the period-``t``
            vector $h_t$. Rows must already be in time order.
        lags: Bartlett bandwidth $L$. ``0`` collapses to $\Omega_0$
            (the no-autocorrelation White form). Lags beyond ``T - 1``
            contribute nothing and are skipped.
    """
    H = np.atleast_2d(scores_per_period)
    T = H.shape[0]
    cov = H.T @ H  # Ω_0
    max_lag = min(lags, T - 1)
    for j in range(1, max_lag + 1):
        omega_j = H[j:].T @ H[:-j]
        weight = 1.0 - j / (lags + 1)
        cov = cov + weight * (omega_j + omega_j.T)
    return cov


def _driscoll_kraay_cov(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    lags: int | None = None,
) -> tuple[np.ndarray, int, int]:
    r"""[Driscoll & Kraay (1998)][driscoll-kraay-1998] cross-section-robust HAC covariance for a pooled OLS fit.

    Aggregates the per-observation OLS scores
    $u_{it} = x_{it}\,\hat e_{it}$ cross-sectionally within each period to
    $h_t = \sum_{i} u_{it}$, runs a Bartlett-kernel HAC
    (:func:`_bartlett_lrcov`) on the $T$-length sequence of $K$-vectors
    $h_t$, and sandwiches with $(X'X)^{-1}$:

        $$
        V = (X'X)^{-1}\,\hat S_T\,(X'X)^{-1},\qquad
        \hat S_T = \hat\Omega_0 + \sum_{j=1}^{m}\Bigl(1-\tfrac{j}{m+1}\Bigr)
                   (\hat\Omega_j + \hat\Omega_j').
        $$

    Robust to **arbitrary contemporaneous cross-sectional correlation**
    (and serial correlation up to lag $m$): collapsing each period's
    cross-section into the single sum $h_t$ folds the within-period
    dependence into one $K$-vector per period, so the SE only needs the
    time-series HAC of that sequence. This is the gap a one-way
    cluster-on-date SE leaves open — clustering on date treats periods as
    independent and so understates SE when shocks persist across periods,
    while DK is robust to both axes at once.

    Args:
        X: ``(N, K)`` pooled design matrix.
        resid: ``(N,)`` OLS residuals $\hat e_{it}$.
        time_ids: ``(N,)`` period label per row. Cross-sectional sums are
            taken within each distinct label; ``np.unique`` ordering
            (sorted) sets the chronological order of the HAC sequence, so
            sortable date labels keep the lag structure honest.
        lags: Bartlett bandwidth $m$. ``None`` → [Newey-West
            (1994)][newey-west-1994] auto-bandwidth ``auto_bartlett(T)``
            on the *period* count $T$ (not the row count $N$). Clipped to
            ``[0, T - 1]``.

    Returns:
        ``(cov, n_periods, lags_used)`` — ``cov`` is the $K\times K$
        covariance $V$; ``n_periods`` is the number of distinct
        ``time_ids``; ``lags_used`` is the resolved bandwidth $m$.

    Raises:
        numpy.linalg.LinAlgError: ``X'X`` is singular.

    References:
        - [Driscoll & Kraay (1998)][driscoll-kraay-1998]. "Consistent
          Covariance Matrix Estimation with Spatially Dependent Panel
          Data." Review of Economics and Statistics, 80(4), 549–560.
    """
    from factrix._stats.constants import auto_bartlett

    scores = X * resid[:, None]  # (N, K) per-obs score u_it
    # Sum scores within each period → H (T, K). Sorted unique labels give
    # the chronological order the Bartlett lags assume.
    uniq, inverse = np.unique(time_ids, return_inverse=True)
    n_periods = len(uniq)
    cross_section_sums = np.zeros((n_periods, X.shape[1]))
    np.add.at(cross_section_sums, inverse.ravel(), scores)

    lags_used = auto_bartlett(n_periods) if lags is None else lags
    lags_used = max(0, min(lags_used, n_periods - 1))

    long_run_cov = _bartlett_lrcov(cross_section_sums, lags_used)
    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ long_run_cov @ xtx_inv
    return cov, n_periods, lags_used


def _hansen_hodrick_t_test(
    values: np.ndarray,
    forward_periods: int,
) -> tuple[float, float, str, bool]:
    """Hansen-Hodrick t-test for ``H₀: mean = 0`` on an overlapping series.

    Returns ``(t, p, marker, clamped)``. The 4-tuple deviates from
    ``_newey_west_t_test``'s 3-tuple deliberately: rectangular-kernel
    variance has no PSD guarantee and callers must surface the clamp
    case as a warning rather than silently treat it as a non-rejection.
    SE → 0 (whether by near-zero raw variance or by clamping) returns
    ``(0.0, 1.0, "", clamped)`` — the conservative "cannot reject"
    direction.
    """
    n = len(values)
    if n < 3 or forward_periods < 1:
        return 0.0, 1.0, "", False

    mean = float(np.mean(values))
    se, clamped = _hansen_hodrick_se(values, forward_periods)
    if se < EPSILON:
        return 0.0, 1.0, "", clamped

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p), clamped
