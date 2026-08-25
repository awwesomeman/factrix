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

# Shared "this sample admits no t-statistic" return for the HAC t-tests:
# a sample too short for the kernel, or one whose HAC SE collapses to zero.
# NaN rather than the former ``(0.0, 1.0)``: a zero-SE sample is degenerate
# in the maximum-evidence direction (or an undefined 0/0), never the null,
# so reporting ``p = 1`` inverted the meaning. Matches
# ``factrix._stats.core._calc_t_stat`` — see its docstring for the
# scipy / R reference behaviour and why ±inf is not used instead.
# Callers hold the metric- or inference-level context needed to label the
# cause and surface it as ``WarningCode.DEGENERATE_VARIANCE``. The two
# too-short-sample branches also return this sentinel, but that is a data
# shortage rather than degeneracy, and callers flag it as such.
_NOT_COMPUTABLE: tuple[float, float, str] = (float("nan"), float("nan"), "")


def _require_finite(values: np.ndarray, func_name: str) -> np.ndarray:
    """Coerce to a float array and reject non-finite entries.

    A NaN propagates through ``np.mean`` / ``np.dot`` and then slips past
    the ``se < EPSILON`` degeneracy guard (``max(nan, 0.0)`` is ``nan`` and
    every comparison with NaN is False), surfacing as a NaN t / p far from
    the cause. Callers must drop or impute upstream; the public
    ``stationary_bootstrap_resamples`` enforces the same contract.
    """
    values = np.asarray(values, dtype=float)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"{func_name}: values must be finite (no NaN / inf).")
    return values


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


# Andrews-Monahan clip on the AR(1) prewhitening coefficient: at |phi| -> 1 the
# recolouring factor 1 / (1 - phi)^2 explodes, so the fit is bounded away from
# the unit root exactly as Andrews & Monahan (1992) recommend.
_PREWHITEN_PHI_CLIP = 0.97


def _ar1_phi(demeaned: np.ndarray) -> float:
    """Clipped least-squares AR(1) coefficient of an already-demeaned series."""
    denom = float(np.dot(demeaned[:-1], demeaned[:-1]))
    if denom < EPSILON:
        return 0.0
    phi = float(np.dot(demeaned[1:], demeaned[:-1]) / denom)
    return float(np.clip(phi, -_PREWHITEN_PHI_CLIP, _PREWHITEN_PHI_CLIP))


def _bartlett_long_run_variance(demeaned: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance ``γ_0 + 2 Σ w_j γ_j`` of a demeaned series."""
    n = len(demeaned)
    weighted_sum = float(np.dot(demeaned, demeaned)) / n
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        weighted_sum += 2.0 * (1.0 - j / (lags + 1)) * gamma_j
    return weighted_sum


def _newey_west_se(
    values: np.ndarray,
    lags: int | None = None,
    forward_periods: int | None = None,
    *,
    prewhiten: bool = True,
) -> float:
    """Newey-West HAC standard error for the mean of a time series.

    Bartlett kernel weights ``w_j = 1 - j/(L+1)`` on the AR(1)-**prewhitened**
    series ([Andrews-Monahan (1992)][andrews-monahan-1992]): fit
    ``x_t = φ x_{t-1} + e_t`` on the demeaned series, run the Bartlett sum on
    ``e``, and recolour the long-run variance by ``1 / (1 - φ̂)²``.

    Why prewhiten. The plain Bartlett estimate at the automatic bandwidth
    understates the long-run variance of a persistent series badly in the
    sample sizes factor research works with — measured on an AR(0.6) series
    it recovers 50% of the truth at ``n = 50`` and 61% at ``n = 150``, and the
    resulting mean test rejects 11–21% at a nominal 5%. Prewhitening removes
    the dominant AR(1) component before the kernel sees it: on the same
    series 93–97% of the truth is recovered and the test sits at 5–8%. On
    iid input the two agree (0.92 vs 0.93 at ``n = 50``), and on real
    overlapping forward-return IC series they are indistinguishable
    (5–9% either way), so the change is confined to the regime it targets.
    On a realistic persistent IC series — where the AR(1) fit is only an
    approximation — it halves the excess rather than removing it (measured
    33% → 16% at φ ≈ 0.85), which is why ``SERIAL_CORRELATION_DETECTED``
    still fires there.

    The multivariate ``_nw_hac_vector_mean`` and the regression kernels
    ``_ols_nw_slope_t`` / ``_ols_nw_multivariate`` are *not* prewhitened:
    a vector series needs a VAR(1) fit and regression scores a different
    derivation, and neither has been measured. ``statistical-methods``
    section 6 records the asymmetry.

    Args:
        values: 1-D array of time series observations.
        lags: Number of lags. Defaults to ``auto_bartlett(T)``.
        forward_periods: Overlap horizon of the input series. When set,
            enforces ``lags >= forward_periods - 1`` — the minimum
            consistent bandwidth for overlapping h-period returns
            ([Hansen-Hodrick (1980)][hansen-hodrick-1980] MA(h-1) structure).
        prewhiten: ``False`` gives the plain Bartlett estimate. Exists for
            the characterisation tests that pin the difference; every
            library path uses the default.

    Returns:
        HAC-adjusted standard error of the mean. ``0.0`` for ``n < 2``; a
        series too short to fit the AR(1) (``n < 4``) falls back to plain
        Bartlett.
    """
    values = _require_finite(values, "_newey_west_se")
    n = len(values)
    if n < 2:
        return 0.0

    lags = _resolve_nw_lags(n, lags, forward_periods)
    demeaned = values - float(np.mean(values))

    if prewhiten and n >= 4:
        phi = _ar1_phi(demeaned)
        resid = demeaned[1:] - phi * demeaned[:-1]
        lrv = (
            _bartlett_long_run_variance(resid, min(lags, len(resid) - 1))
            / (1.0 - phi) ** 2
        )
    else:
        lrv = _bartlett_long_run_variance(demeaned, lags)

    variance_of_mean = max(lrv / n, 0.0)
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
        ``(t_stat, p_value, significance_marker)``. A sample too short to
        run the kernel (``n < 3``) or one whose HAC SE collapses to zero
        returns ``(nan, nan, "")`` — see the degeneracy note below.
    """
    from factrix._logging import get_metrics_logger

    values = _require_finite(values, "_newey_west_t_test")
    n = len(values)
    if n < 3:
        return _NOT_COMPUTABLE

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
        return _NOT_COMPUTABLE

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
    values = _require_finite(values, "_hansen_hodrick_se")
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
    ``(nan, nan, "", clamped)`` — see the degeneracy note below.
    """
    values = _require_finite(values, "_hansen_hodrick_t_test")
    n = len(values)
    if n < 3 or forward_periods < 1:
        return (*_NOT_COMPUTABLE, False)

    mean = float(np.mean(values))
    se, clamped = _hansen_hodrick_se(values, forward_periods)
    if se < EPSILON:
        return (*_NOT_COMPUTABLE, clamped)

    t = mean / se
    p = _p_value_from_t(t, n)
    return t, p, _significance_marker(p), clamped
