"""Ordinary least squares (OLS) estimators with Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) covariance.

Univariate slope-only (``_ols_nw_slope_t``) and full multivariate
(``_ols_nw_multivariate``). Bartlett-kernel math kept in sync with
``factrix._stats.hac._newey_west_se`` so the HAC convention stays in
one place.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats as sp_stats

from factrix._stats.hac import (
    _hac_bandwidth_ill_conditioned,
    _har_dof,
    _require_finite,
    _resolve_scalar_wald_hac,
)
from factrix._types import EPSILON


def _ols_nw_slope_se(
    y: np.ndarray,
    x: np.ndarray,
    *,
    lags: int,
) -> tuple[float, float, np.ndarray]:
    """OLS ``y = α + β·x + ε`` with the Newey-West HAC SE of β.

    Returns (β̂, SE(β̂), residuals). Centring is done in-place; the residuals
    are computed in the de-meaned space (same as full-rank OLS up to the
    constant), and the score ``u_t = x̃_t · ε_t`` is fed to the same
    Bartlett kernel used by ``_newey_west_se`` so the HAC math stays in
    one place. :func:`_ols_nw_slope_t` turns the pair into a t / p;
    ``common_beta`` consumes the SE directly because it adds a second
    variance component before forming its t.

    Degenerate samples return ``(NaN, NaN, zeros(n))``, never a zero SE
    that a caller must recognise as a signal. Two cases:

    - ``n < 3`` or mismatched lengths — no fit at all, so β̂ is NaN too.
    - ``Var(x) ≈ 0`` — a constant regressor carries no identifying
      variation; β̂ is undefined (0/0), not zero.

    A perfect fit (``se_β ≈ 0``) is left to the t / p consumer
    (:func:`_ols_nw_slope_t`): it is degenerate in the *maximum*-evidence
    direction, so the t and p are withheld as NaN while β̂ is kept.
    NaN rather than ±∞ for the same reason as
    ``factrix._stats.core._calc_t_stat``: an infinity spreads through
    serialization, aggregation and plotting as a legitimate extreme
    value. Metric callers pass the NaN to
    ``factrix.metrics._helpers._degenerate_test_fields``, which withholds
    the test under ``WarningCode.DEGENERATE_VARIANCE``.

    Raises:
        ValueError: ``y`` or ``x`` holds a non-finite value. A NaN flows
            through ``np.mean`` / ``np.dot`` and slips past the
            ``sxx < EPSILON`` degeneracy guard (every comparison with NaN
            is False), surfacing as a NaN β / t far from the cause. Same
            contract as ``_newey_west_se``; callers drop or impute
            upstream.
    """
    y = _require_finite(y, "_ols_nw_slope_se")
    x = _require_finite(x, "_ols_nw_slope_se")
    n = len(y)
    nan = float("nan")
    if n < 3 or len(x) != n:
        return nan, nan, np.zeros(n)

    x_c = x - float(np.mean(x))
    y_c = y - float(np.mean(y))
    sxx = float(np.dot(x_c, x_c))
    if sxx < EPSILON:
        return nan, nan, np.zeros(n)

    beta = float(np.dot(x_c, y_c)) / sxx
    resid = y_c - beta * x_c
    u = x_c * resid

    # Bartlett-kernel long-run variance of Σu_t (sum form, not mean):
    # S = γ_0 + 2 Σ_{k=1..L} (1 - k/(L+1)) γ_k where γ_k = Σ u_t u_{t-k}.
    gamma_0 = float(np.dot(u, u))
    long_run = gamma_0
    L = max(0, min(lags, n - 1))
    for k in range(1, L + 1):
        gamma_k = float(np.dot(u[k:], u[:-k]))
        weight = 1.0 - k / (L + 1)
        long_run += 2.0 * weight * gamma_k
    long_run = max(long_run, 0.0)

    var_beta = long_run / (sxx * sxx)
    return beta, float(np.sqrt(var_beta)), resid


def _ols_nw_slope_t(
    y: np.ndarray,
    x: np.ndarray,
    *,
    lags: int,
) -> tuple[float, float, float, np.ndarray]:
    """OLS ``y = α + β·x + ε`` with Newey-West HAC SE on β.

    Returns (β̂, t-stat, two-sided p-value with df=n-2, residuals) — the
    t / p form of :func:`_ols_nw_slope_se`, which holds the estimator and
    its contract.

    Degenerate samples return NaN for ``t`` and ``p``, never ``(0.0,
    1.0)``: an unformable fit (``n < 3``, ``Var(x) ≈ 0``) also leaves β̂
    NaN, and a perfect fit (``se_β ≈ 0``) keeps β̂ but withholds the test
    — the former ``p = 1.0`` there reported "no relationship" for a sample
    that fits exactly.
    """
    # Own the finiteness contract under this function's name; the estimator
    # re-checks, cheaply, under its own.
    y = _require_finite(y, "_ols_nw_slope_t")
    x = _require_finite(x, "_ols_nw_slope_t")
    beta, se_beta, resid = _ols_nw_slope_se(y, x, lags=lags)
    n = len(y)
    nan = float("nan")
    if not np.isfinite(se_beta) or se_beta < EPSILON:
        return beta, nan, nan, resid

    t_stat = beta / se_beta
    # df = n - 2 for univariate OLS with intercept (Greene §4.5).
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=max(n - 2, 1)))
    return beta, t_stat, p_value, resid


def _ols_nw_multivariate(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-regressor OLS ``y = Xβ + ε`` with Newey-West HAC covariance.

    Returns ``(β̂, V_hac, resid)``. ``X`` carries its own intercept column
    if needed — this routine does not auto-add one. Bartlett kernel
    matches ``_newey_west_se`` / ``_ols_nw_slope_t`` so HAC math stays
    in one place.

    Returns ``(full(k, nan), full((k,k), nan), zeros(n))`` if ``X'X`` is
    singular (e.g. perfectly collinear columns) or ``n < k + 1``. NaN
    rather than the former zero matrices: a zero ``V_hac`` produced a
    ``t = 0/0`` downstream that read as a non-rejection instead of a
    refusal, and a zero β̂ is not the estimate for a rank-deficient
    design — there is no estimate. Callers map the NaN to
    ``WarningCode.DEGENERATE_VARIANCE``.

    Raises:
        ValueError: ``y`` or ``X`` holds a non-finite value. ``np.linalg.inv``
            does not raise on a NaN-bearing matrix — it returns a NaN inverse,
            so the singularity branch never fires and a NaN β / V_hac is
            returned silently. Same contract as ``_newey_west_se``.
    """
    y = _require_finite(y, "_ols_nw_multivariate")
    X = _require_finite(X, "_ols_nw_multivariate")
    return _ols_nw_multivariate_from_finite(y, X, lags=lags)


def _ols_nw_multivariate_from_finite(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit :func:`_ols_nw_multivariate` after the caller validated input."""
    n, k = X.shape
    not_computable = (np.full(k, np.nan), np.full((k, k), np.nan), np.zeros(n))
    if len(y) != n or n < k + 1:
        return not_computable

    # numpy < 2.4 on Apple Accelerate raises spurious FP flags from small
    # dense matmuls on finite input; singular designs are caught via
    # ``LinAlgError`` and the degenerate-SE checks downstream.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return not_computable

    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Score matrix: u_t = x_t * e_t (n × k).
    U = X * resid[:, None]

    # S = Γ_0 + Σ_{j=1..L} w_j (Γ_j + Γ_j')
    # Γ_0 = Σ u_t u_t' (sum form, matches _ols_nw_slope_t convention).
    S = U.T @ U
    L = max(0, min(lags, n - 1))
    for j in range(1, L + 1):
        gamma_j = U[j:].T @ U[:-j]
        weight = 1.0 - j / (L + 1)
        S += weight * (gamma_j + gamma_j.T)

    V_hac = XtX_inv @ S @ XtX_inv
    return beta, V_hac, resid


class _ScalarWaldHacFit(NamedTuple):
    """OLS fit carrying the complete scalar HAR covariance contract."""

    beta: np.ndarray
    covariance: np.ndarray
    resid: np.ndarray
    lags: int
    dof: float
    bandwidth_ill_conditioned: bool


def _ols_scalar_wald_hac(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lags: int | None = None,
    overlap_periods: int | None = None,
) -> _ScalarWaldHacFit:
    """OLS with the complete covariance contract for a rank-one HAC test.

    Resolves the scalar HAR bandwidth, fits the Newey-West covariance, and
    applies the matching finite-sample variance scale in one entry point.
    ``dof`` is the effective reference degrees of freedom for the caller's
    rank-one t or Wald statistic. ``bandwidth_ill_conditioned`` follows the
    shared warning policy, including suppression below the common warning
    floor.
    """
    # Keep the historical error label from the kernel while routing the
    # validated arrays through the shared fit below. ``ols_alpha`` uses that
    # fit directly after its field-specific validation, avoiding a second
    # full-array finiteness scan.
    y = _require_finite(y, "_ols_nw_multivariate")
    X = _require_finite(X, "_ols_nw_multivariate")
    return _ols_scalar_wald_hac_from_finite(
        y,
        X,
        lags=lags,
        overlap_periods=overlap_periods,
    )


def _ols_scalar_wald_hac_from_finite(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lags: int | None = None,
    overlap_periods: int | None = None,
) -> _ScalarWaldHacFit:
    """Fit :func:`_ols_scalar_wald_hac` after caller-side validation."""
    resolved_lags, variance_scale, dof = _resolve_scalar_wald_hac(
        len(y), lags, overlap_periods
    )
    beta, covariance, resid = _ols_nw_multivariate_from_finite(y, X, lags=resolved_lags)
    return _ScalarWaldHacFit(
        beta=beta,
        covariance=covariance * variance_scale,
        resid=resid,
        lags=resolved_lags,
        dof=dof,
        bandwidth_ill_conditioned=_hac_bandwidth_ill_conditioned(len(y), resolved_lags),
    )


def _ols_homoskedastic(
    y: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain OLS fit with the textbook ``s^2 (X'X)^-1`` covariance.

    The homoskedastic counterpart of :func:`_ols_nw_multivariate`, for the
    one caller that wants the covariance the estimator's source paper uses
    rather than a sandwich (:func:`_amihud_hurvich_beta` at
    ``overlap_periods = 1``). Same NaN-on-degenerate contract.
    """
    y = _require_finite(y, "_ols_homoskedastic")
    X = _require_finite(X, "_ols_homoskedastic")
    n, k = X.shape
    not_computable = (np.full(k, np.nan), np.full((k, k), np.nan), np.zeros(n))
    if len(y) != n or n < k + 1:
        return not_computable
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return not_computable
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sigma2 = float(np.dot(resid, resid)) / max(n - k, 1)
    return beta, sigma2 * XtX_inv, resid


class AmihudHurvichFit(NamedTuple):
    """Reduced-bias predictive-regression fit. See :func:`_amihud_hurvich_beta`.

    ``alpha``, ``n_used`` and ``resid`` describe the **reported structural
    model** ``y = alpha + beta·x + e`` on the rows the augmented design kept,
    so a caller can report diagnostics for the fit it actually publishes
    rather than for the OLS regression that preceded it. ``alpha`` is *not*
    the augmented regression's intercept: that one absorbs the innovation
    proxy and is not the intercept implied by ``beta``.

    A not-computable fit fills every float with NaN, ``n_used`` with ``0``
    and ``resid`` with an empty array.
    """

    beta: float
    t_stat: float
    p_value: float
    se: float
    gamma: float
    phi: float
    phi_corrected: float
    innovation_corr: float
    alpha: float
    n_used: int
    resid: np.ndarray


def _amihud_hurvich_beta(
    y: np.ndarray,
    x: np.ndarray,
    *,
    lags: int,
    overlap_periods: int = 1,
) -> AmihudHurvichFit:
    r"""[Amihud-Hurvich (2004)][amihud-hurvich-2004] reduced-bias predictive regression.

    In the predictive system

        $$y_{t+1} = \alpha + \beta x_t + e_{t+1},\qquad
          x_t = \theta + \phi x_{t-1} + v_t$$

    OLS on $\beta$ is biased whenever the innovations are correlated:
    [Stambaugh (1999)][stambaugh-1999] gives
    $E[\hat\beta] - \beta \approx (\sigma_{ev}/\sigma_v^2)(1+3\phi)/T$. The
    bias is driven by the **product** of persistence $\phi$ and innovation
    correlation $
    ho = \mathrm{corr}(e, v)$ — which is why an ADF screen on
    $x$ alone cannot detect it: ADF proxies $\phi$ and carries no
    information about $
    ho$.

    Amihud-Hurvich's augmented regression removes the bias by construction
    rather than estimating and subtracting it:

    1. Fit the AR(1) on the predictor and bias-correct the coefficient,
       $\hat\phi_c = \hat\phi + (1+3\hat\phi)/n + 3(1+3\hat\phi)/n^2$
       (AH eq. 6).
    2. Form the corrected innovation proxy
       $\hat v^c_t = x_t - \hat\theta_c - \hat\phi_c x_{t-1}$, summed over
       the return window when ``overlap_periods > 1`` (the Stambaugh
       channel then spans $v_{t+1},\dots,v_{t+h}$).
    3. Regress $y_t$ on $[1,\ x_t,\ \hat v^c]$. The coefficient on $x_t$ is
       the reduced-bias $\hat\beta_c$.

    **Generated-regressor standard error.** Writing
    $\hat v^c = v + (\phi - \hat\phi_c)J$ with $J$ the horizon-summed
    predictor, the augmented fit returns
    $\hat\beta_c = \beta + \gamma(\hat\phi_c - \phi)\,c$ where $c$ is the
    loading of $J$ on $x$ inside the augmented design ($c = 1$ at
    ``overlap_periods = 1``, where $J = x$). So

        $$\widehat{\mathrm{Var}}(\hat\beta_c)
          = \widehat{\mathrm{Var}}_{\text{aug}}(\hat\beta_c)
            + \hat\gamma^2\,\widehat{\mathrm{Var}}(\hat\phi_c)\,c^2 .$$

    Without this term the SE is badly *understated*: the proxy absorbs the
    correlated part of $e$, so the augmented residual variance is
    $\sigma_e^2(1-\rho^2)$, and the raw augmented SE alone rejects ~50% of
    true nulls at $\rho = -0.9$.
    $\widehat{\mathrm{Var}}(\hat\phi_c)$ is *not* $\widehat{\mathrm{Var}}
    (\hat\phi)$: the correction map is affine in $\hat\phi$ with slope
    $1 + 3/n + 9/n^2$, so the variance carries that factor squared (AH's own
    form; omitting it understates the term by 10% at $n = 60$).

    **Two departures from AH (2004), both factrix choices.**

    1. *The covariance inside the augmented regression is horizon-dependent.*
       At ``overlap_periods = 1`` it is AH's own homoskedastic
       $s^2(X'X)^{-1}$ and the $t$ is read against $t_{m-3}$; the regression
       is not overlapping there, so a Bartlett kernel is pure downward bias
       in the SE. At $h > 1$ it is the Bartlett HAC covariance at ``lags``,
       read against the fixed-$b$ effective df of
       :func:`factrix._stats.hac._har_dof` — this is a single-restriction
       slope test, so the $K \times K$ Wald degradation that keeps the
       multivariate paths on the narrow bandwidth rule does not apply.
    2. *The $h > 1$ extension is self-derived.* AH (2004) is an $h = 1$
       method. Summing $\hat v^c$ over $t+1 \dots t+h$ is the natural
       extension of the Stambaugh channel to a horizon-$h$ return and the
       measured bias supports it, but it is not in the paper. Only windows
       that fit inside the sample are kept, so the design loses its last
       $h - 1$ rows rather than carrying zero-padded truncated proxy sums.

    Measured on 2000 draws per cell with true $\beta = 0$, against plain
    Newey-West OLS. The $\rho = 0$ rows carry no Stambaugh channel at all
    and exist to separate the correction from the inference:

    | T    | φ    | ρ    | h | bias (OLS → AH)  | size (OLS → AH) |
    |------|------|------|---|------------------|-----------------|
    | 60   | 0.50 |  0.0 | 1 | +0.0010 → +0.0012| 0.086 → 0.043   |
    | 60   | 0.90 |  0.0 | 1 | +0.0009 → −0.0007| 0.091 → 0.050   |
    | 60   | 0.95 |  0.0 | 1 | −0.0014 → −0.0022| 0.102 → 0.055   |
    | 60   | 0.99 |  0.0 | 1 | +0.0017 → +0.0015| 0.084 → 0.050   |
    | 60   | 0.95 | −0.9 | 1 | +0.0670 → +0.0114| 0.183 → 0.075   |
    | 120  | 0.95 | −0.9 | 1 | +0.0313 → +0.0026| 0.125 → 0.083   |
    | 240  | 0.95 | −0.9 | 1 | +0.0147 → +0.0007| 0.088 → 0.062   |
    | 60   | 0.50 |  0.0 | 5 | −0.0095 → −0.0137| 0.144 → 0.121   |
    | 60   | 0.90 |  0.0 | 5 | −0.0029 → −0.0115| 0.214 → 0.145   |
    | 120  | 0.50 |  0.0 | 5 | −0.0050 → −0.0059| 0.107 → 0.097   |
    | 240  | 0.95 |  0.0 | 5 | +0.0002 → −0.0000| 0.130 → 0.102   |
    | 60   | 0.95 | −0.9 | 5 | +0.2733 → −0.0031| 0.370 → 0.058   |
    | 120  | 0.95 | −0.9 | 5 | +0.1449 → +0.0058| 0.262 → 0.068   |
    | 1000 | 0.90 |  0.0 | 5 | −0.0005 → +0.0018| 0.103 → 0.075   |

    Read the two blocks separately. At $h = 1$ the test is calibrated
    (4.3-5.5% at $\rho = 0$, 6.2-8.3% in the strongest Stambaugh cells).
    At $h > 1$ it is **not**, and the residual excess is not the Stambaugh
    channel and not the near-unit-root regime: it is present at $\rho = 0$
    for every $\phi$, it tracks $h/T$, and plain OLS-NW carries it too. It
    is the overlapping-regression HAC problem, only partly repaired by the
    HAR bandwidth and fixed-$b$ df this function now uses (measured
    10-19% before, 7.5-14.5% after). The textbook resolution is fixed-$b$
    critical values for the regression Wald, which factrix does not
    implement; until it does, read an $h > 1$ predictive $p$ against a
    raised hurdle regardless of the correction.

    **The correction costs power**, because plain OLS's apparent power at
    $\rho \ne 0$ is partly its own bias: at $T=60,\ \phi=0.95,\ \rho=-0.9$
    the corrected test rejects 28.8% of a true alternative against OLS's
    88.6% (T=120: 44.4% against 90.7%). At $\rho = 0$, where OLS is
    unbiased, the gap is small (63.2% against 70.5% at $T=60$). A metric
    that stops being significant after the correction was not necessarily
    significant before it.

    $\hat\phi_c$ is deliberately not clamped below one. Clamping at
    $1 - 1/n$ was measured and re-opens the bias the estimator exists to
    close ($T=60,\ \phi=0.95,\ \rho=-0.9,\ h=5$: bias +0.044 clamped
    against −0.008 unclamped). Callers read ``phi_corrected >= 1`` off the
    fit and raise ``WarningCode.PERSISTENT_REGRESSOR``.

    Args:
        y: ``(n,)`` forward returns, ``y[t]`` spanning ``(t, t+h]``.
        x: ``(n,)`` predictor, aligned so ``x[t]`` predicts ``y[t]``.
        lags: Bartlett bandwidth for the augmented regression's HAC
            covariance at ``overlap_periods > 1``; ignored at
            ``overlap_periods = 1``, which uses the homoskedastic OLS
            covariance instead.
        overlap_periods: Overlap horizon ``h`` of ``y``.

    Returns:
        :class:`AmihudHurvichFit`. ``n_used`` is the row count the augmented
        design ran on — ``n - h``, the first observation going to the AR(1)
        lag and the last ``h - 1`` windows to the horizon-summed proxy — and
        ``alpha`` / ``resid`` are the intercept and residual of the reported
        structural model ``y = alpha + beta·x + e`` on exactly those rows.
        Every field is NaN (``n_used = 0``, ``resid`` empty) when the sample
        is too short (``n < 5``) or the predictor is degenerate.
    """
    y = _require_finite(y, "_amihud_hurvich_beta")
    x = _require_finite(x, "_amihud_hurvich_beta")
    n = len(y)
    nan = float("nan")
    not_computable = AmihudHurvichFit(
        nan, nan, nan, nan, nan, nan, nan, nan, nan, 0, np.empty(0)
    )
    if n < 5 or len(x) != n:
        return not_computable

    x_lag, x_cur = x[:-1], x[1:]
    dev = x_lag - float(np.mean(x_lag))
    sxx = float(np.dot(dev, dev))
    if sxx < EPSILON:
        return not_computable

    phi = float(np.dot(dev, x_cur - float(np.mean(x_cur)))) / sxx
    # AH (2004) eq. 6: second-order bias correction of the AR(1) coefficient.
    # The correction is additive and unbounded, so at phi near one in a short
    # sample it can push phi_c past the unit circle - AH note this themselves.
    # It is NOT clamped: clamping at 1 - 1/n was measured and it re-opens the
    # bias the estimator exists to close (T=60, phi=0.95, rho=-0.9, h=5: bias
    # +0.044 clamped against -0.008 unclamped), because the draws that need
    # the largest correction are exactly the ones the clamp truncates. The
    # regime is disclosed instead - callers read ``phi_corrected >= 1`` off
    # the fit and raise ``WarningCode.PERSISTENT_REGRESSOR``.
    phi_c = phi + (1.0 + 3.0 * phi) / n + 3.0 * (1.0 + 3.0 * phi) / n**2
    theta_c = float(np.mean(x_cur)) - phi_c * float(np.mean(x_lag))
    innovation = x_cur - theta_c - phi_c * x_lag

    h = max(int(overlap_periods), 1)
    if h > 1:
        # The channel spans v_{t+1}..v_{t+h}; the Jacobian of the proxy with
        # respect to phi is the matching sum of lagged predictor levels.
        # Only windows that fit inside the sample are kept: a zero-padded
        # truncated sum is a different regressor from the h-term one every
        # other row carries, and feeding those last h-1 rows in adds an
        # errors-in-variables nuisance at the sample end for no information.
        n_windows = len(innovation) - (h - 1)
        if n_windows < 5:
            return not_computable
        proxy = np.array([innovation[i : i + h].sum() for i in range(n_windows)])
        jacobian = np.array([x_lag[i : i + h].sum() for i in range(n_windows)])
    else:
        proxy = innovation
        jacobian = x_lag.copy()

    m = len(proxy)
    design = np.column_stack([np.ones(m), x[:m], proxy])
    # At h = 1 the augmented regression is not overlapping and its residual
    # carries no MA structure to absorb, so AH's own plain OLS covariance
    # s^2 (X'X)^-1 is the right one; a Bartlett kernel there is pure downward
    # bias in the SE (measured: 8.1-10.6% rejection at rho = 0, T = 60). Even
    # the zero-lag sandwich is the HC0 form, whose own small-sample
    # under-coverage costs another 1-2pp at T = 60. HAC is kept for h > 1,
    # where the overlap is real.
    if h == 1:
        beta_vec, cov, _ = _ols_homoskedastic(y[:m], design)
    else:
        beta_vec, cov, _ = _ols_nw_multivariate(y[:m], design, lags=lags)
    if not np.all(np.isfinite(beta_vec)):
        return not_computable
    se_aug = float(np.sqrt(max(float(cov[1, 1]), 0.0)))
    gamma = float(beta_vec[2])

    sigma2_v = float(np.dot(innovation, innovation)) / max(n - 3, 1)
    # sqrt(sigma2_v / sxx) is SE(phi_hat); the term the augmented SE needs is
    # SE(phi_c). The correction map is affine in phi_hat with slope
    # 1 + 3/n + 9/n^2, so Var(phi_c) = (1 + 3/n + 9/n^2)^2 Var(phi_hat) - the
    # form AH use. Omitting the factor understates that variance by 10% at
    # n = 60, 5% at n = 120.
    se_phi = float(np.sqrt(sigma2_v / sxx)) * (1.0 + 3.0 / n + 9.0 / n**2)
    loading_vec, _, _ = _ols_nw_multivariate(jacobian, design, lags=0)
    loading = float(loading_vec[1]) if np.isfinite(loading_vec[1]) else 1.0

    se = float(np.sqrt(se_aug**2 + (gamma * se_phi * loading) ** 2))
    beta = float(beta_vec[1])
    # The intercept the REPORTED model implies, on the rows this design used.
    # It is not ``beta_vec[0]``: the augmented intercept is fitted alongside
    # the innovation proxy and absorbs its (non-zero) mean, so it does not
    # close the structural equation ``y = alpha + beta x + e``. The two differ
    # by the constant ``gamma * mean(proxy)`` only.
    alpha = float(np.mean(y[:m]) - beta * np.mean(x[:m]))
    # rho is the correlation between the PREDICTIVE residual and the AR
    # innovation - the Stambaugh channel itself. It must be measured off the
    # structural residual ``y - alpha - beta x``, NOT the augmented fit's
    # residual: the latter has the proxy projected out of it by construction,
    # so it is orthogonal to ``innovation`` and would report rho = 0 always.
    # The same series is handed back on the fit so callers can screen the
    # residuals of the model they report; a constant shift leaves rho alone.
    resid_e = y[:m] - alpha - beta * x[:m]
    corr_matrix = np.corrcoef(resid_e, innovation[:m])
    rho = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else nan
    if se < EPSILON:
        return AmihudHurvichFit(
            beta, nan, nan, se, gamma, phi, phi_c, rho, alpha, m, resid_e
        )

    t_stat = beta / se
    # h = 1: the textbook residual df of the augmented regression. h > 1: the
    # augmented SE is a Bartlett HAC one, so the t is read against the same
    # fixed-b effective df the scalar HAR path uses (``_har_dof``). This is a
    # single-restriction slope test, so the K x K Wald argument that keeps the
    # multivariate paths on the narrow rule does not apply to it.
    dof = float(max(m - 3, 1)) if h == 1 else _har_dof(m, lags, h)
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=dof))
    return AmihudHurvichFit(
        beta, t_stat, p_value, se, gamma, phi, phi_c, rho, alpha, m, resid_e
    )
