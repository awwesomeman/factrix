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

from factrix._stats.hac import _require_finite
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


class AmihudHurvichFit(NamedTuple):
    """Reduced-bias predictive-regression fit. See :func:`_amihud_hurvich_beta`."""

    beta: float
    t_stat: float
    p_value: float
    se: float
    gamma: float
    phi: float
    phi_corrected: float
    innovation_corr: float


def _amihud_hurvich_beta(
    y: np.ndarray,
    x: np.ndarray,
    *,
    lags: int,
    forward_periods: int = 1,
) -> AmihudHurvichFit:
    r"""[Amihud-Hurvich (2004)][amihud-hurvich-2004] reduced-bias predictive regression.

    In the predictive system

        $$y_{t+1} = lpha + eta x_t + e_{t+1},\qquad
          x_t = 	heta + \phi x_{t-1} + v_t$$

    OLS on $eta$ is biased whenever the innovations are correlated:
    [Stambaugh (1999)][stambaugh-1999] gives
    $E[\hateta] - eta pprox (\sigma_{ev}/\sigma_v^2)(1+3\phi)/T$. The
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
       $\hat v^c_t = x_t - \hat	heta_c - \hat\phi_c x_{t-1}$, summed over
       the return window when ``forward_periods > 1`` (the Stambaugh
       channel then spans $v_{t+1},\dots,v_{t+h}$).
    3. Regress $y_t$ on $[1,\ x_t,\ \hat v^c]$. The coefficient on $x_t$ is
       the reduced-bias $\hateta_c$.

    **Generated-regressor standard error.** Writing
    $\hat v^c = v + (\phi - \hat\phi_c)J$ with $J$ the horizon-summed
    predictor, the augmented fit returns
    $\hateta_c = eta + \gamma(\hat\phi_c - \phi)\,c$ where $c$ is the
    loading of $J$ on $x$ inside the augmented design ($c = 1$ at
    ``forward_periods = 1``, where $J = x$). So

        $$\widehat{\mathrm{Var}}(\hateta_c)
          = \widehat{\mathrm{Var}}_{	ext{aug}}(\hateta_c)
            + \hat\gamma^2\,\widehat{\mathrm{Var}}(\hat\phi_c)\,c^2 .$$

    Without this term the SE is badly *understated*: the proxy absorbs the
    correlated part of $e$, so the augmented residual variance is
    $\sigma_e^2(1-
    ho^2)$, and the raw augmented SE alone rejects ~50% of
    true nulls at $
    ho = -0.9$.

    Measured on 400 draws per cell with true $eta = 0$, against plain
    Newey-West OLS:

    | T   | φ    | ρ    | h | bias (OLS → AH)  | size (OLS → AH) |
    |-----|------|------|---|------------------|-----------------|
    | 60  | 0.95 | −0.9 | 1 | +0.0758 → +0.0218| 0.203 → 0.113   |
    | 120 | 0.95 | −0.9 | 1 | +0.0358 → +0.0080| 0.130 → 0.085   |
    | 240 | 0.95 | −0.9 | 1 | +0.0173 → +0.0028| 0.100 → 0.072   |
    | 120 | 0.50 | −0.9 | 1 | +0.0167 → −0.0016| 0.062 → 0.052   |
    | 120 | 0.95 |  0.0 | 1 | −0.0024 → −0.0017| 0.065 → 0.068   |
    | 120 | 0.00 | −0.9 | 1 | +0.0061 → −0.0014| 0.052 → 0.037   |
    | 120 | 0.95 | −0.9 | 5 | +0.1423 → +0.0043| 0.263 → 0.102   |

    The residual excess at ``φ = 0.95`` is the near-unit-root regime, not
    the Stambaugh channel; callers flag it with
    ``WarningCode.PERSISTENT_REGRESSOR``.

    Args:
        y: ``(n,)`` forward returns, ``y[t]`` spanning ``(t, t+h]``.
        x: ``(n,)`` predictor, aligned so ``x[t]`` predicts ``y[t]``.
        lags: Bartlett bandwidth for the augmented regression's HAC
            covariance.
        forward_periods: Overlap horizon ``h`` of ``y``.

    Returns:
        :class:`AmihudHurvichFit`. Every field is NaN when the sample is too
        short (``n < 5``) or the predictor is degenerate.
    """
    y = _require_finite(y, "_amihud_hurvich_beta")
    x = _require_finite(x, "_amihud_hurvich_beta")
    n = len(y)
    nan = float("nan")
    not_computable = AmihudHurvichFit(nan, nan, nan, nan, nan, nan, nan, nan)
    if n < 5 or len(x) != n:
        return not_computable

    x_lag, x_cur = x[:-1], x[1:]
    dev = x_lag - float(np.mean(x_lag))
    sxx = float(np.dot(dev, dev))
    if sxx < EPSILON:
        return not_computable

    phi = float(np.dot(dev, x_cur - float(np.mean(x_cur)))) / sxx
    # AH (2004) eq. 6: second-order bias correction of the AR(1) coefficient.
    phi_c = phi + (1.0 + 3.0 * phi) / n + 3.0 * (1.0 + 3.0 * phi) / n**2
    theta_c = float(np.mean(x_cur)) - phi_c * float(np.mean(x_lag))
    innovation = x_cur - theta_c - phi_c * x_lag

    h = max(int(forward_periods), 1)
    if h > 1:
        # The channel spans v_{t+1}..v_{t+h}; the Jacobian of the proxy with
        # respect to phi is the matching sum of lagged predictor levels.
        padded_v = np.concatenate([innovation, np.zeros(h)])
        padded_x = np.concatenate([x_lag, np.zeros(h)])
        proxy = np.array([padded_v[i : i + h].sum() for i in range(len(innovation))])
        jacobian = np.array([padded_x[i : i + h].sum() for i in range(len(innovation))])
    else:
        proxy = innovation
        jacobian = x_lag.copy()

    m = len(proxy)
    design = np.column_stack([np.ones(m), x[:m], proxy])
    beta_vec, cov, _ = _ols_nw_multivariate(y[:m], design, lags=lags)
    if not np.all(np.isfinite(beta_vec)):
        return not_computable
    se_aug = float(np.sqrt(max(float(cov[1, 1]), 0.0)))
    gamma = float(beta_vec[2])

    sigma2_v = float(np.dot(innovation, innovation)) / max(n - 3, 1)
    se_phi = float(np.sqrt(sigma2_v / sxx))
    loading_vec, _, _ = _ols_nw_multivariate(jacobian, design, lags=0)
    loading = float(loading_vec[1]) if np.isfinite(loading_vec[1]) else 1.0

    se = float(np.sqrt(se_aug**2 + (gamma * se_phi * loading) ** 2))
    beta = float(beta_vec[1])
    # rho is the correlation between the PREDICTIVE residual and the AR
    # innovation - the Stambaugh channel itself. It must be measured off the
    # structural residual ``y - alpha - beta x``, NOT the augmented fit's
    # residual: the latter has the proxy projected out of it by construction,
    # so it is orthogonal to ``innovation`` and would report rho = 0 always.
    resid_e = y[:m] - beta_vec[0] - beta_vec[1] * x[:m]
    corr_matrix = np.corrcoef(resid_e, innovation)
    rho = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else nan
    if se < EPSILON:
        return AmihudHurvichFit(beta, nan, nan, se, gamma, phi, phi_c, rho)

    t_stat = beta / se
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=max(m - 3, 1)))
    return AmihudHurvichFit(beta, t_stat, p_value, se, gamma, phi, phi_c, rho)
