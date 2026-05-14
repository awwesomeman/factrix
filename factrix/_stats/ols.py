"""Ordinary least squares (OLS) estimators with Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) covariance.

Univariate slope-only (``_ols_nw_slope_t``) and full multivariate
(``_ols_nw_multivariate``). Bartlett-kernel math kept in sync with
``factrix._stats.hac._newey_west_se`` so the HAC convention stays in
one place.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factrix._types import EPSILON


def _ols_nw_slope_t(
    y: np.ndarray,
    x: np.ndarray,
    *,
    lags: int,
) -> tuple[float, float, float, np.ndarray]:
    """OLS ``y = α + β·x + ε`` with Newey-West HAC SE on β.

    Returns (β̂, t-stat, two-sided p-value with df=n-2, residuals).
    Centring is done in-place; the residuals are computed in the
    de-meaned space (same as full-rank OLS up to the constant), and
    the score ``u_t = x̃_t · ε_t`` is fed to the same Bartlett kernel
    used by ``_newey_west_se`` so the HAC math stays in one place.

    Returns ``(0.0, 0.0, 1.0, np.zeros(n))`` for n < 3 or degenerate
    inputs (``Var(x) ≈ 0``).
    """
    n = len(y)
    if n < 3 or len(x) != n:
        return 0.0, 0.0, 1.0, np.zeros(n)

    x_c = x - float(np.mean(x))
    y_c = y - float(np.mean(y))
    sxx = float(np.dot(x_c, x_c))
    if sxx < EPSILON:
        return 0.0, 0.0, 1.0, np.zeros(n)

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
    se_beta = float(np.sqrt(var_beta))
    if se_beta < EPSILON:
        return beta, 0.0, 1.0, resid

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

    Returns ``(zeros(k), zeros((k,k)), zeros(n))`` if ``X'X`` is singular
    (e.g. perfectly collinear columns) or ``n < k + 1``.
    """
    n, k = X.shape
    if len(y) != n or n < k + 1:
        return np.zeros(k), np.zeros((k, k)), np.zeros(n)

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.zeros(k), np.zeros((k, k)), np.zeros(n)

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
