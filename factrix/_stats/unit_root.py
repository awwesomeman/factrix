"""Augmented Dickey-Fuller (constant-only) unit-root test.

Lean-dependency implementation: pure NumPy, no statsmodels. Sufficient
for flagging "likely persistent" factors before downstream regressions;
not a substitute for a full unit-root toolkit.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from factrix._types import EPSILON

# MacKinnon (1996) asymptotic critical values, constant-only ADF model.
# Used for a linear-interpolation p-value approximation — precise to
# ~±0.02 across the decision-relevant tail. For production-grade
# p-values, call statsmodels.tsa.stattools.adfuller instead.
_ADF_CRITS_CONSTANT: tuple[tuple[float, float], ...] = (
    (-3.96, 0.001),
    (-3.43, 0.01),
    (-2.86, 0.05),
    (-2.57, 0.10),
    (-1.62, 0.50),
    (-0.44, 0.90),
    (-0.07, 0.95),
    (0.23, 0.975),
    (0.60, 0.99),
)


def _adf_pvalue_interp(tau: float) -> float:
    """Linear interpolation of ADF p-value from [MacKinnon (1996)][mackinnon-1996] crits.

    Behaviour at the tails is driven by the outermost critical points
    in ``_ADF_CRITS_CONSTANT``: τ below the leftmost point clamps to
    0.001 (strongly reject unit root); τ above the rightmost clamps to
    0.99 — this is the rightmost tabulated value (Fuller's τ_μ 99%
    point), **not** a hardcoded cap.
    """
    if tau <= _ADF_CRITS_CONSTANT[0][0]:
        return _ADF_CRITS_CONSTANT[0][1]
    if tau >= _ADF_CRITS_CONSTANT[-1][0]:
        return _ADF_CRITS_CONSTANT[-1][1]
    for (t1, p1), (t2, p2) in pairwise(_ADF_CRITS_CONSTANT):
        if t1 <= tau <= t2:
            return p1 + (p2 - p1) * (tau - t1) / (t2 - t1)
    return 0.5


def _schwert_maxlag(n: int) -> int:
    """Schwert (1989) lag ceiling ``floor(12 · (T/100)^{1/4})`` (statsmodels default)."""
    return int(np.floor(12.0 * (n / 100.0) ** 0.25))


def _adf_fit(y: np.ndarray, lags: int, *, n_drop: int) -> tuple[float, float, int]:
    """Fit the ADF regression at ``lags`` using observations after ``n_drop``.

    Returns ``(tau, aic, n_used)``; ``tau`` is ``nan`` on a degenerate fit.
    ``n_drop >= lags`` lets every candidate lag share one estimation sample,
    which the AIC comparison requires.
    """
    dy = np.diff(y)
    y_lag1 = y[:-1]
    T = len(dy) - n_drop
    if T < 5:
        return float("nan"), float("inf"), T
    target = dy[n_drop:]
    X_cols = [np.ones(T), y_lag1[n_drop:]]
    for i in range(1, lags + 1):
        X_cols.append(dy[n_drop - i : len(dy) - i])
    X = np.column_stack(X_cols)
    k = X.shape[1]
    dof = T - k
    if dof <= 0:
        return float("nan"), float("inf"), T
    try:
        beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("nan"), float("inf"), T
    resid = target - X @ beta
    ssr = float(np.dot(resid, resid))
    sigma2 = ssr / dof
    if sigma2 < EPSILON:
        return float("nan"), float("inf"), T
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    if se < EPSILON:
        return float("nan"), float("inf"), T
    aic = T * np.log(ssr / T) + 2.0 * k
    return float(beta[1] / se), float(aic), T


def _adf(y: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Augmented Dickey-Fuller test with drift (constant, no trend).

    Estimates Δy_t = α + β·y_{t-1} + Σ γ_i·Δy_{t-i} + ε and returns
    (τ, p_approx) where τ = β̂ / SE(β̂) and p_approx comes from linear
    interpolation of [MacKinnon (1996)][mackinnon-1996] asymptotic critical values for
    the constant-only specification. H0: unit root (β = 0); small τ
    rejects in favour of stationarity.

    Args:
        y: Series to test.
        lags: Number of lagged differences. ``None`` (default) selects the
            lag by AIC over ``0..floor(12·(T/100)^{1/4})`` (Schwert 1989
            ceiling) on a common estimation sample, then refits the chosen
            lag on the full sample — the same procedure as
            ``statsmodels.tsa.stattools.adfuller(autolag="AIC")``. Pass an
            explicit ``int`` to fix the lag (``0`` is the plain
            Dickey-Fuller regression).

    Why auto-lag by default: the series this is applied to (per-date IC,
    spread) carry MA(h-1) autocorrelation from overlapping forward returns;
    the un-augmented ``lags=0`` regression is then mis-sized. AIC selection
    is the mainstream default (statsmodels, R ``urca``); the lean
    NumPy implementation keeps factrix free of a statsmodels dependency.

    Lean-dependency implementation: no statsmodels. Sufficient for
    flagging "likely persistent" factors before downstream regressions;
    not a substitute for a full unit-root toolkit.
    """
    y = np.asarray(y, dtype=np.float64)
    # Defence-in-depth for callers that didn't pre-filter: NaN / Inf
    # inputs feed straight into np.linalg.lstsq and trip LAPACK's
    # DLASCL "parameter had an illegal value" emission at process exit.
    # Return the same "can't reject unit root" shape the short sample
    # guard returns; this is the honest answer on a degenerate input.
    if not np.isfinite(y).all():
        return 0.0, 1.0
    n = len(y)
    if lags is None:
        # Keep the candidate ceiling small enough that the shared sample
        # retains a usable number of observations.
        maxlag = min(_schwert_maxlag(n), max(0, (n - 1) // 2 - 3))
        if n < 10 + maxlag or maxlag < 0:
            return 0.0, 1.0
        best_lag, best_aic = 0, float("inf")
        for cand in range(0, maxlag + 1):
            _, aic, _ = _adf_fit(y, cand, n_drop=maxlag)
            if aic < best_aic:
                best_lag, best_aic = cand, aic
        lags = best_lag
    if n < 10 + lags:
        return 0.0, 1.0

    tau, _, _ = _adf_fit(y, lags, n_drop=lags)
    if not np.isfinite(tau):
        return 0.0, 1.0
    return tau, _adf_pvalue_interp(tau)
