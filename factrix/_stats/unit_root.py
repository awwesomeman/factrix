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
    (0.23, 0.95),
)


def _adf_pvalue_interp(tau: float) -> float:
    """Linear interpolation of ADF p-value from MacKinnon (1996) crits.

    Behaviour at the tails is driven by the outermost critical points
    in ``_ADF_CRITS_CONSTANT``: τ below the leftmost point clamps to
    0.001 (strongly reject unit root); τ above the rightmost clamps to
    0.95 — this is the rightmost MacKinnon value, **not** a hardcoded
    cap. Extending the right tail would require adding critical points
    beyond τ = 0.23.
    """
    if tau <= _ADF_CRITS_CONSTANT[0][0]:
        return _ADF_CRITS_CONSTANT[0][1]
    if tau >= _ADF_CRITS_CONSTANT[-1][0]:
        return _ADF_CRITS_CONSTANT[-1][1]
    for (t1, p1), (t2, p2) in pairwise(_ADF_CRITS_CONSTANT):
        if t1 <= tau <= t2:
            return p1 + (p2 - p1) * (tau - t1) / (t2 - t1)
    return 0.5


def _adf(y: np.ndarray, lags: int = 0) -> tuple[float, float]:
    """Augmented Dickey-Fuller test with drift (constant, no trend).

    Estimates Δy_t = α + β·y_{t-1} + Σ γ_i·Δy_{t-i} + ε and returns
    (τ, p_approx) where τ = β̂ / SE(β̂) and p_approx comes from linear
    interpolation of MacKinnon (1996) asymptotic critical values for
    the constant-only specification. H0: unit root (β = 0); small τ
    rejects in favour of stationarity.

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
    if n < 10 + lags:
        return 0.0, 1.0

    dy = np.diff(y)
    y_lag1 = y[:-1]
    T = len(dy) - lags
    if T < 5:
        return 0.0, 1.0

    target = dy[lags:]
    X_cols = [np.ones(T), y_lag1[lags:]]
    for i in range(1, lags + 1):
        X_cols.append(dy[lags - i : len(dy) - i])
    X = np.column_stack(X_cols)

    try:
        beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    resid = target - X @ beta
    dof = T - X.shape[1]
    if dof <= 0:
        return 0.0, 1.0
    sigma2 = float(np.dot(resid, resid)) / dof
    if sigma2 < EPSILON:
        return 0.0, 1.0
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    if se < EPSILON:
        return 0.0, 1.0
    tau = float(beta[1] / se)
    return tau, _adf_pvalue_interp(tau)
