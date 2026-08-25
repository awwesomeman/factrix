"""Residual diagnostics — Ljung-Box portmanteau test and lag-1 autocorrelation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factrix._types import EPSILON


def _ljung_box(
    resid: np.ndarray,
    *,
    lags: int | None = None,
) -> tuple[int, float, float]:
    """Resolved lag count, Q statistic, two-sided p-value for residual autocorrelation.

    ``Q = n(n+2) Σ_{k=1..h} ρ̂_k² / (n - k)`` evaluated against
    ``χ²_h``; the H₀ is "no autocorrelation up to lag h". Default
    ``lags = min(10, n // 10)`` per plan §5.2.

    Returns ``(0, NaN, 1.0)`` for ``n < 4`` or unresolvable lag inputs
    — Q is undefined when no lag can be applied. ``(h, NaN, 1.0)`` for
    zero-variance residuals — the lag was resolved but the statistic
    itself is undefined. NaN on Q lets downstream readers distinguish
    "not computable" from "computed and equal to zero". The resolved
    ``h`` lag count is returned so callers can record it as a
    hyperparameter.
    """
    n = len(resid)
    if n < 4:
        return 0, np.nan, 1.0
    h = lags if lags is not None else min(10, n // 10)
    if h < 1:
        return 0, np.nan, 1.0
    h = min(h, n - 1)

    centred = resid - float(np.mean(resid))
    var = float(np.dot(centred, centred))
    if var < EPSILON:
        return h, np.nan, 1.0

    q = 0.0
    for k in range(1, h + 1):
        cov_k = float(np.dot(centred[k:], centred[:-k]))
        rho_k = cov_k / var
        q += rho_k * rho_k / (n - k)
    q *= n * (n + 2)
    return h, float(q), float(sp_stats.chi2.sf(q, df=h))


def _lag1_autocorr(values: np.ndarray) -> float:
    """Sample lag-1 autocorrelation of a 1-D series; ``0.0`` when undefined.

    The persistence screen behind ``WarningCode.SERIAL_CORRELATION_DETECTED``:
    a single number that separates the regime where the HAC / bootstrap
    mean tests are roughly calibrated (≈ 0) from the one where none of them
    is (≥ 0.3). It is computed on the series the test actually runs on —
    the metric's per-period series (the IC series, the per-period beta series,
    the spread series), *not* the raw factor or return columns, whose
    persistence can differ by an order of magnitude from the series that
    is averaged. Returns ``0.0`` for ``n < 3`` or a zero-variance series so a
    degenerate input never trips the screen on its own.
    """
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return 0.0
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom < EPSILON:
        return 0.0
    return float((xc[1:] @ xc[:-1]) / denom)
