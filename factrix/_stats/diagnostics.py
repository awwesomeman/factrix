"""Residual diagnostics — Ljung-Box portmanteau autocorrelation test."""

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
    hyperparameter (#188).
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
