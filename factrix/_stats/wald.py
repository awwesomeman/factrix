"""Wald χ² test for linear restrictions on an estimated coefficient vector."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def _wald_p_linear(
    beta: np.ndarray,
    V: np.ndarray,
    R: np.ndarray,
    q: np.ndarray | float = 0.0,
) -> tuple[float, float]:
    """Wald χ² test of the linear restriction ``Rβ = q``.

    ``R`` is ``(r, k)``; ``q`` is ``(r,)`` or scalar for r=1. Returns
    ``(W, p)`` with ``W ~ χ²_r`` under H₀. Returns ``(0.0, 1.0)`` if
    the middle matrix is singular (degenerate restriction).
    """
    R = np.atleast_2d(R)
    diff = R @ beta - np.atleast_1d(q)
    middle = R @ V @ R.T
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    W = float(diff @ middle_inv @ diff)
    p = float(sp_stats.chi2.sf(W, df=R.shape[0]))
    return W, p
