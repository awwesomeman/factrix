"""Generalized method of moments (GMM, [Hansen (1982)][hansen-1982]) two-step efficient GMM.

Provides the J-statistic for over-identifying-restriction tests on a
moment-condition system. Hand-rolled to keep the dep surface lean,
matching ``hac.py`` (Newey-West / Hansen-Hodrick) and ``ols.py`` —
statsmodels is not a factrix dependency.

The long-run covariance ``S`` of the moment vector uses the Bartlett
kernel sharing ``_resolve_nw_lags`` with ``hac._newey_west_se`` so the
HAC-bandwidth convention (with the ``forward_periods - 1`` overlap
floor) is uniform across single- and multi-moment inference.

References:
    - Hansen, L. P. (1982). "Large sample properties of generalized
      method of moments estimators." Econometrica, 50(4), 1029–1054.
"""

from __future__ import annotations

import numpy as np

from factrix._stats.hac import _resolve_nw_lags
from factrix._types import EPSILON


def _long_run_covariance(
    moments: np.ndarray,
    *,
    forward_periods: int | None = None,
    lags: int | None = None,
) -> np.ndarray:
    """Bartlett-kernel long-run covariance of a multivariate moment series.

    Multivariate generalization of ``hac._newey_west_se``:

        S = Γ_0 + Σ_{j=1..L} w_j (Γ_j + Γ_j'),    w_j = 1 - j/(L+1)

    where Γ_j = (1/T) Σ_t (g_t - ḡ)(g_{t-j} - ḡ)' is the lag-j
    auto-covariance of the moment vector. Symmetrized via ``Γ_j + Γ_j'``
    so ``S`` is symmetric by construction; positive-semidefiniteness is
    inherited from the Bartlett kernel ([Newey-West (1987)][newey-west-1987]).

    Bandwidth defaults to the [Andrews (1991)][andrews-1991]
    ``floor(T^(1/3))`` Bartlett growth rate (Andrews 1991 Proposition 1
    — MSE-optimal rate for the Bartlett kernel); the
    [Newey-West (1994)][newey-west-1994] data-driven plug-in is the
    textbook GMM choice on high-dimensional or strongly persistent
    moment systems and may be added as an opt-in later.

    Args:
        moments: ``(T, K)`` array of per-period moment vectors.
        forward_periods: Overlap horizon. When set, bandwidth is floored
            at ``forward_periods - 1`` for MA(h-1) consistency.
        lags: Explicit bandwidth override. ``None`` uses ``floor(T^(1/3))``.

    Returns:
        ``(K, K)`` symmetric long-run covariance estimate.
    """
    n = moments.shape[0]
    effective_lags = _resolve_nw_lags(n, lags, forward_periods)

    demeaned = moments - moments.mean(axis=0, keepdims=True)
    s = (demeaned.T @ demeaned) / n  # Γ_0
    for j in range(1, effective_lags + 1):
        gamma_j = (demeaned[j:].T @ demeaned[:-j]) / n
        weight = 1.0 - j / (effective_lags + 1)
        s += weight * (gamma_j + gamma_j.T)
    # Enforce exact symmetry — Bartlett sum is symmetric in theory but
    # rounding accumulates on near-singular S; matches wald.py convention.
    return 0.5 * (s + s.T)


def _two_step_gmm_j_stat(
    moments: np.ndarray,
    *,
    n_params: int = 0,
    forward_periods: int | None = None,
    lags: int | None = None,
    max_iter: int = 2,
) -> tuple[float, int, int, bool]:
    """Two-step efficient GMM J-statistic ([Hansen (1982)][hansen-1982]) under H₀: E[g] = 0.

    For pure over-identification (``n_params = 0``) the parameter vector
    is empty and the test reduces to a Wald-style quadratic form on the
    sample mean of the moment vector:

        J = T · ḡ' Ŝ⁻¹ ḡ,    df = K (number of moments)

    where ``Ŝ`` is the Bartlett-kernel long-run covariance. ``J`` is
    asymptotically χ²(df) under H₀.

    For ``n_params = 0`` the first-step estimate is fixed (no parameter
    to update), so iterating beyond step 1 rebuilds ``Ŝ`` on the same
    residuals; ``max_iter`` is capped at 2 and exposed as a forward hook
    for parametric (``n_params > 0``) GMM.

    Args:
        moments: ``(T, K)`` per-period moment matrix.
        n_params: Free parameter count. ``df = K - n_params``. Only
            ``n_params = 0`` (pure overid) is supported; non-zero raises
            ``NotImplementedError``.
        forward_periods: Overlap horizon for the long-run covariance.
        lags: Optional explicit Bartlett-kernel bandwidth.
        max_iter: Two-step (default) vs iterated GMM. Capped silently;
            actual iteration count returned in tuple position 2.

    Returns:
        ``(j_stat, df, n_iter, weight_singular)`` — ``weight_singular``
        is ``True`` when ``Ŝ`` was numerically singular and a
        Moore-Penrose pseudo-inverse was used; callers should surface
        this as an unreliable-SE warning.
    """
    if n_params != 0:
        raise NotImplementedError(
            "Parametric GMM (n_params > 0) is not yet supported; "
            "use n_params = 0 (pure over-identification)."
        )

    n, k = moments.shape
    df = k - n_params
    g_bar = moments.mean(axis=0)

    s = _long_run_covariance(moments, forward_periods=forward_periods, lags=lags)
    # Pseudo-inverse fallback: rank-deficient Ŝ on short / collinear-
    # moment samples would explode np.linalg.solve. pinv is consistent
    # under deficiency and the singular flag drives the caller warning.
    try:
        w = np.linalg.inv(s)
        weight_singular = False
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(s)
        weight_singular = True

    j_stat = float(n * g_bar @ w @ g_bar)
    n_iter = min(max_iter, 2)

    if abs(j_stat) < EPSILON:
        j_stat = 0.0

    return j_stat, df, n_iter, weight_singular
