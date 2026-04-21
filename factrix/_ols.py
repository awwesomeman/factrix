"""Shared OLS helpers used by spanning (metrics/) and orthogonalize (preprocess/).

Extracted to top-level to avoid circular dependency between metrics/ and preprocess/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factrix._types import EPSILON


@dataclass
class OLSResult:
    """Result of a single OLS regression."""

    alpha: float
    alpha_t: float
    betas: list[float] = field(default_factory=list)
    r_squared: float = 0.0


def ols_alpha(
    candidate: np.ndarray,
    base_matrix: np.ndarray,
) -> OLSResult:
    """OLS regression: candidate = alpha + beta @ base + epsilon.

    Returns:
        OLSResult with alpha, t_stat, betas, and R².
    """
    n_obs = len(candidate)
    if n_obs < 3:
        return OLSResult(alpha=0.0, alpha_t=0.0)

    ones = np.ones((n_obs, 1))
    X = np.hstack([ones, base_matrix]) if base_matrix.shape[1] > 0 else ones

    try:
        beta, _, _, _ = np.linalg.lstsq(X, candidate, rcond=None)
    except np.linalg.LinAlgError:
        return OLSResult(alpha=0.0, alpha_t=0.0)

    alpha = float(beta[0])
    betas = [float(b) for b in beta[1:]]

    resid = candidate - X @ beta

    ss_res = float(np.dot(resid, resid))
    centered = candidate - np.mean(candidate)
    ss_tot = float(np.dot(centered, centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0

    dof = n_obs - X.shape[1]
    if dof <= 0:
        return OLSResult(alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared)

    sigma2 = ss_res / dof
    if sigma2 < EPSILON:
        return OLSResult(alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared)

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    except np.linalg.LinAlgError:
        return OLSResult(alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared)

    if se_alpha < EPSILON:
        return OLSResult(alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared)

    return OLSResult(
        alpha=alpha,
        alpha_t=alpha / se_alpha,
        betas=betas,
        r_squared=r_squared,
    )
