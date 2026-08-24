"""Shared ordinary least squares (OLS) helpers used by spanning (metrics/) and orthogonalize (preprocess/).

Extracted to top-level to avoid circular dependency between metrics/ and preprocess/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factrix._stats.constants import auto_bartlett
from factrix._stats.ols import _ols_nw_multivariate
from factrix._types import EPSILON


@dataclass
class _OLSResult:
    """Result of a single OLS regression."""

    alpha: float
    alpha_t: float
    betas: list[float] = field(default_factory=list)
    r_squared: float = 0.0
    df_resid: int = 0


def ols_alpha(
    candidate: np.ndarray,
    base_matrix: np.ndarray,
) -> _OLSResult:
    """OLS regression ``candidate = alpha + beta @ base + epsilon`` with a HAC t on alpha.

    Point estimates are plain OLS. ``alpha_t`` divides by the Newey-West
    HAC standard error (Bartlett kernel, [Newey-West (1994)][newey-west-1994]
    automatic bandwidth) rather than the homoskedastic OLS SE. Spanning
    alphas are routinely reported with HAC t-stats (Barillas-Shanken,
    Fama-French).

    **The trade, measured.** Empirical size at a nominal 5% under a true
    null (``alpha = 0``, one base factor, 4000 draws, read against
    ``t(n - k)``):

    | residuals | OLS | HAC |
    |---|---|---|
    | iid, n=60 | 0.047 | 0.071 |
    | iid, n=120 | 0.052 | 0.065 |
    | iid, n=240 | 0.047 | 0.054 |
    | AR(0.6), n=60 | 0.337 | 0.185 |
    | AR(0.6), n=120 | 0.332 | 0.135 |
    | AR(0.6), n=240 | 0.330 | 0.115 |

    On the case the retired assumption was written for — iid, genuinely
    non-overlapping — HAC costs 1–2 points of size, the usual small-sample
    HAC noise every other HAC path in factrix carries and
    ``statistical-methods`` §6 discloses. On the case the assumption was
    silently covering, the OLS SE rejects at **33%** and does not improve
    with more data, because it is estimating the wrong quantity: the
    autocorrelation is in the residuals, and more of them does not help.
    Trading ~1.5 points for ~20 is not a close call.

    HAC is not itself well calibrated on the autocorrelated case (0.115 at
    n=240 and converging slowly); it is merely far closer. That residual
    gap is a Bartlett small-sample property shared by every HAC path here,
    not something specific to spanning.

    An earlier version used the OLS SE on the stated assumption that
    callers pass non-overlap spreads; nothing in the ``(date, spread)``
    input could verify that, so the assumption was retired rather than
    documented harder.

    Returns:
        _OLSResult with alpha, HAC t_stat, betas, R², and residual degrees
        of freedom ``df_resid = n_obs - (1 + n_base_factors)`` (the
        reference the t is read against — see ``_stats.hac`` on why the
        full-count df is kept).

    Raises:
        ValueError: ``candidate`` or ``base_matrix`` holds a non-finite
            value. ``np.linalg.lstsq`` does not raise on NaN input — it
            returns a NaN solution, so an unguarded NaN became a NaN alpha
            and a NaN t far from the cause. The guard lives in the kernel
            rather than at the call sites: ``spanning`` filtered before all
            three of its calls and its own comment says why, which made the
            protection a property of those callers instead of the function,
            and left any fourth caller exposed. Same contract as
            ``factrix._stats.hac._require_finite``.
    """
    candidate = np.asarray(candidate, dtype=float)
    base_matrix = np.asarray(base_matrix, dtype=float)
    if candidate.size and not np.all(np.isfinite(candidate)):
        raise ValueError("ols_alpha: candidate must be finite (no NaN / inf).")
    if base_matrix.size and not np.all(np.isfinite(base_matrix)):
        raise ValueError("ols_alpha: base_matrix must be finite (no NaN / inf).")

    n_obs = len(candidate)
    if n_obs < 3:
        return _OLSResult(alpha=0.0, alpha_t=0.0)

    ones = np.ones((n_obs, 1))
    X = np.hstack([ones, base_matrix]) if base_matrix.shape[1] > 0 else ones

    try:
        beta, _, _, _ = np.linalg.lstsq(X, candidate, rcond=None)
    except np.linalg.LinAlgError:
        return _OLSResult(alpha=0.0, alpha_t=0.0)

    alpha = float(beta[0])
    betas = [float(b) for b in beta[1:]]

    resid = candidate - X @ beta

    ss_res = float(np.dot(resid, resid))
    centered = candidate - np.mean(candidate)
    ss_tot = float(np.dot(centered, centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0

    dof = n_obs - X.shape[1]
    if dof <= 0:
        return _OLSResult(alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared)

    sigma2 = ss_res / dof
    if sigma2 < EPSILON:
        return _OLSResult(
            alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared, df_resid=dof
        )

    # HAC covariance of the OLS coefficients; ``_ols_nw_multivariate`` returns
    # zeros when X'X is singular, which the EPSILON guard below turns into
    # the same degenerate result the OLS path produced.
    _, v_hac, _ = _ols_nw_multivariate(candidate, X, lags=auto_bartlett(n_obs))
    se_alpha = float(np.sqrt(max(v_hac[0, 0], 0.0)))

    if se_alpha < EPSILON:
        return _OLSResult(
            alpha=alpha, alpha_t=0.0, betas=betas, r_squared=r_squared, df_resid=dof
        )

    return _OLSResult(
        alpha=alpha,
        alpha_t=alpha / se_alpha,
        betas=betas,
        r_squared=r_squared,
        df_resid=dof,
    )
