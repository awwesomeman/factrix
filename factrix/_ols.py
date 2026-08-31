"""Shared ordinary least squares (OLS) helpers used by spanning (metrics/) and orthogonalize (preprocess/).

Extracted to top-level to avoid circular dependency between metrics/ and preprocess/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factrix._stats.hac import _resolve_scalar_wald_hac
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
    #: Degrees of freedom ``alpha_t`` is to be read against. The fixed-b
    #: effective df of the HAC kernel (``_resolve_scalar_wald_hac``), which is
    #: well below ``df_resid`` at a research sample size; ``df_resid`` stays
    #: the plain regression residual count for reporting.
    alpha_dof: float = 0.0


def ols_alpha(
    candidate: np.ndarray,
    base_matrix: np.ndarray,
    *,
    overlap_periods: int = 1,
) -> _OLSResult:
    """OLS regression ``candidate = alpha + beta @ base + epsilon`` with a HAC t on alpha.

    Point estimates are plain OLS. ``alpha_t`` divides by the Newey-West
    HAC standard error (Bartlett kernel, [Newey-West (1994)][newey-west-1994]
    automatic bandwidth, floored at ``3 * (overlap_periods - 1)``) rather than
    the homoskedastic OLS SE. Spanning alphas are routinely reported with HAC
    t-stats (Barillas-Shanken, Fama-French).

    ``overlap_periods`` is the overlap horizon of the spread series being
    regressed. Spreads built from ``h``-period overlapping forward returns
    carry MA(``h-1``) residual autocorrelation
    ([Hansen-Hodrick (1980)][hansen-hodrick-1980]), and a Bartlett kernel
    must run enough lags to absorb it. The alpha test is a **single
    restriction**, so the bandwidth, the ``T / (T - L - 1)`` variance scale
    and the fixed-``b`` effective degrees of freedom all come from
    :func:`factrix._stats.hac._resolve_scalar_wald_hac` — the scalar HAR
    recipe, whose ``3(h - 1)`` overlap floor is the calibrated one for a
    scalar statistic. The default ``1`` (no floor) is the non-overlapping
    case.

    **The trade, measured.** Both columns below predate the current scalar HAR
    recipe and are retained because they motivated retiring the OLS SE. The
    current recipe's measurements follow the table. Empirical size at a
    nominal 5% under a true null (``alpha = 0``, one base factor, 4000 draws,
    read against ``t(n - k)``):

    | residuals | OLS | HAC (narrow rule, superseded) |
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

    The HAC column above was measured while this path still used the narrow
    ``max(auto_bartlett(T), h - 1)`` rule and the ``t_{T-k}`` reference.
    Moving it to the scalar HAR recipe roughly halves the excess on an
    overlapping null: on two independent AR(``phi``) series summed over
    ``h`` periods (300 draws, seed ``20260830 + rep``) the size goes
    9.0 -> 6.0%, 18.7 -> 9.3% and 46.0 -> 11.7% at ``n = 60`` for
    ``h = 1, 5, 21``, and 50.0 -> 29.3%, 50.3 -> 30.3%, 60.0 -> 21.7% on
    the ``phi = 0.9`` version of the same null. The persistent column stays
    the uncalibrated regime ``statistical-methods`` section 6 discloses;
    spanning has no persistence screen of its own, so read it there.

    An earlier version used the OLS SE on the stated assumption that
    callers pass non-overlap spreads; nothing in the ``(date, spread)``
    input could verify that, so the assumption was retired rather than
    documented harder.

    Returns:
        _OLSResult with alpha, HAC t_stat, betas, R², and residual degrees
        of freedom ``df_resid = n_obs - (1 + n_base_factors)`` (the
        reference the t is read against — see ``_stats.hac`` on why the
        full-count df is kept).

    A fit that cannot be formed -- fewer than three observations, or a
    rank-deficient design -- returns ``alpha`` and ``alpha_t`` as NaN.
    Callers withhold the test rather than reporting the zeros an earlier
    version returned, which claimed the factor added exactly nothing.

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
        return _OLSResult(alpha=float("nan"), alpha_t=float("nan"))

    ones = np.ones((n_obs, 1))
    X = np.hstack([ones, base_matrix]) if base_matrix.shape[1] > 0 else ones

    try:
        beta, _, _, _ = np.linalg.lstsq(X, candidate, rcond=None)
    except np.linalg.LinAlgError:
        # Rank-deficient design (collinear base factors, or a base factor
        # equal to the candidate). The fit does not exist, so neither does
        # alpha: NaN rather than 0.0, which would read as "this factor adds
        # exactly nothing" -- a decisive claim from a failed computation.
        return _OLSResult(alpha=float("nan"), alpha_t=float("nan"))

    alpha = float(beta[0])
    betas = [float(b) for b in beta[1:]]

    resid = candidate - X @ beta

    ss_res = float(np.dot(resid, resid))
    centered = candidate - np.mean(candidate)
    ss_tot = float(np.dot(centered, centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0

    dof = n_obs - X.shape[1]
    if dof <= 0:
        # No residual degrees of freedom: the design saturates the sample,
        # so there is nothing left to estimate a standard error from.
        return _OLSResult(
            alpha=alpha, alpha_t=float("nan"), betas=betas, r_squared=r_squared
        )

    sigma2 = ss_res / dof
    if sigma2 < EPSILON:
        # Residuals vanished -- a perfect (typically rank-deficient) fit.
        # ``lstsq`` does not raise there, it returns a minimum-norm
        # solution, so this is the branch collinear designs actually reach.
        return _OLSResult(
            alpha=alpha,
            alpha_t=float("nan"),
            betas=betas,
            r_squared=r_squared,
            df_resid=dof,
        )

    # HAC covariance of the OLS coefficients; ``_ols_nw_multivariate`` returns
    # zeros when X'X is singular, which the EPSILON guard below turns into
    # the same degenerate result the OLS path produced.
    lags, hac_scale, hac_dof = _resolve_scalar_wald_hac(n_obs, None, overlap_periods)
    _, v_hac, _ = _ols_nw_multivariate(candidate, X, lags=lags)
    v_hac = v_hac * hac_scale
    se_alpha = float(np.sqrt(max(v_hac[0, 0], 0.0)))

    if se_alpha < EPSILON:
        # A collapsed HAC SE -- a perfect fit, or a design so nearly
        # rank-deficient that the residuals vanish -- leaves no t. NaN, not
        # 0.0: the alpha itself is real (a duplicated base column still
        # yields an intercept), so reporting t = 0 turns a live estimate
        # into a decisive "not significant". Same rule as ``_calc_t_stat``.
        return _OLSResult(
            alpha=alpha,
            alpha_t=float("nan"),
            betas=betas,
            r_squared=r_squared,
            df_resid=dof,
        )

    return _OLSResult(
        alpha=alpha,
        alpha_t=alpha / se_alpha,
        betas=betas,
        r_squared=r_squared,
        df_resid=dof,
        alpha_dof=hac_dof,
    )
