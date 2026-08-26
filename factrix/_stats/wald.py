"""Wald χ² tests for linear restrictions on coefficient vectors.

Three layers of heteroskedasticity-and-autocorrelation-consistent (HAC) / cluster covariance feed the same closing
``_wald_p_linear`` core (``W = (Rβ̂ - q)' [RVR']^{-1} (Rβ̂ - q)``):

- **Generic** — ``_wald_p_linear(beta, V, R, q)`` accepts any
  pre-computed coefficient vector and covariance matrix; the
  bedrock used by every higher-level helper here.
- **Vector-mean Newey-West (NW) HAC** — ``_wald_nw_cluster_means(Y, R, q, lags)``
  for the stacked per-period metric panel: rows are joint per-period
  observations of K slice means; cross-slice covariance is the
  joint Bartlett-kernel HAC of the K-vector series. Backs the
  ``WaldNWCluster`` Estimator.
- **Two-way cluster on (date, asset)** —
  ``_wald_two_way_cluster(y, X, *, R, q, date_ids, asset_ids)`` for
  the raw asset-date panel. [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011]
  ``V_DC = V_date + V_asset - V_intersection`` shape. Backs the
  ``WaldTwoWayCluster`` Estimator — interface preserved;
  no public function consumes it until ``factor_decomposition`` lands
  later.

References:
    - Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). "Robust
      Inference With Multiway Clustering." Journal of Business &
      Economic Statistics, 29(2), 238–249.
    - Newey, W. K. & West, K. D. (1987). "A Simple, Positive
      Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
      Covariance Matrix." Econometrica, 55(3), 703–708.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from factrix._stats.constants import auto_bartlett
from factrix._types import EPSILON

# Shared "this sample admits no Wald statistic" return: a singular middle
# matrix ``R V R'``, a singular ``X'X``, too few clusters for the
# finite-sample correction, or a CGM two-way covariance that came out
# non-finite / negative-definite. NaN rather than the former ``(0.0, 1.0)``:
# a restriction whose covariance collapsed is degenerate, never the null, so
# ``p = 1`` read "no effect" off a sample that carried no test. Matches
# ``factrix._stats.core._calc_t_stat`` and ``hac._NOT_COMPUTABLE``; callers
# hold the metric-level context and surface it as
# ``WarningCode.DEGENERATE_VARIANCE``.
_NOT_COMPUTABLE: tuple[float, float] = (float("nan"), float("nan"))


def _wald_p_linear(
    beta: np.ndarray,
    V: np.ndarray,
    R: np.ndarray,
    q: np.ndarray | float = 0.0,
    *,
    df_denom: float | None = None,
) -> tuple[float, float]:
    """Wald test of the linear restriction ``Rβ = q``.

    ``R`` is ``(r, k)``; ``q`` is ``(r,)`` or scalar for r=1. Returns the
    Wald statistic ``W`` and its p-value.

    With ``df_denom=None`` (default) the reference distribution is the
    asymptotic ``W ~ χ²_r`` — correct only when ``V`` is an analytic
    covariance with effectively infinite degrees of freedom. When ``V`` is
    estimated from a finite sample the χ² reference over-rejects; pass an
    explicit ``df_denom`` to use the finite-sample ``F = W / r ~ F_{r,
    df_denom}`` reference instead:

    - single-series NW-HAC regression (common_quantile / common_asymmetry):
      ``df_denom = T - k`` (residual dof, ``k`` = regressors);
    - one-way cluster-robust: ``df_denom = G - 1``;
    - two-way cluster-robust: ``df_denom = min(G_a, G_b) - 1``.

    ``W`` itself is returned unchanged in both cases; only the p-value differs.

    Returns ``(nan, nan)`` if the middle matrix is singular or any
    restriction's SE is at ``EPSILON`` (degenerate restriction), or
    ``df_denom`` is given but non-positive — see ``_NOT_COMPUTABLE``.
    """
    R = np.atleast_2d(R)
    r = R.shape[0]
    diff = R @ beta - np.atleast_1d(q)
    middle = R @ V @ R.T
    # A restriction whose own standard error is at floating-point residue
    # (``sqrt(var) <= EPSILON``, the same rule ``_calc_t_stat`` applies to a
    # std) has no test: inverting the residue returns W ~ 1e32 / p = 0, which
    # is the mirror image of the (0, 1) this replaced, not a result.
    if not np.all(np.isfinite(middle)) or np.any(
        np.sqrt(np.clip(np.diag(middle), 0.0, None)) <= EPSILON
    ):
        return _NOT_COMPUTABLE
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError:
        return _NOT_COMPUTABLE
    W = float(diff @ middle_inv @ diff)
    if df_denom is None:
        p = float(sp_stats.chi2.sf(W, df=r))
    elif df_denom < 1:
        return _NOT_COMPUTABLE
    else:
        p = float(sp_stats.f.sf(W / r, dfn=r, dfd=df_denom))
    return W, p


def _nw_hac_vector_mean(
    Y: np.ndarray,
    *,
    lags: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint sample mean and Newey-West HAC variance of a vector series.

    For ``Y`` of shape ``(T, K)`` (rows = joint per-period observations
    of K parallel series), returns ``(mean_K, V_KxK)`` where ``V`` is
    the Bartlett-kernel HAC variance of the sample mean — handles both
    serial autocovariance within each column AND cross-column
    contemporaneous + lagged covariance via the matrix autocovariances
    ``Γ_j = (1/T) X̃_t' X̃_{t-j}``. Default lag rule
    ``L = auto_bartlett(T)`` matches ``factrix._stats.hac._newey_west_se``.

    Caller must pre-handle missing data (drop / interpolate) — joint
    NW HAC requires aligned rows; partial NaN within a row would
    require pairwise complete-case autocovariances and a different
    PSD guarantee.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (T, K); got shape {Y.shape}.")
    n, k = Y.shape
    if n < 2:
        return Y.mean(axis=0) if n else np.zeros(k), np.zeros((k, k))

    L = auto_bartlett(n) if lags is None else lags
    L = max(0, min(L, n - 1))

    mean = Y.mean(axis=0)
    centred = Y - mean

    gamma_0 = (centred.T @ centred) / n
    weighted = gamma_0.copy()
    for j in range(1, L + 1):
        gamma_j = (centred[j:].T @ centred[:-j]) / n
        weight = 1.0 - j / (L + 1)
        weighted += weight * (gamma_j + gamma_j.T)

    # Symmetrise — floating-point asymmetry from the running sum can
    # break the strict PSD requirement of `_wald_p_linear`'s inverse.
    weighted = 0.5 * (weighted + weighted.T)
    var_of_mean = weighted / n
    return mean, var_of_mean


def _wald_nw_cluster_means(
    per_date_metric: np.ndarray,
    *,
    R: np.ndarray,
    q: np.ndarray | float = 0.0,
    lags: int | None = None,
) -> tuple[float, float]:
    """Wald χ² for ``R · mean = q`` on a (T, K) per-period metric panel.

    Tests linear restrictions on the K-vector of per-slice means under
    Newey-West HAC + 1-way cluster on the slice grouping. Equivalent
    to running OLS on the long-format stacked panel
    ``y_{t,k} = α + Σ β_j · 1{slice = j}`` with cluster-NW errors,
    but operating directly on the per-period metric matrix is faster and
    more numerically stable than rebuilding the design matrix.

    Args:
        per_date_metric: ``(T, K)`` matrix; ``per_date_metric[t, k]``
            is the metric value for slice ``k`` on date ``t``.
        R: Restriction matrix ``(r, K)``. For pairwise contrast
            "slice 0 vs slice 1": ``[[1, -1, 0, …, 0]]``. For omnibus
            "all K slices equal" (K-1 contrasts vs the first):
            ``[[1, -1, 0, …], [1, 0, -1, 0, …], …]``.
        q: Restriction RHS, scalar or ``(r,)``; default 0.
        lags: Bartlett-kernel bandwidth; ``None`` → ``auto_bartlett(T)``.

    Returns:
        ``(W, p)`` with the Wald statistic ``W`` and a finite-sample
        p-value. The covariance is a one-way (date) cluster estimate with
        ``G = T`` clusters, so the reference is ``F = W / r ~ F_{r, T-1}``
        rather than the over-rejecting asymptotic χ². Returns ``(nan,
        nan)`` on degenerate input (T < 2 or singular middle matrix).
    """
    Y = np.asarray(per_date_metric, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"per_date_metric must be 2-D (T, K); got shape {Y.shape}.")
    n_clusters = Y.shape[0]
    if n_clusters < 2:
        return _NOT_COMPUTABLE
    mean, V = _nw_hac_vector_mean(Y, lags=lags)
    return _wald_p_linear(mean, V, R, q, df_denom=n_clusters - 1)


def _cluster_meat(
    X: np.ndarray,
    u: np.ndarray,
    cluster_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    """One-way cluster meat matrix ``Σ_g (X_g' u_g)(X_g' u_g)'`` and ``G``.

    Sandwich pieces are cached at the cluster level; complexity is
    O(n · k²) regardless of cluster count. Used as the building block
    for both single-cluster and [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011]
    two-way cluster covariance. Returns the uncorrected meat plus the
    cluster count ``G``, which the caller needs for the ``G/(G-1)``
    finite-sample factor.
    """
    _, k = X.shape
    score = X * u[:, None]
    unique = np.unique(cluster_ids)
    meat = np.zeros((k, k))
    for g in unique:
        mask = cluster_ids == g
        s_g = score[mask].sum(axis=0)
        meat += np.outer(s_g, s_g)
    return meat, len(unique)


def _encode_pair_ids(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer id per distinct ``(a, b)`` pair, dtype-agnostic.

    ``np.unique(np.column_stack([a, b]), axis=0)`` raises for
    ``object``/``datetime64`` label arrays (and silently up-casts mixed
    dtypes), so each side is factorised to integer codes first and the
    pair is then encoded as ``code_a * (max_b + 1) + code_b``. Same
    pattern as ``factrix.metrics.fm_beta.pooled_beta``.
    """
    _, ids_a = np.unique(np.asarray(a), return_inverse=True)
    _, ids_b = np.unique(np.asarray(b), return_inverse=True)
    ids_a = np.ravel(ids_a).astype(np.int64)
    ids_b = np.ravel(ids_b).astype(np.int64)
    return ids_a * (int(ids_b.max()) + 1) + ids_b


def _wald_two_way_cluster(
    y: np.ndarray,
    X: np.ndarray,
    *,
    R: np.ndarray,
    q: np.ndarray | float = 0.0,
    date_ids: np.ndarray,
    asset_ids: np.ndarray,
) -> tuple[float, float]:
    """Wald χ² with two-way cluster covariance on (date, asset).

    [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011] construction:
    ``V_DC = V_date + V_asset - V_intersection``, with each piece a
    sandwich ``(X'X)^{-1} M_cluster (X'X)^{-1}``. ``V_intersection``
    clusters by the (date, asset) intersection — for a panel where
    each row is a unique (date, asset) cell this collapses to the
    HC0 heteroskedasticity-robust variance ``M = Σ x_i x_i' u_i²``.

    The returned ``V_DC`` is symmetrised before passing to
    ``_wald_p_linear`` (CGM construction is theoretically symmetric
    but the subtraction can introduce floating-point asymmetry that
    breaks the inverse).

    Notes:
        Finite-sample correction: each meat component carries its own
        ``G/(G-1)`` factor and the whole sandwich carries ``(N-1)/(N-K)`` —
        the CGM / Stata ``vce(cluster …)`` convention, and byte-for-byte the
        same factors :func:`factrix.metrics.fm_beta.pooled_beta` applies on
        its two-way path, so the two estimators now agree on identical input.
        This routine previously omitted the correction entirely and returned
        an anti-conservative (too small) ``V_DC`` — the asymptotic
        no-correction variant is defensible only as ``G → ∞``, which is
        never the regime a factor panel with a few hundred dates is in.

        Cluster labels of any dtype are accepted: both margins are factorised
        with ``np.unique(..., return_inverse=True)`` before the intersection
        id is formed. The natural-looking
        ``np.unique(np.column_stack([date_ids, asset_ids]), axis=0)`` raises
        outright for ``datetime64`` dates paired with string asset ids —
        i.e. for every realistic panel.

        With fewer than 2 clusters on either margin the correction and the
        F reference are both undefined; ``(nan, nan)`` is returned.

    Args:
        y: ``(n,)`` outcome vector.
        X: ``(n, k)`` regressor matrix; caller adds intercept column
            if needed (this routine does not auto-add).
        R: Restriction matrix ``(r, k)``.
        q: Restriction RHS; default 0.
        date_ids: ``(n,)`` date label per observation.
        asset_ids: ``(n,)`` asset label per observation.

    Returns:
        ``(W, p)`` with the Wald statistic ``W`` and a finite-sample
        p-value. With two-way clustering the effective cluster count is
        ``min(G_date, G_asset)``, so the reference is ``F = W / r ~
        F_{r, min(G_date, G_asset) - 1}`` rather than the over-rejecting
        asymptotic χ². Returns ``(nan, nan)`` if ``X'X`` is singular,
        ``n < k + 1``, or the CGM covariance is non-finite /
        negative-definite.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    date_ids = np.asarray(date_ids)
    asset_ids = np.asarray(asset_ids)
    n, k = X.shape
    if len(y) != n or n < k + 1:
        return _NOT_COMPUTABLE
    if len(date_ids) != n or len(asset_ids) != n:
        raise ValueError(
            f"date_ids / asset_ids length must match X rows ({n}); got "
            f"{len(date_ids)} / {len(asset_ids)}."
        )

    # numpy < 2.4 on Apple Accelerate raises spurious FP flags from small
    # dense matmuls on finite input; singular designs are caught via
    # ``LinAlgError`` and the degenerate-SE checks downstream.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return _NOT_COMPUTABLE
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta

    M_date, g_date = _cluster_meat(X, resid, date_ids)
    M_asset, g_asset = _cluster_meat(X, resid, asset_ids)
    # Intersection cluster = HC0 when each row is a unique (date, asset) cell.
    # Build a composite scalar ID by factorising each side to integer codes
    # first: ``np.unique(column_stack([...]), axis=0)`` raises on object /
    # datetime64 label arrays, which is exactly what a real panel carries
    # (datetime dates, string asset ids). Panels with repeated (date, asset)
    # cells (e.g. multiple events per asset-date) still cluster correctly.
    combined_ids = _encode_pair_ids(date_ids, asset_ids)
    M_intersection, g_inter = _cluster_meat(X, resid, combined_ids)

    # CGM / Stata small-sample correction, applied per cluster dimension and
    # then once on the observation axis — identical to the factors
    # ``factrix.metrics.fm_beta.pooled_beta`` uses, so the two estimators now
    # agree on the same panel.
    if min(g_date, g_asset) < 2:
        # G = 1 on either margin: G/(G-1) is undefined and the F reference has
        # zero denominator dof. Nothing is estimable; report NaN rather than
        # a correction-free number that looks like a result.
        return _NOT_COMPUTABLE
    c_obs = (n - 1) / (n - k)
    meat = (
        (g_date / (g_date - 1)) * M_date
        + (g_asset / (g_asset - 1)) * M_asset
        - (g_inter / max(g_inter - 1, 1)) * M_intersection
    )
    V_dc = c_obs * XtX_inv @ meat @ XtX_inv
    V_dc = 0.5 * (V_dc + V_dc.T)
    # CGM caveat: the subtraction can leave V_DC non-PSD on small
    # samples. Symmetrising fixes asymmetry but not negative-definite
    # diagonals; `_wald_p_linear` returns NaN on a singular middle.
    if not np.all(np.isfinite(V_dc)):
        return _NOT_COMPUTABLE
    if np.any(np.diag(V_dc) < -EPSILON):
        return _NOT_COMPUTABLE
    # Finite-sample F reference: the effective cluster count under two-way
    # clustering is the smaller margin (Cameron-Miller 2015), so df_denom =
    # min(G_date, G_asset) - 1.
    return _wald_p_linear(beta, V_dc, R, q, df_denom=min(g_date, g_asset) - 1)
