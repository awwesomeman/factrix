r"""Fama-MacBeth regression — FM-canonical metric for the
``Individual × Continuous`` cell.

``compute_fm_betas``: per-date cross-sectional ordinary least squares (OLS) → (date, beta) DataFrame.
``fm_beta``: Newey-West t-test on the beta series.
``pooled_beta``: pooled OLS with clustered SE by date.
``beta_sign_consistency``: fraction of periods with correct beta sign.

Notes:
    **Pipeline.** Per-date cross-sectional OLS slope $\lambda$
    (cross-section step) → time series of $\lambda$, then Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) $t$
    on its mean; pooled OLS variant clusters SE by date.

References:
    - [Fama & MacBeth (1973)][fama-macbeth-1973], "Risk, Return, and
      Equilibrium: Empirical Tests."
    - [Newey & West (1987)][newey-west-1987], "A Simple, Positive
      Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
      Covariance Matrix."
    - [Petersen (2009)][petersen-2009], "Estimating Standard Errors in
      Finance Panel Data Sets: Comparing Approaches."
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    SEMethod,
    TestMethod,
)
from factrix._codes import WarningCode
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._stats import (
    _newey_west_t_test,
    _p_value_from_t,
)
from factrix._types import DDOF, EPSILON, ShankenVarSource
from factrix.metrics import metric
from factrix.metrics._helpers import _short_circuit_output
from factrix.metrics._metric_capabilities import per_date_series_rename
from factrix.metrics._primitives import compute_fm_betas

__all__ = [  # noqa: RUF022 (teaching order, see #322 SSOT note)
    "fm_beta",
    "pooled_beta",
    "beta_sign_consistency",
]

_FM_CELL = cell(
    FactorScope.INDIVIDUAL,
    FactorDensity.DENSE,
    structure=DataStructure.PANEL,
)

# Slice-test contract (#153 §5): Fama-MacBeth runs a per-date
# OLS regression on the cross-section, not a bucket sort, so slice
# tests never need to downscale `n_groups`. Sample-size constraints
# (T < HARD short-circuit, T < WARN warning) live in the procedure
# below; the cross-section minimum per regression is enforced inside
# the per-date OLS rather than via this attribute.
min_assets_per_group: int | None = None
per_date_series = per_date_series_rename("beta")

# Two-tier sample-size guard on the FM β series. ``T < HARD`` short-
# circuits — NW HAC SE on a 3-period series is undefined. ``HARD ≤ T <
# WARN`` returns the stat with ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS``
# attached (literature floor: Fama-MacBeth originally used T~30+; below
# that the asymptotic t is borderline). ``T ≥ WARN`` is silent.
MIN_FM_PERIODS_HARD: int = 4
MIN_FM_PERIODS_WARN: int = 30

# ---------------------------------------------------------------------------
# Fama-MacBeth significance (parallel to ic())
# ---------------------------------------------------------------------------


@metric(
    cell=_FM_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    requires={"beta_df": compute_fm_betas},
)
def fm_beta(
    beta_df: pl.DataFrame,
    *,
    newey_west_lags: int | None = None,
    forward_periods: int | None = None,
    is_estimated_factor: bool = False,
    factor_return_var: float | None = None,
) -> MetricResult:
    r"""Newey-West t-test on FM beta series. $H_0: \mathrm{mean}(\beta) = 0$.

    Args:
        beta_df: DataFrame with ``date, beta`` columns (from compute_fm_betas).
        newey_west_lags: Number of Newey-West (NW) lags. Defaults to $\lfloor T^{1/3} \rfloor$.
        forward_periods: Overlap horizon of the regression's forward
            return. When set, the NW bandwidth is floored at
            ``forward_periods - 1`` so the kernel is consistent under
            the MA($h-1$) overlap structure of $h$-period returns.
        is_estimated_factor: Set True when the ``Signal_i`` column used by
            ``compute_fm_betas`` is itself an **estimated** quantity
            (rolling ordinary least squares (OLS) $\beta$ to another factor, PCA score,
            ML-predicted score, residual from a first-stage regression).
            [Shanken (1992)][shanken-1992] shows the naive FM SE ignores sampling error
            in the regressor, inflating $t$-stats. **Do NOT** set this
            on raw characteristics (book-to-market, momentum price
            density, accounting ratios) — those are observed, not
            estimated, and enabling the correction will spuriously
            deflate $t$-stats.

            Implementation: [Shanken (1992)][shanken-1992] single-factor
            special case — the NW SE is scaled by
            $\sqrt{1 + \hat\lambda^2/\sigma^2_f}$ (Shanken's general
            multi-factor multiplicative term $1 + \lambda'\Sigma_f^{-1}\lambda$
            collapses to $1 + \hat\lambda^2/\sigma^2_f$ when there is
            one factor). factrix's simplification *omits* the additive
            $+\sigma^2_f/T$ term of the full Shanken variance and is
            therefore only honest for large $T$.

            Note: ``is_estimated_factor`` corrects the **sampling-error**
            dimension of using an estimated regressor. A separate failure
            structure — the estimated factor itself being weak or unidentified
            — produces its own spuriously-significant FM $t$-stats and is
            not addressed by this scaling; see
            [Kan-Zhang (1999)][kan-zhang-1999] for the useless-factor
            diagnostic literature.

        factor_return_var: $\sigma^2_f$, the time-series variance of the
            factor-mimicking portfolio return. Prefer supplying this when
            you have a spread-portfolio return series (the long-short
            spread actually traded on the density). When ``None`` and
            ``is_estimated_factor=True``, falls back to
            $\mathrm{var}(\beta_t)$ as a rough placeholder —
            $\hat\beta_t$ is *not* the factor-mimicking return but is
            usually the only readily-available series. Because
            $\mathrm{var}(\hat\beta_t)$ already absorbs upstream
            estimation noise, it inflates the denominator of the EIV
            factor and so deflates the SE correction; treat the
            ``betas_timeseries_proxy`` result as a **lower bound on the
            true SE inflation** — i.e. an **upper bound on the reported
            $t$-stat** — not a precise estimate.

    Notes:
        Stage 2 of FM:
        $\overline{\beta} = \mathrm{mean}_t\,\beta_t$;
        $t = \overline{\beta} / \mathrm{SE}_{\mathrm{NW}}(\beta)$
        with kernel lag
        $L = \max(\lfloor T^{1/3} \rfloor,\, h - 1)$.
        With ``is_estimated_factor=True``, the
        [Shanken (1992)][shanken-1992] single-factor correction scales
        SE by $\sqrt{1 + \overline{\beta}^2 / \sigma^2_f}$.

        factrix uses the [Andrews (1991)][andrews-1991] $T^{1/3}$ bandwidth floored
        against the Hansen-Hodrick overlap horizon rather than the
        [Newey-West (1994)][newey-west-1994] data-adaptive plug-in — simpler, deterministic,
        and adequate at typical research $T$. factrix's simplification
        of the Shanken variance omits the additive $+\sigma^2_f / T$ term,
        so the correction is honest only for large $T$.

    References:
        - [Fama & MacBeth (1973)][fama-macbeth-1973]. "Risk, Return, and
          Equilibrium: Empirical Tests." Journal of Political Economy,
          81(3), 607–636. Two-stage λ procedure underlying this test.
        - [Newey & West (1987)][newey-west-1987]. "A Simple, Positive
          Semi-Definite, Heteroskedasticity and Autocorrelation
          Consistent Covariance Matrix." Econometrica, 55(3), 703–708.
          HAC variance estimator.
        - [Andrews (1991)][andrews-1991]. "Heteroskedasticity and
          Autocorrelation Consistent Covariance Matrix Estimation."
          Econometrica, 59(3), 817–858. Optimal Bartlett growth rate.
        - [Hansen & Hodrick (1980)][hansen-hodrick-1980]. "Forward
          Exchange Rates as Optimal Predictors of Future Spot Rates."
          Journal of Political Economy, 88(5), 829–853. Overlap horizon
          flooring the kernel.
        - [Shanken (1992)][shanken-1992]. "On the Estimation of
          Beta-Pricing Models." Review of Financial Studies, 5(1), 1–33.
          Errors-in-variables correction for FM stage-2 t when the
          regressor is itself estimated.
        - [Kan & Zhang (1999)][kan-zhang-1999]. "Two-Pass Tests of Asset
          Pricing Models with Useless Factors." Journal of Finance,
          54(1), 203–235. Useless-factor diagnostic; cited as cautionary
          background on factor validity beyond the EIV inflation that
          ``is_estimated_factor`` addresses.

    Examples:
        Chain from :func:`compute_fm_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.fm_beta import compute_fm_betas, fm_beta
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> beta_df = compute_fm_betas(panel)
        >>> result = fm_beta(beta_df, forward_periods=5)
        >>> result.name == ""
        True
    """
    betas = beta_df["beta"].drop_nulls().to_numpy()
    n = len(betas)

    if n < MIN_FM_PERIODS_HARD:
        return _short_circuit_output(
            "fm_beta",
            "insufficient_fm_periods",
            n_obs=n,
            min_required=MIN_FM_PERIODS_HARD,
        )

    warning_codes: list[str] = []
    if n < MIN_FM_PERIODS_WARN:
        warning_codes.append(WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value)
        warnings.warn(
            f"fm_beta: n_periods={n} below MIN_FM_PERIODS_WARN="
            f"{MIN_FM_PERIODS_WARN}; NW HAC SE on a short β series is "
            f"borderline (Fama-MacBeth convention is T≥30). t-stat is "
            f"returned but read p-values cautiously.",
            UserWarning,
            stacklevel=2,
        )

    from factrix._stats import _resolve_nw_lags

    mean_beta = float(np.mean(betas))
    t, p, _ = _newey_west_t_test(
        betas,
        lags=newey_west_lags,
        forward_periods=forward_periods,
    )
    actual_lags = _resolve_nw_lags(n, newey_west_lags, forward_periods)

    metadata: dict = {
        "p_value": p,
        "stat_type": "t",
        "h0": "mean(β)=0",
        "method": "Fama-MacBeth + Newey-West",
        "n_periods": n,
        "newey_west_lags": actual_lags,
        "forward_periods": forward_periods,
        "is_estimated_factor": is_estimated_factor,
    }
    if warning_codes:
        metadata["warning_codes"] = warning_codes

    if is_estimated_factor:
        sigma2_f = (
            float(factor_return_var)
            if factor_return_var is not None
            else float(np.var(betas, ddof=DDOF))
        )
        # σ²_f ≈ 0 means the factor premium series is flat; Shanken's
        # denominator collapses and the correction is undefined. Skip
        # rather than divide into EPSILON — the uncorrected NW result
        # is the honest answer in a degenerate regime.
        if sigma2_f < EPSILON:
            metadata["shanken_correction"] = "skipped_zero_factor_variance"
        else:
            c = 1.0 + (mean_beta**2) / sigma2_f
            sqrt_c = math.sqrt(c)
            t_shanken = t / sqrt_c
            p_shanken = _p_value_from_t(t_shanken, n)
            source: ShankenVarSource = (
                "user_supplied"
                if factor_return_var is not None
                else "betas_timeseries_proxy"
            )
            metadata.update(
                {
                    "p_value_uncorrected": p,
                    "stat_uncorrected": t,
                    "shanken_c": c,
                    "shanken_factor_return_var": sigma2_f,
                    "shanken_factor_return_var_source": source,
                    "p_value": p_shanken,
                    "method": ("Fama-MacBeth + Newey-West + Shanken (1992) EIV"),
                }
            )
            t = t_shanken

    return MetricResult(
        p=metadata.get("p_value"),
        value=mean_beta,
        stat=t,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Pooled OLS with clustered SE
# ---------------------------------------------------------------------------


def _cluster_meat(
    X: np.ndarray,
    resid: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, int]:
    r"""$\sum_g (X_g' e_g)(X_g' e_g)'$ over the groups encoded by ``clusters``.

    Returns ``(meat, G)`` where ``G`` is the number of distinct clusters.
    """
    unique = np.unique(clusters)
    k = X.shape[1]
    meat = np.zeros((k, k))
    for c in unique:
        mask = clusters == c
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)
    return meat, len(unique)


@metric(
    cell=_FM_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
)
def pooled_beta(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    cluster_col: str = "date",
    two_way_cluster_col: str | None = None,
) -> MetricResult:
    r"""Pooled ordinary least squares (OLS) with clustered SE — robustness check against FM.

    Clustering on date alone catches contemporaneous cross-sectional
    dependence but misses asset-level persistence; on asset alone the
    reverse. [Petersen (2009)][petersen-2009] shows panel data usually has both —
    single-way clusters understate SE by 20-50% in that regime.

    FM and single-way share the same point estimate under a balanced
    panel but typically disagree on SE; when $\hat\beta$ and FM
    $\hat\lambda$ have **opposite signs**, ``profile.diagnose()``
    flags an FM/pooled sign-mismatch — a red flag for misspecification.

    Short-circuits when $N < 10$ (no regression), returns ``stat=None``
    with $p=1.0$ when the effective $G < 3$ (SE undefined with < 3
    clusters).

    Formula:
        Point estimate:

        $$
        [\hat\alpha, \hat\beta] = (X'X)^{-1} X'R
        $$

        where $X = [1, \text{Signal}]$ stacked across all
        $(\text{date}, \text{asset})$ observations.

        Single-way clustered sandwich SE (default, cluster on
        ``cluster_col``):

        $$
        \mathrm{meat}_g = \sum_g (X_g' e_g)(X_g' e_g)', \quad
        V = c \cdot (X'X)^{-1} \cdot \mathrm{meat}_g \cdot (X'X)^{-1},
        $$

        with finite-sample correction
        $c = \tfrac{G}{G-1} \cdot \tfrac{N-1}{N-K}$,
        $\mathrm{SE}(\hat\beta) = \sqrt{V_{1,1}}$,
        $t = \hat\beta / \mathrm{SE}$, $\mathrm{df} = G - 1$.

        Two-way clustered sandwich SE (when ``two_way_cluster_col`` is
        set — [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011] /
        [Petersen (2009)][petersen-2009]):

        $$
        V_{\text{two-way}} = V_A + V_B - V_{A \cap B}
        $$

        where $V_A$, $V_B$, $V_{A \cap B}$ are single-way variances
        clustered on $A$, on $B$, and on the intersection cells
        $(A, B)$. Each component uses its own finite-sample correction.
        $\mathrm{df} = \min(G_A, G_B) - 1$ ([Thompson (2011)][thompson-2011]).

    Notes:
        Pool ``(date, asset)`` rows and run a single OLS ``R = alpha +
        beta * Signal + eps`` with the appropriate cluster-robust
        sandwich covariance described above. Single-way: ``df = G - 1``
        with ``G`` the number of clusters; two-way:
        ``df = min(G_A, G_B) - 1`` per [Thompson (2011)][thompson-2011].

        factrix reports ``stat = None`` (rather than 0) when ``G < 3``
        because the cluster-robust variance is undefined with too few
        clusters; falling back to a homoskedastic SE in that regime
        would silently break the panel-correlation guarantee that
        motivated using clustered SE in the first place.

    References:
        - [Petersen (2009)][petersen-2009]. "Estimating Standard Errors
          in Finance Panel Data Sets: Comparing Approaches." Review of
          Financial Studies, 22(1), 435–480. Comparison of FM, clustered,
          and two-way SE under firm/time correlation.
        - [Cameron, Gelbach & Miller (2011)][cameron-gelbach-miller-2011].
          "Robust Inference With Multiway Clustering." Journal of
          Business & Economic Statistics, 29(2), 238–249. Two-way
          clustering formula `V_AB = V_A + V_B − V_{A∩B}`.
        - [Thompson (2011)][thompson-2011]. "Simple Formulas for Standard
          Errors that Cluster by Both Firm and Time." Journal of
          Financial Economics, 99(1), 1–10. Finite-sample df correction
          `min(G_A, G_B) − 1`.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.fm_beta import pooled_beta
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = pooled_beta(panel)
        >>> result.name == ""
        True
    """
    y = df[return_col].to_numpy().astype(np.float64)
    x = df[factor_col].to_numpy().astype(np.float64)
    n_obs = len(y)

    if n_obs < 10:
        return _short_circuit_output(
            "pooled_beta",
            "insufficient_pooled_observations",
            n_obs=n_obs,
            min_required=10,
        )

    X = np.column_stack([np.ones(n_obs), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return _short_circuit_output(
            "pooled_beta",
            "singular_pooled_design_matrix",
            n_obs=n_obs,
        )

    slope = float(beta[1])
    resid = y - X @ beta
    k = X.shape[1]

    clusters_a = df[cluster_col].to_numpy()
    meat_a, g_a = _cluster_meat(X, resid, clusters_a)

    # Finite-sample factor shared across all meat components (Stata /
    # statsmodels convention); the per-cluster-count factor differs by
    # component in the two-way path.
    c_obs = (n_obs - 1) / (n_obs - k)

    if two_way_cluster_col is None:
        if g_a < 3:
            return MetricResult(
                p=1.0,
                value=slope,
                n_obs=n_obs,
                stat=None,
                metadata={
                    "reason": "insufficient_clusters",
                    "n_clusters": g_a,
                    "min_required": 3,
                    "p_value": 1.0,
                },
            )
        effective_meat = (g_a / (g_a - 1)) * meat_a
        df_t = g_a
        method_desc = f"Pooled OLS + clustered SE ({cluster_col})"
        cluster_metadata: dict = {"n_clusters": g_a}
    else:
        clusters_b = df[two_way_cluster_col].to_numpy()
        meat_b, g_b = _cluster_meat(X, resid, clusters_b)
        # Composite key for intersection cells. Factor each side to
        # integer ids then combine — np.unique(axis=0) chokes on object
        # dtype, so we avoid stacking heterogeneous types.
        _, ids_a = np.unique(clusters_a, return_inverse=True)
        _, ids_b = np.unique(clusters_b, return_inverse=True)
        inter_ids = ids_a.astype(np.int64) * (int(ids_b.max()) + 1) + ids_b
        meat_i, g_i = _cluster_meat(X, resid, inter_ids)
        if min(g_a, g_b) < 3:
            return MetricResult(
                p=1.0,
                value=slope,
                n_obs=n_obs,
                stat=None,
                metadata={
                    "reason": "insufficient_clusters",
                    "n_clusters": min(g_a, g_b),
                    "min_required": 3,
                    "n_clusters_a": g_a,
                    "n_clusters_b": g_b,
                    "p_value": 1.0,
                },
            )
        effective_meat = (
            (g_a / (g_a - 1)) * meat_a
            + (g_b / (g_b - 1)) * meat_b
            - (g_i / max(g_i - 1, 1)) * meat_i
        )
        df_t = min(g_a, g_b)
        method_desc = (
            f"Pooled OLS + two-way clustered SE ({cluster_col}, {two_way_cluster_col})"
        )
        cluster_metadata = {
            "n_clusters_a": g_a,
            "n_clusters_b": g_b,
            "n_clusters_intersection": g_i,
        }

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return MetricResult(
            value=slope,
            stat=0.0,
        )

    V = c_obs * xtx_inv @ effective_meat @ xtx_inv
    v_slope = V[1, 1]
    non_psd_fallback = False
    # Two-way V can be numerically non-PSD in small samples (CGM 2011
    # §2.2). Clipping to 0 would report SE=0 / p=1 — that looks like
    # "accept null" but is actually "variance undefined", the opposite
    # of honest. Cameron-Miller (2015, JHR) recommend falling back to
    # the larger-dimension single-way V, which is always PSD.
    if v_slope < 0.0 and two_way_cluster_col is not None:
        v_slope = float(
            c_obs * (xtx_inv @ ((g_a / (g_a - 1)) * meat_a) @ xtx_inv)[1, 1]
        )
        non_psd_fallback = True
    se_slope = float(np.sqrt(max(v_slope, 0.0)))

    t_stat = 0.0 if se_slope < EPSILON else slope / se_slope

    p = _p_value_from_t(t_stat, df_t)

    metadata = {
        "p_value": p,
        "stat_type": "t",
        "h0": "β=0",
        "method": method_desc,
        **cluster_metadata,
    }
    if non_psd_fallback:
        metadata["variance_non_psd_fallback"] = f"one_way_{cluster_col}"

    return MetricResult(
        p=p,
        value=slope,
        n_obs=n_obs,
        stat=t_stat,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Beta sign consistency (parallel to hit_rate)
# ---------------------------------------------------------------------------


@metric(
    cell=_FM_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    requires={"beta_df": compute_fm_betas},
)
def beta_sign_consistency(
    beta_df: pl.DataFrame,
    *,
    expected_sign: int = 1,
) -> MetricResult:
    r"""Fraction of FM per-date $\beta$s carrying the expected sign — ``value`` $= \mathrm{mean}_t \mathbb{1}\{\mathrm{sign}(\beta_t) = s^\star\}$.

    $\beta_t$ is the per-date ordinary least squares (OLS) $\beta$ from ``compute_fm_betas``.
    Range $[0, 1]$; $1.0$ = $\beta$ always has the expected sign across
    periods. Unlike ``ts_beta_sign_consistency`` (which symmetrizes via
    $\max(p, 1-p)$ where $p$ is the positive-sign fraction), this one is directional —
    you must supply the a-priori expected sign. Typical use: paired with
    a prior on factor direction to check stability.

    Short-circuits to NaN when no non-null $\beta$ observations exist.

    Notes:
        ``value`` $= \mathrm{mean}_t \mathbb{1}\{\mathrm{sign}(\beta_t) = s^\star\}$
        over the FM per-date beta series. Range $[0, 1]$; $1.0$ = beta
        always agrees with the prior. Descriptive (no formal $H_0$);
        pair with ``fm_beta`` for inferential significance.

        factrix splits this directional check from the symmetric
        ``ts_beta_sign_consistency`` so the two answer different
        questions: this one requires the caller to commit to a prior
        sign; the symmetric variant tests cross-asset agreement only.

    Examples:
        Chain from :func:`compute_fm_betas` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.fm_beta import (
        ...     compute_fm_betas,
        ...     beta_sign_consistency,
        ... )
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> beta_df = compute_fm_betas(panel)
        >>> result = beta_sign_consistency(beta_df, expected_sign=1)
        >>> result.name == ""
        True
    """
    betas = beta_df["beta"].drop_nulls().to_numpy()
    n = len(betas)
    if n == 0:
        return _short_circuit_output(
            "beta_sign_consistency",
            "no_beta_observations",
            n_obs=0,
            min_required=1,
        )

    if expected_sign >= 0:
        consistent = float(np.mean(betas > 0))
    else:
        consistent = float(np.mean(betas < 0))

    return MetricResult(
        value=consistent,
        metadata={
            "expected_sign": expected_sign,
            "n_periods": n,
        },
    )
