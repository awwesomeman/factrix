r"""Fama-MacBeth regression — FM-canonical metric for the
``Individual × Continuous`` cell.

Aggregation: per-date cross-sectional OLS slope $\lambda$
(cross-section step) → time series of $\lambda$, then NW HAC $t$ on
its mean; pooled OLS variant clusters SE by date.

``compute_fm_betas``: per-date cross-sectional OLS → (date, beta) DataFrame.
``fama_macbeth``: Newey-West t-test on the beta series.
``pooled_ols``: pooled OLS with clustered SE by date.
``beta_sign_consistency``: fraction of periods with correct beta sign.

References:
    Fama & MacBeth (1973), "Risk, Return, and Equilibrium."
    Newey & West (1987), "HAC Covariance Matrix."
    Petersen (2009), "Estimating Standard Errors in Finance Panel Data Sets."

Matrix-row: compute_fm_betas, fama_macbeth, pooled_ols, beta_sign_consistency | (INDIVIDUAL, CONTINUOUS, FM, PANEL) | CS-first | NW HAC / clustered t | _newey_west_t_test, _p_value_from_t, _significance_marker, _short_circuit_output
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl

from factrix._codes import WarningCode
from factrix._stats import (
    _newey_west_t_test,
    _p_value_from_t,
    _significance_marker,
)
from factrix._types import DDOF, EPSILON, MetricOutput, ShankenVarSource
from factrix.metrics._helpers import _short_circuit_output

# Two-tier sample-size guard on the FM β series. ``T < HARD`` short-
# circuits — NW HAC SE on a 3-period series is undefined. ``HARD ≤ T <
# WARN`` returns the stat with ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS``
# attached (literature floor: Fama-MacBeth originally used T~30+; below
# that the asymptotic t is borderline). ``T ≥ WARN`` is silent.
MIN_FM_PERIODS_HARD: int = 4
MIN_FM_PERIODS_WARN: int = 30


# ---------------------------------------------------------------------------
# Raw computation (parallel to compute_ic)
# ---------------------------------------------------------------------------


def compute_fm_betas(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    r"""Per-date cross-sectional OLS: $R_i = \alpha + \beta \cdot \text{Signal}_i + \varepsilon$.

    Args:
        df: Long panel with ``date, asset_id, factor, forward_return``.
        factor_col: Column carrying the factor exposure.
        return_col: Column carrying the forward return.

    Returns:
        DataFrame with ``date, beta`` (one row per date that admits a
        finite OLS solution; dates with fewer than 3 observations or
        a singular design are dropped).

    Notes:
        Per date $t$, solve the cross-sectional OLS
        $R_{i,t} = \alpha_t + \beta_t \cdot \text{Signal}_{i,t} + \varepsilon_{i,t}$
        and emit the slope $\beta_t$. The output series feeds the
        stage-2 NW HAC $t$-test in
        ``fama_macbeth``.

        factrix drops dates with fewer than 3 cross-sectional
        observations or a singular design rather than coercing to NaN —
        this keeps stage-2 a clean t-test on a finite, well-defined
        series with no NaN propagation in the NW kernel.

    References:
        [Fama-MacBeth 1973][fama-macbeth-1973]: the per-date cross-
        sectional regression at stage 1 of the FM procedure.
    """
    dates = df["date"].unique().sort()
    rows: list[dict] = []

    for dt in dates:
        chunk = df.filter(pl.col("date") == dt)
        y = chunk[return_col].to_numpy().astype(np.float64)
        x = chunk[factor_col].to_numpy().astype(np.float64)

        if len(y) < 3:
            continue

        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        rows.append({"date": dt, "beta": float(beta[1])})

    if not rows:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Datetime("ms")),
                "beta": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fama-MacBeth significance (parallel to ic())
# ---------------------------------------------------------------------------


def fama_macbeth(
    beta_df: pl.DataFrame,
    *,
    newey_west_lags: int | None = None,
    forward_periods: int | None = None,
    is_estimated_factor: bool = False,
    factor_return_var: float | None = None,
) -> MetricOutput:
    r"""Newey-West t-test on FM beta series. $H_0: \mathrm{mean}(\beta) = 0$.

    Args:
        beta_df: DataFrame with ``date, beta`` columns (from compute_fm_betas).
        newey_west_lags: Number of NW lags. Defaults to $\lfloor T^{1/3} \rfloor$.
        forward_periods: Overlap horizon of the regression's forward
            return. When set, the NW bandwidth is floored at
            ``forward_periods - 1`` so the kernel is consistent under
            the MA($h-1$) overlap structure of $h$-period returns.
        is_estimated_factor: Set True when the ``Signal_i`` column used by
            ``compute_fm_betas`` is itself an **estimated** quantity
            (rolling OLS $\beta$ to another factor, PCA score,
            ML-predicted score, residual from a first-stage regression).
            Shanken (1992) shows the naive FM SE ignores sampling error
            in the regressor, inflating $t$-stats. **Do NOT** set this
            on raw characteristics (book-to-market, momentum price
            signal, accounting ratios) — those are observed, not
            estimated, and enabling the correction will spuriously
            deflate $t$-stats.

            Implementation: Kan-Zhang (1999) single-factor simplification
            — the NW SE is scaled by $\sqrt{1 + \hat\lambda^2/\sigma^2_f}$.
            This *omits* the additive $+\sigma^2_f/T$ term of the full
            Shanken variance and is therefore only honest for large $T$.

        factor_return_var: $\sigma^2_f$, the time-series variance of the
            factor-mimicking portfolio return. Prefer supplying this when
            you have a spread-portfolio return series (the long-short
            spread actually traded on the signal). When ``None`` and
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
        With ``is_estimated_factor=True``, the Shanken-Kan-Zhang
        single-factor correction scales SE by
        $\sqrt{1 + \overline{\beta}^2 / \sigma^2_f}$.

        factrix uses the Andrews (1991) $T^{1/3}$ bandwidth floored
        against the Hansen-Hodrick overlap horizon rather than the
        Newey-West (1994) data-adaptive plug-in — simpler, deterministic,
        and adequate at typical research $T$. The Kan-Zhang simplification
        omits the additive $+\sigma^2_f / T$ term of full Shanken EIV,
        so the correction is honest only for large $T$.

    References:
        [Fama-MacBeth 1973][fama-macbeth-1973]: two-stage lambda
        procedure underlying this test.
        [Newey-West 1987][newey-west-1987]: HAC variance estimator.
        [Andrews 1991][andrews-1991]: optimal Bartlett growth rate.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: overlap horizon
        flooring the kernel.
        [Shanken 1992][shanken-1992]: errors-in-variables correction
        for FM stage-2 t when the regressor is itself estimated.
        [Kan-Zhang 1999][kan-zhang-1999]: single-factor simplification
        of the Shanken EIV correction that factrix actually applies.
    """
    betas = beta_df["beta"].drop_nulls().to_numpy()
    n = len(betas)

    if n < MIN_FM_PERIODS_HARD:
        return _short_circuit_output(
            "fm_beta",
            "insufficient_fm_periods",
            n_observed=n,
            min_required=MIN_FM_PERIODS_HARD,
        )

    warning_codes: list[str] = []
    if n < MIN_FM_PERIODS_WARN:
        warning_codes.append(WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value)
        warnings.warn(
            f"fama_macbeth: n_periods={n} below MIN_FM_PERIODS_WARN="
            f"{MIN_FM_PERIODS_WARN}; NW HAC SE on a short β series is "
            f"borderline (Fama-MacBeth convention is T≥30). t-stat is "
            f"returned but read p-values cautiously.",
            UserWarning,
            stacklevel=2,
        )

    from factrix._stats import _resolve_nw_lags

    mean_beta = float(np.mean(betas))
    t, p, sig = _newey_west_t_test(
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
            sig_shanken = _significance_marker(p_shanken)
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
            t, sig = t_shanken, sig_shanken

    return MetricOutput(
        name="fm_beta",
        value=mean_beta,
        stat=t,
        significance=sig,
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


def pooled_ols(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    cluster_col: str = "date",
    two_way_cluster_col: str | None = None,
) -> MetricOutput:
    r"""Pooled OLS with clustered SE — robustness check against FM.

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
        set — Cameron-Gelbach-Miller 2011 / Petersen 2009):

        $$
        V_{\text{two-way}} = V_A + V_B - V_{A \cap B}
        $$

        where $V_A$, $V_B$, $V_{A \cap B}$ are single-way variances
        clustered on $A$, on $B$, and on the intersection cells
        $(A, B)$. Each component uses its own finite-sample correction.
        $\mathrm{df} = \min(G_A, G_B) - 1$ (Thompson 2011).

    Clustering on date alone catches contemporaneous cross-sectional
    dependence but misses asset-level persistence; on asset alone the
    reverse. Petersen (2009) shows panel data usually has both —
    single-way clusters understate SE by 20-50% in that regime.

    FM and single-way share the same point estimate under a balanced
    panel but typically disagree on SE; when $\hat\beta$ and FM
    $\hat\lambda$ have **opposite signs**, ``profile.diagnose()``
    flags an FM/pooled sign-mismatch — a red flag for misspecification.

    Short-circuits when $N < 10$ (no regression), returns ``stat=None``
    with $p=1.0$ when the effective $G < 3$ (SE undefined with < 3
    clusters).

    Notes:
        Pool ``(date, asset)`` rows and run a single OLS ``R = alpha +
        beta * Signal + eps`` with the appropriate cluster-robust
        sandwich covariance described above. Single-way: ``df = G - 1``
        with ``G`` the number of clusters; two-way:
        ``df = min(G_A, G_B) - 1`` per Thompson (2011).

        factrix reports ``stat = None`` (rather than 0) when ``G < 3``
        because the cluster-robust variance is undefined with too few
        clusters; falling back to a homoskedastic SE in that regime
        would silently break the panel-correlation guarantee that
        motivated using clustered SE in the first place.

    References:
        [Petersen 2009][petersen-2009]: comparison of FM, clustered, and
        two-way SE under firm/time correlation.
        [Newey-West 1987][newey-west-1987]: HAC variance ancestor of the
        sandwich form used here.
        Cameron, Gelbach & Miller (2011), "Robust Inference With
        Multiway Clustering."
        Thompson (2011), "Simple Formulas for Standard Errors that
        Cluster by Both Firm and Time."
    """
    y = df[return_col].to_numpy().astype(np.float64)
    x = df[factor_col].to_numpy().astype(np.float64)
    n_obs = len(y)

    if n_obs < 10:
        return _short_circuit_output(
            "pooled_beta",
            "insufficient_pooled_observations",
            n_observed=n_obs,
            min_required=10,
        )

    X = np.column_stack([np.ones(n_obs), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return _short_circuit_output(
            "pooled_beta",
            "singular_pooled_design_matrix",
            n_observed=n_obs,
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
            return MetricOutput(
                name="pooled_beta",
                value=slope,
                stat=None,
                significance="",
                metadata={
                    "reason": "insufficient_clusters",
                    "n_observed": g_a,
                    "min_required": 3,
                    "n_obs": n_obs,
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
            return MetricOutput(
                name="pooled_beta",
                value=slope,
                stat=None,
                significance="",
                metadata={
                    "reason": "insufficient_clusters",
                    "n_observed": min(g_a, g_b),
                    "min_required": 3,
                    "n_obs": n_obs,
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
        return MetricOutput(
            name="pooled_beta",
            value=slope,
            stat=0.0,
            significance="",
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
        "n_obs": n_obs,
        **cluster_metadata,
    }
    if non_psd_fallback:
        metadata["variance_non_psd_fallback"] = f"one_way_{cluster_col}"

    return MetricOutput(
        name="pooled_beta",
        value=slope,
        stat=t_stat,
        significance=_significance_marker(p),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Beta sign consistency (parallel to hit_rate)
# ---------------------------------------------------------------------------


def beta_sign_consistency(
    beta_df: pl.DataFrame,
    *,
    expected_sign: int = 1,
) -> MetricOutput:
    r"""Fraction of FM per-date $\beta$s carrying the expected sign — ``value`` $= \mathrm{mean}_t \mathbb{1}\{\mathrm{sign}(\beta_t) = s^\star\}$.

    $\beta_t$ is the per-date OLS $\beta$ from ``compute_fm_betas``.
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
        pair with ``fama_macbeth`` for inferential significance.

        factrix splits this directional check from the symmetric
        ``ts_beta_sign_consistency`` so the two answer different
        questions: this one requires the caller to commit to a prior
        sign; the symmetric variant tests cross-asset agreement only.
    """
    betas = beta_df["beta"].drop_nulls().to_numpy()
    n = len(betas)
    if n == 0:
        return _short_circuit_output(
            "beta_sign_consistency",
            "no_beta_observations",
            n_observed=0,
            min_required=1,
        )

    if expected_sign >= 0:
        consistent = float(np.mean(betas > 0))
    else:
        consistent = float(np.mean(betas < 0))

    return MetricOutput(
        name="beta_sign_consistency",
        value=consistent,
        metadata={
            "expected_sign": expected_sign,
            "n_periods": n,
        },
    )
