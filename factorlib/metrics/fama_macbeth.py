"""Fama-MacBeth regression and macro panel metrics.

``compute_fm_betas``: per-date cross-sectional OLS → (date, beta) DataFrame.
``fama_macbeth``: Newey-West t-test on the beta series.
``pooled_ols``: pooled OLS with clustered SE by date.
``beta_sign_consistency``: fraction of periods with correct beta sign.

References:
    Fama & MacBeth (1973), "Risk, Return, and Equilibrium."
    Newey & West (1987), "HAC Covariance Matrix."
    Petersen (2009), "Estimating Standard Errors in Finance Panel Data Sets."
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from factorlib._types import DDOF, EPSILON, MetricOutput
from factorlib.metrics._helpers import _short_circuit_output
from factorlib._stats import (
    _newey_west_t_test,
    _p_value_from_t,
    _significance_marker,
)

MIN_FM_PERIODS: int = 20


# ---------------------------------------------------------------------------
# Raw computation (parallel to compute_ic)
# ---------------------------------------------------------------------------

def compute_fm_betas(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-date cross-sectional OLS: R_i = α + β · Signal_i + ε.

    Returns DataFrame with columns ``date, beta``.
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
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Datetime("ms")),
                             "beta": pl.Series([], dtype=pl.Float64)})

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fama-MacBeth significance (parallel to ic())
# ---------------------------------------------------------------------------

def fama_macbeth(
    beta_df: pl.DataFrame,
    *,
    newey_west_lags: int | None = None,
) -> MetricOutput:
    """Newey-West t-test on FM beta series. H₀: mean(β) = 0.

    Args:
        beta_df: DataFrame with ``date, beta`` columns (from compute_fm_betas).
        newey_west_lags: Number of NW lags. Defaults to floor(T^(1/3)).
    """
    betas = beta_df["beta"].drop_nulls().to_numpy()
    n = len(betas)

    if n < MIN_FM_PERIODS:
        return _short_circuit_output(
            "fm_beta", "insufficient_fm_periods",
            n_observed=n, min_required=MIN_FM_PERIODS,
        )

    mean_beta = float(np.mean(betas))
    t, p, sig = _newey_west_t_test(betas, lags=newey_west_lags)
    actual_lags = newey_west_lags if newey_west_lags is not None else int(np.floor(n ** (1 / 3)))

    return MetricOutput(
        name="fm_beta",
        value=mean_beta,
        stat=t,
        significance=sig,
        metadata={
            "p_value": p,
            "stat_type": "t",
            "h0": "mean(β)=0",
            "method": "Fama-MacBeth + Newey-West",
            "n_periods": n,
            "newey_west_lags": actual_lags,
        },
    )


# ---------------------------------------------------------------------------
# Pooled OLS with clustered SE
# ---------------------------------------------------------------------------

def pooled_ols(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    cluster_col: str = "date",
) -> MetricOutput:
    """Pooled OLS with date-clustered SE — robustness check against FM.

    Formula:
        Point estimate:
            [α̂, β̂] = (X'X)⁻¹ X'R   where X = [1, Signal] stacked across
                                    all (date, asset) observations
        Clustered sandwich SE (cluster on date):
            meat = Σ_g (X_g' e_g)(X_g' e_g)'   over date groups g
            V = c · (X'X)⁻¹ · meat · (X'X)⁻¹
            c = G/(G−1) · (N−1)/(N−K)    (finite-sample correction)
            SE(β̂) = √V[1,1]
            t = β̂ / SE
        G = number of date clusters, N = total obs, K = 2 (α + β).

    Clustering by date accounts for within-date cross-sectional
    correlation (contemporaneously correlated residuals across assets).
    FM and this share the same point estimate under a balanced panel
    but typically disagree on SE; when β̂ and FM λ̂ have **opposite
    signs**, the ``macro_panel.fm_pooled_sign_mismatch`` veto rule
    fires — a red flag for misspecification.

    Short-circuits when N < 10 (no regression), returns stat=None with
    p=1.0 when G < 3 (SE undefined with < 3 date clusters).
    """
    y = df[return_col].to_numpy().astype(np.float64)
    x = df[factor_col].to_numpy().astype(np.float64)
    n_obs = len(y)

    if n_obs < 10:
        return _short_circuit_output(
            "pooled_beta", "insufficient_pooled_observations",
            n_observed=n_obs, min_required=10,
        )

    X = np.column_stack([np.ones(n_obs), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return _short_circuit_output(
            "pooled_beta", "singular_pooled_design_matrix",
            n_observed=n_obs,
        )

    slope = float(beta[1])
    resid = y - X @ beta

    # Clustered SE by date
    clusters = df[cluster_col].to_numpy()
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    if n_clusters < 3:
        return MetricOutput(
            name="pooled_beta", value=slope, stat=None, significance="",
            metadata={
                "reason": "insufficient_clusters",
                "n_observed": n_clusters,
                "min_required": 3,
                "n_obs": n_obs,
                "p_value": 1.0,
            },
        )

    # B = Σ_g (X_g' e_g)(X_g' e_g)' — the "meat" of the sandwich
    k = X.shape[1]
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        X_g = X[mask]
        e_g = resid[mask]
        score = X_g.T @ e_g
        meat += np.outer(score, score)

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return MetricOutput(
            name="pooled_beta", value=slope, stat=0.0, significance="",
        )

    # WHY: finite-sample correction factor = G/(G-1) * (N-1)/(N-K)
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k))
    V = correction * xtx_inv @ meat @ xtx_inv
    se_slope = float(np.sqrt(max(V[1, 1], 0.0)))

    if se_slope < EPSILON:
        t_stat = 0.0
    else:
        t_stat = slope / se_slope

    p = _p_value_from_t(t_stat, n_clusters)

    return MetricOutput(
        name="pooled_beta",
        value=slope,
        stat=t_stat,
        significance=_significance_marker(p),
        metadata={
            "p_value": p,
            "stat_type": "t",
            "h0": "β=0",
            "method": "Pooled OLS + clustered SE (by date)",
            "n_obs": n_obs,
            "n_clusters": n_clusters,
        },
    )


# ---------------------------------------------------------------------------
# Beta sign consistency (parallel to hit_rate)
# ---------------------------------------------------------------------------

def beta_sign_consistency(
    beta_df: pl.DataFrame,
    *,
    expected_sign: int = 1,
) -> MetricOutput:
    """Fraction of FM per-date βs carrying the expected sign — `value = mean_t 1{sign(β_t) == expected_sign}`.

    β_t is the per-date OLS β from ``compute_fm_betas``. Range [0, 1];
    1.0 = β always has the expected sign across periods. Unlike
    ``ts_beta_sign_consistency`` (which symmetrizes via
    ``max(pos%, 1-pos%)``), this one is directional — you must supply
    the a-priori expected sign. Typical use: paired with a prior on
    factor direction to check stability.

    Short-circuits to NaN when no non-null β observations exist.
    """
    betas = beta_df["beta"].drop_nulls().to_numpy()
    n = len(betas)
    if n == 0:
        return _short_circuit_output(
            "beta_sign_consistency", "no_beta_observations",
            n_observed=0, min_required=1,
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


