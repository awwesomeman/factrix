"""Time-series beta metrics for macro common factors.

Aggregation: per-asset full-sample OLS β (time-series step), then
cross-asset t on the β distribution; rolling-window variant slices
the time axis before the per-asset step.

macro_common factors (VIX, gold, USD index) are a single time series
shared across all assets. Per-asset time-series regression measures
each asset's sensitivity (β) to the common factor.

``compute_ts_betas``: per-asset full-sample TS regression.
``ts_beta``: cross-sectional test on the β distribution.
``mean_r_squared``: average explanatory power across assets.
``compute_rolling_mean_beta``: rolling window mean β for stability analysis.

Matrix-row: compute_ts_betas, ts_beta, mean_r_squared, compute_rolling_mean_beta, ts_beta_sign_consistency, ts_beta_single_asset_fallback | (COMMON, CONTINUOUS, *, PANEL) | TS-first | cross-asset t | _calc_t_stat, _p_value_from_t, _significance_marker, _short_circuit_output
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import DDOF, EPSILON, MetricOutput
from factrix.metrics._helpers import _short_circuit_output
from factrix._stats import (
    _calc_t_stat,
    _p_value_from_t,
    _significance_marker,
)


MIN_TS_OBS: int = 20


# ---------------------------------------------------------------------------
# Per-asset TS regression
# ---------------------------------------------------------------------------

def compute_ts_betas(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-asset time-series OLS: R_{i,t} = α_i + β_i · F_t + ε.

    Args:
        df: Long panel with ``date, asset_id, factor, forward_return``.
        factor_col: Column carrying the (broadcast) factor.
        return_col: Column carrying the per-asset forward return.

    Returns:
        DataFrame with ``asset_id, beta, alpha, t_stat, r_squared,
        n_obs``. Assets with fewer than ``MIN_TS_OBS`` valid rows or a
        singular design are dropped.

    Notes:
        Per asset ``i``, run OLS ``R_{i,t} = alpha_i + beta_i * F_t +
        eps`` over the asset's full sample with homoskedastic SE; emit
        ``beta_i, alpha_i, t_i, R^2_i, n_i``. The output is the
        per-asset stage feeding the cross-asset Black-Jensen-Scholes
        style aggregation in ``ts_beta``.

        factrix reports homoskedastic per-asset t (not NW HAC) at this
        stage because the inferential burden lives downstream — the
        cross-asset t in ``ts_beta`` is what a caller decides on, and
        adding HAC at the per-asset stage would only smear the
        per-asset diagnostic without affecting stage-2 inference.

    References:
        [Black-Jensen-Scholes 1972][black-jensen-scholes-1972]: per-
        asset time-series beta then cross-asset aggregation; the
        order this two-stage path mirrors.
    """
    assets = df["asset_id"].unique().sort()
    rows: list[dict] = []

    for asset in assets:
        chunk = df.filter(pl.col("asset_id") == asset).sort("date")
        y = chunk[return_col].drop_nulls().to_numpy().astype(np.float64)
        x = chunk[factor_col].drop_nulls().to_numpy().astype(np.float64)

        n = min(len(y), len(x))
        if n < MIN_TS_OBS:
            continue
        y, x = y[:n], x[:n]

        X = np.column_stack([np.ones(n), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha_val = float(beta[0])
        beta_val = float(beta[1])

        resid = y - X @ beta
        ss_res = float(np.dot(resid, resid))
        centered = y - np.mean(y)
        ss_tot = float(np.dot(centered, centered))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0

        dof = n - 2
        if dof > 0 and ss_res / dof > EPSILON:
            sigma2 = ss_res / dof
            try:
                xtx_inv = np.linalg.inv(X.T @ X)
                se_beta = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
                t_stat = beta_val / se_beta if se_beta > EPSILON else 0.0
            except np.linalg.LinAlgError:
                t_stat = 0.0
        else:
            t_stat = 0.0

        rows.append({
            "asset_id": asset,
            "beta": beta_val,
            "alpha": alpha_val,
            "t_stat": t_stat,
            "r_squared": r_sq,
            "n_obs": n,
        })

    if not rows:
        return pl.DataFrame({
            "asset_id": pl.Series([], dtype=pl.String),
            "beta": pl.Series([], dtype=pl.Float64),
            "alpha": pl.Series([], dtype=pl.Float64),
            "t_stat": pl.Series([], dtype=pl.Float64),
            "r_squared": pl.Series([], dtype=pl.Float64),
            "n_obs": pl.Series([], dtype=pl.Int64),
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-sectional test on β distribution
# ---------------------------------------------------------------------------

def ts_beta_single_asset_fallback(ts_betas_df: pl.DataFrame) -> MetricOutput:
    """N=1 fallback: report the single-asset regression's own t-stat.

    The cross-sectional t-test in ``ts_beta`` needs N≥2 assets. With a
    single asset, both ``MacroCommonProfile.from_artifacts`` and
    ``MacroCommonFactor.ts_beta`` want the same degenerate-case output:
    take the row's per-asset beta + t_stat, mark ``p_value=1.0`` so the
    row is suppressed from BHY, and label the method. Centralizing here
    keeps Profile and Factor paths bit-identical.

    Statistical caveat: the returned t-stat tests the **time-series**
    hypothesis ``H₀: β_i = 0 for this asset``, which is **not** the
    ``ts_beta`` cross-sectional hypothesis ``H₀: mean(β) = 0 across
    assets``. The two are not exchangeable — a single-asset t-stat of
    2.5 says that asset's β differs from zero over time, it does not
    say the common factor is priced. ``p_value=1.0`` enforces this by
    keeping the row out of BHY adjudication.

    Notes:
        N=1 path: ``value = beta_1``, ``stat = t_1`` from the single
        asset's TS OLS, ``p_value = 1.0`` so BHY skips the row. No
        cross-asset inference is attempted.

        factrix centralises the degenerate-case output here so
        Profile / Factor paths produce bit-identical metadata —
        otherwise each entry point would coerce the single-asset row
        differently and the verdict surface would diverge.
    """
    row = ts_betas_df.row(0, named=True)
    return MetricOutput(
        name="ts_beta",
        value=float(row["beta"]),
        stat=float(row["t_stat"]),
        metadata={
            "n_assets": 1,
            "p_value": 1.0,
            "method": "single-asset TS regression (no cross-asset test)",
        },
    )


def ts_beta(ts_betas_df: pl.DataFrame) -> MetricOutput:
    """Test H₀: mean(β) = 0 across assets.

    Uses the cross-sectional distribution of per-asset betas.

    Notes:
        Stage 2 of the BJS aggregation: ``mean_beta = mean_i beta_i``;
        ``t = mean_beta / (std(beta) / sqrt(N))`` with ``H0: E[beta] = 0``
        across assets. The std is the sample cross-sectional std with
        ``ddof=1``.

        factrix uses an iid cross-asset t at this stage rather than a
        clustered/HAC variant: per-asset betas come from non-overlapping
        time-series fits in ``compute_ts_betas``, so the betas are
        approximately independent across assets unless a strong
        latent common factor links them.

    References:
        [Black-Jensen-Scholes 1972][black-jensen-scholes-1972]: the
        cross-asset t on E[beta] this function implements.
    """
    betas = ts_betas_df["beta"].drop_nulls().to_numpy()
    n = len(betas)

    if n < 3:
        return _short_circuit_output(
            "ts_beta", "insufficient_assets",
            n_observed=n, min_required=3,
        )

    mean_b = float(np.mean(betas))
    std_b = float(np.std(betas, ddof=DDOF))
    t = _calc_t_stat(mean_b, std_b, n)
    p = _p_value_from_t(t, n)

    return MetricOutput(
        name="ts_beta",
        value=mean_b,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "p_value": p,
            "stat_type": "t",
            "h0": "mean(β)=0",
            "method": "cross-sectional t-test on per-asset TS betas",
            "n_assets": n,
            "beta_std": std_b,
            "median_beta": float(np.median(betas)),
        },
    )


# ---------------------------------------------------------------------------
# Mean R²
# ---------------------------------------------------------------------------

def mean_r_squared(ts_betas_df: pl.DataFrame) -> MetricOutput:
    """Average R² across per-asset TS regressions — `value = mean_i R²_i`.

    R²_i comes from asset i's regression R_{i,t} = α_i + β_i·F_t + ε
    (computed upstream in ``compute_ts_betas``). Metadata carries
    ``median_r_squared`` as well — useful when a few high-R² assets
    pull the mean. Low values (< 0.05) indicate the factor is too
    weak or noisy to drive individual-asset returns even when its
    cross-asset mean β looks nonzero.

    Short-circuits to NaN when no assets have a non-null R².

    Notes:
        ``value = mean_i R^2_i`` and ``median_r_squared = median_i R^2_i``
        on the per-asset OLS fits from ``compute_ts_betas``. Pure
        descriptive statistic — no formal H0.

        factrix reports both mean and median because a few high-R^2
        assets can dominate the mean; large mean-vs-median gaps signal
        the factor explains a small subset of assets rather than the
        cross-section as a whole.
    """
    r2_vals = ts_betas_df["r_squared"].drop_nulls().to_numpy()
    n = len(r2_vals)

    if n == 0:
        return _short_circuit_output(
            "mean_r_squared", "no_asset_r_squared_observations",
            n_observed=0, min_required=1,
        )

    return MetricOutput(
        name="mean_r_squared",
        value=float(np.mean(r2_vals)),
        metadata={
            "n_assets": n,
            "median_r_squared": float(np.median(r2_vals)),
            "min_r_squared": float(np.min(r2_vals)),
            "max_r_squared": float(np.max(r2_vals)),
        },
    )


# ---------------------------------------------------------------------------
# Rolling mean beta for stability / OOS analysis
# ---------------------------------------------------------------------------

def compute_rolling_mean_beta(
    df: pl.DataFrame,
    *,
    window: int = 60,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Rolling-window mean β across assets — time-series input for OOS / trend.

    Formula (per date t ≥ ``window``):
        For each asset i, take the trailing ``window`` rows ending at t.
        If ≥ 10 valid (factor, return) pairs, run OLS:
            R_{i,s} = α_i + β_i·F_s + ε   (s in window)
        β_t = mean_i β_i   (cross-asset mean of this window's βs)

    Dates with fewer than ``window`` trailing rows are skipped. Assets
    with < 10 valid obs in the window are dropped from that date's β
    calculation. If no asset qualifies at a given date, that date is
    absent from the output entirely.

    Returns:
        DataFrame with ``date, value`` where ``value`` is the rolling
        cross-asset mean β. Shape compatible with ``oos`` / ``ic_trend``.

    Notes:
        Per date ``t >= window``, run the per-asset TS OLS over the
        trailing ``window`` rows and compute ``value_t = mean_i beta_i``.
        Output schema matches the time-series tools (``oos`` /
        ``ic_trend``), so callers can pipe rolling betas into stability
        and trend diagnostics.

        factrix requires at least 10 valid rows per asset within each
        rolling window; below that, the asset is dropped from that
        date's mean rather than imputed — keeps each ``value_t`` an
        average over identifiable per-asset slopes.
    """
    dates = df["date"].unique().sort()
    if len(dates) < window:
        return pl.DataFrame({
            "date": pl.Series([], dtype=pl.Datetime("ms")),
            "value": pl.Series([], dtype=pl.Float64),
        })

    rows: list[dict] = []
    for i in range(window, len(dates)):
        window_dates = dates[i - window:i]
        chunk = df.filter(pl.col("date").is_in(window_dates.implode()))

        betas_per_asset: list[float] = []
        for asset in chunk["asset_id"].unique():
            a_data = chunk.filter(pl.col("asset_id") == asset)
            y = a_data[return_col].to_numpy().astype(np.float64)
            x = a_data[factor_col].to_numpy().astype(np.float64)
            n = min(len(y), len(x))
            if n < 10:
                continue
            y, x = y[:n], x[:n]
            X = np.column_stack([np.ones(n), x])
            try:
                b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                betas_per_asset.append(float(b[1]))
            except np.linalg.LinAlgError:
                continue

        if betas_per_asset:
            rows.append({
                "date": dates[i],
                "value": float(np.mean(betas_per_asset)),
            })

    if not rows:
        return pl.DataFrame({
            "date": pl.Series([], dtype=pl.Datetime("ms")),
            "value": pl.Series([], dtype=pl.Float64),
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# β sign consistency (per-asset version)
# ---------------------------------------------------------------------------

def ts_beta_sign_consistency(ts_betas_df: pl.DataFrame) -> MetricOutput:
    """Symmetric sign-agreement across per-asset βs — `value = max(pos, 1−pos)` where `pos = mean_i 1{β_i > 0}`.

    Range [0.5, 1.0]: 0.5 = βs evenly split (no directional consensus);
    1.0 = all βs share one sign. Unlike
    ``fama_macbeth.beta_sign_consistency`` this is **direction-agnostic**
    — it does not require a prior on the factor's expected sign.

    Requires N ≥ 2: a single β is trivially "100% consistent with
    itself" (the max collapses to 1.0 for any nonzero β), which would
    read as strong evidence on a dashboard but carries zero information.
    Short-circuits to NaN in that case so the degenerate value never
    leaks into verdict decisions.

    Notes:
        ``pos = mean_i 1{beta_i > 0}``; ``value = max(pos, 1 - pos)``.
        Direction-agnostic: returns 1 when all assets have positive
        beta or all negative.

        factrix gates this metric at ``N >= 2`` so a single-asset
        ``max(pos, 1-pos) = 1.0`` cannot leak into verdict surfaces as
        spurious "perfect agreement". Pair with
        ``fama_macbeth.beta_sign_consistency`` when a directional prior
        is available.
    """
    betas = ts_betas_df["beta"].drop_nulls().to_numpy()
    n = len(betas)
    if n < 2:
        return _short_circuit_output(
            "ts_beta_sign_consistency", "insufficient_assets_for_sign_consistency",
            n_observed=n, min_required=2,
        )

    positive = float(np.mean(betas > 0))
    consistency = max(positive, 1.0 - positive)

    return MetricOutput(
        name="ts_beta_sign_consistency",
        value=consistency,
        metadata={
            "n_assets": n,
            "fraction_positive": positive,
        },
    )
