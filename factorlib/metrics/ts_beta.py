"""Time-series beta metrics for macro common factors.

macro_common factors (VIX, gold, USD index) are a single time series
shared across all assets. Per-asset time-series regression measures
each asset's sensitivity (β) to the common factor.

``compute_ts_betas``: per-asset full-sample TS regression.
``ts_beta``: cross-sectional test on the β distribution.
``mean_r_squared``: average explanatory power across assets.
``compute_rolling_mean_beta``: rolling window mean β for stability analysis.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import DDOF, EPSILON, MetricOutput
from factorlib._stats import (
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

    Returns DataFrame with ``asset_id, beta, alpha, t_stat, r_squared, n_obs``.
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
    """
    betas = ts_betas_df["beta"].drop_nulls().to_numpy()
    n = len(betas)

    if n < 3:
        return MetricOutput(
            name="ts_beta", value=0.0, stat=0.0, significance="",
            metadata={
                "reason": "insufficient_assets",
                "n_observed": n,
                "min_required": 3,
            },
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
    """Average R² across assets — explanatory power of the common factor."""
    r2_vals = ts_betas_df["r_squared"].drop_nulls().to_numpy()
    n = len(r2_vals)

    if n == 0:
        return MetricOutput(
            name="mean_r_squared", value=0.0,
            metadata={
                "reason": "no_asset_r_squared_observations",
                "n_observed": 0,
                "min_required": 1,
            },
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
    """Rolling-window mean β across assets.

    At each date, compute per-asset TS β using the trailing window,
    then take the cross-asset mean. Returns ``date, value`` for
    compatibility with OOS/trend tools.
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
    """Fraction of assets with β > 0 (or consistent sign)."""
    betas = ts_betas_df["beta"].drop_nulls().to_numpy()
    n = len(betas)
    if n == 0:
        return MetricOutput(
            name="ts_beta_sign_consistency", value=0.0,
            metadata={
                "reason": "no_beta_observations",
                "n_observed": 0,
                "min_required": 1,
            },
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
