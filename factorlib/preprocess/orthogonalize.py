"""Factor orthogonalization (Step 6): remove known factor exposures.

Per-date cross-sectional OLS regression:
    factor_z = β₁·Size + β₂·Value + β₃·Momentum + Σβ_k·Industry_k + ε

The residual ε replaces the original factor value, so that downstream
Gates and Profile reflect "pure alpha" net of known risk exposures.

This module is independently usable for any analysis that requires
"what remains after removing base factor influence."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorlib._types import EPSILON

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalizeResult:
    """Result of factor orthogonalization with attribution info."""

    df: pl.DataFrame
    mean_betas: dict[str, float] = field(default_factory=dict)
    mean_r_squared: float = 0.0
    n_dates: int = 0
    coverage: float = 0.0
    n_base: int = 0


def orthogonalize_factor(
    factor_df: pl.DataFrame,
    base_factors: pl.DataFrame,
    factor_col: str = "factor",
    base_cols: list[str] | None = None,
) -> OrthogonalizeResult:
    """Orthogonalize factor against base factors via per-date OLS.

    Args:
        factor_df: Panel with ``date, asset_id, {factor_col}``.
            factor_col should already be z-scored (Step 5 output).
        base_factors: Panel with ``date, asset_id`` and base factor columns.
            Industry dummies should be pre-encoded as 0/1 columns.
        factor_col: Column name of the factor to orthogonalize.
        base_cols: List of column names in ``base_factors`` to regress on.
            If None, uses all columns except ``date`` and ``asset_id``.

    Returns:
        OrthogonalizeResult with:
        - ``df``: factor_df with ``factor_col`` replaced by residual,
          ``factor_pre_ortho`` preserving original value.
        - ``mean_betas``: average beta per base factor across dates.
        - ``mean_r_squared``: average R² across dates.
    """
    if base_cols is None:
        base_cols = [
            c for c in base_factors.columns if c not in ("date", "asset_id")
        ]

    if not base_cols:
        logger.warning("orthogonalize_factor: no base_cols specified, returning unchanged")
        return OrthogonalizeResult(df=factor_df)

    # WHY: join 確保 date × asset_id 對齊
    merged = factor_df.join(
        base_factors.select(["date", "asset_id", *base_cols]),
        on=["date", "asset_id"],
        how="inner",
    )

    dates = merged["date"].unique().sort()
    residuals_list: list[pl.DataFrame] = []
    all_betas: list[np.ndarray] = []
    all_r2: list[float] = []

    for dt in dates:
        chunk = merged.filter(pl.col("date") == dt)
        y = chunk[factor_col].to_numpy().astype(np.float64)
        X = chunk.select(base_cols).to_numpy().astype(np.float64)

        # WHY: 加截距項讓回歸去均值
        ones = np.ones((len(y), 1))
        X_with_intercept = np.hstack([ones, X])

        # WHY: 處理共線性或全零列（某日某產業無觀測）
        try:
            beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            residual = y - X_with_intercept @ beta
            all_betas.append(beta[1:])  # exclude intercept
            ss_res = float(np.dot(residual, residual))
            centered = y - np.mean(y)
            ss_tot = float(np.dot(centered, centered))
            all_r2.append(1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0)
        except np.linalg.LinAlgError:
            logger.warning("orthogonalize: lstsq failed for date %s, keeping original", dt)
            residual = y

        residuals_list.append(
            chunk.select("date", "asset_id").with_columns(
                pl.Series(name="_residual", values=residual),
            )
        )

    if not residuals_list:
        logger.warning("orthogonalize_factor: no valid dates after join")
        return OrthogonalizeResult(df=factor_df)

    residuals_df = pl.concat(residuals_list)

    # WHY: 保留正交化前的原始值供比較分析
    result = (
        factor_df
        .with_columns(pl.col(factor_col).alias("factor_pre_ortho"))
        .join(residuals_df, on=["date", "asset_id"], how="left")
        .with_columns(
            pl.col("_residual").fill_null(pl.col(factor_col)).alias(factor_col)
        )
        .drop("_residual")
    )

    n_total = len(factor_df)
    n_ortho = len(residuals_df)
    n_dates = len(dates)
    n_base = len(base_cols)
    drop_pct = (n_total - n_ortho) / max(n_total, 1) * 100

    if drop_pct > 5:
        logger.warning(
            "orthogonalize_factor: %.1f%% of rows (%d/%d) not orthogonalized "
            "(base factor coverage gap — original values kept for those rows)",
            drop_pct, n_total - n_ortho, n_total,
        )

    logger.info(
        "orthogonalize_factor: processed %d dates, %d base factors, %.1f%% coverage",
        n_dates, n_base, 100 - drop_pct,
    )

    # Attribution: average betas and R² across dates
    mean_betas: dict[str, float] = {}
    mean_r2 = 0.0
    if all_betas:
        beta_matrix = np.array(all_betas)
        for i, col in enumerate(base_cols):
            mean_betas[col] = float(np.mean(beta_matrix[:, i]))
        mean_r2 = float(np.mean(all_r2))

    return OrthogonalizeResult(
        df=result,
        mean_betas=mean_betas,
        mean_r_squared=mean_r2,
        n_dates=n_dates,
        coverage=(n_ortho / n_total) if n_total else 0.0,
        n_base=n_base,
    )
