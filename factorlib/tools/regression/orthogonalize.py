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

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def orthogonalize_factor(
    factor_df: pl.DataFrame,
    base_factors: pl.DataFrame,
    factor_col: str = "factor",
    base_cols: list[str] | None = None,
) -> pl.DataFrame:
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
        ``factor_df`` with ``factor_col`` replaced by the OLS residual.
        A new column ``factor_pre_ortho`` preserves the original value.
    """
    if base_cols is None:
        base_cols = [
            c for c in base_factors.columns if c not in ("date", "asset_id")
        ]

    if not base_cols:
        logger.warning("orthogonalize_factor: no base_cols specified, returning unchanged")
        return factor_df

    # WHY: join 確保 date × asset_id 對齊
    merged = factor_df.join(
        base_factors.select(["date", "asset_id", *base_cols]),
        on=["date", "asset_id"],
        how="inner",
    )

    dates = merged["date"].unique().sort()
    residuals_list: list[pl.DataFrame] = []

    for dt in dates:
        chunk = merged.filter(pl.col("date") == dt)
        y = chunk[factor_col].to_numpy().astype(np.float64)
        X = chunk.select(base_cols).to_numpy().astype(np.float64)

        # WHY: 加截距項讓回歸去均值
        ones = np.ones((len(y), 1))
        X_with_intercept = np.hstack([ones, X])

        # WHY: 處理共線性或全零列（某日某產業無觀測）
        try:
            # lstsq 比 inv(X'X)X'y 數值穩定
            beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            residual = y - X_with_intercept @ beta
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
        return factor_df

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

    return result
