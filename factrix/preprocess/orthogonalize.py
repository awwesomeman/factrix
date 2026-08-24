"""Factor orthogonalization (Step 6): remove known factor exposures.

Per-date cross-sectional ordinary least squares (OLS) regression:
    factor_z = β₁·Size + β₂·Value + β₃·Momentum + Σβ_k·Industry_k + ε

The residual ε replaces the original factor value, so that downstream
``evaluate`` / metric calls see the factor net of known risk exposures.

This module is independently usable for any analysis that requires
"what remains after removing base factor influence."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factrix._types import EPSILON

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalizeResult:
    """Result of factor orthogonalization with attribution info.

    Attributes:
        data: ``factor_df`` with ``factor_col`` replaced by the residual and
            ``factor_pre_ortho`` preserving the original value.
        mean_betas: Average beta per base factor across regressed dates.
        mean_r_squared: Average R² across regressed dates.
        n_dates: Number of per-date cross-sections seen after the join.
        coverage: Fraction of input rows that carry an actual residual —
            rows dropped by the join, rows on a skipped date, and rows with
            non-finite inputs are all excluded.
        n_base: Number of base factor columns regressed on.
        n_rows_non_finite: Rows whose factor or base values were null / NaN /
            ±Inf on an otherwise regressed date. These rows are excluded from
            the regression and their residual is **null** in ``data``.
        n_dates_skipped: Dates left un-orthogonalised because too few finite
            rows remained (``< len(base_cols) + 2``) or ``lstsq`` failed.
            Original factor values are kept for those dates.
    """

    data: pl.DataFrame
    mean_betas: dict[str, float] = field(default_factory=dict)
    mean_r_squared: float = 0.0
    n_dates: int = 0
    coverage: float = 0.0
    n_base: int = 0
    n_rows_non_finite: int = 0
    n_dates_skipped: int = 0


def orthogonalize_factor(
    factor_df: pl.DataFrame,
    base_factors: pl.DataFrame,
    factor_col: str = "factor",
    base_cols: list[str] | None = None,
) -> OrthogonalizeResult:
    """Orthogonalize factor against base factors via per-date ordinary least squares (OLS).

    Args:
        factor_df: Panel with ``date, asset_id, {factor_col}``.
            factor_col should already be z-scored (Step 5 output).
        base_factors: Panel with ``date, asset_id`` and base factor columns.
            Industry dummies should be pre-encoded as 0/1 columns.
        factor_col: Column name of the factor to orthogonalize.
        base_cols: List of column names in ``base_factors`` to regress on.
            If None, uses all columns except ``date`` and ``asset_id``.

    Returns:
        OrthogonalizeResult with: ``data`` (factor_df with ``factor_col``
        replaced by the residual and ``factor_pre_ortho`` preserving the
        original value), ``mean_betas`` (average beta per base factor
        across dates), and ``mean_r_squared`` (average R² across dates).

    Raises:
        ValueError: ``base_factors`` carries duplicate ``(date, asset_id)``
            keys, which would fan the inner join out and duplicate rows.

    Notes:
        **Non-finite rows.** A single null / NaN / ±Inf in ``factor_col`` or
        in any base column used to make ``np.linalg.lstsq`` return all-NaN
        betas, which turned *every* asset on that date into NaN while the
        date still counted as orthogonalised. The regression now runs on the
        finite rows only. Rows that were not finite come back **null**, not
        with their original value: the residual is undefined for them, and
        keeping the raw value would mix two scales (raw factor and residual)
        inside one column — a silent, unrecoverable contamination. Their
        count is reported as ``n_rows_non_finite`` and logged.

        **Skipped dates.** A date with fewer than ``len(base_cols) + 2``
        finite rows (or one where ``lstsq`` raises) cannot support the
        regression; the original values are kept for that date, as before
        for ``lstsq`` failures, and the date is counted in
        ``n_dates_skipped``. Such dates are excluded from ``coverage``,
        ``mean_betas`` and ``mean_r_squared``.

        **Duplicate keys.** ``base_factors`` must be unique on
        ``(date, asset_id)``. A duplicated key silently fans the inner join
        out (panel height doubles, ``coverage`` exceeds 1.0), so it is
        rejected up front rather than de-duplicated with a guessed rule.

    Examples:
        >>> import factrix as fx
        >>> import polars as pl
        >>> from factrix.preprocess import (
        ...     cross_sectional_zscore,
        ...     orthogonalize_factor,
        ... )
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> factor_df = cross_sectional_zscore(raw).select(
        ...     "date", "asset_id", pl.col("factor_zscore").alias("factor")
        ... )
        >>> base = raw.with_columns(
        ...     pl.col("price").rank().over("date").alias("size")
        ... ).select("date", "asset_id", "size")
        >>> result = orthogonalize_factor(factor_df, base, base_cols=["size"])
        >>> "factor_pre_ortho" in result.data.columns
        True
        >>> isinstance(result.mean_r_squared, float)
        True
    """
    if base_cols is None:
        base_cols = [c for c in base_factors.columns if c not in ("date", "asset_id")]

    if not base_cols:
        logger.warning(
            "orthogonalize_factor: no base_cols specified, returning unchanged"
        )
        return OrthogonalizeResult(data=factor_df)

    # WHY: an inner join on duplicated keys fans out silently — the panel
    # grows rows that never existed and coverage climbs above 1.0. There is
    # no safe de-duplication rule to guess here (first? mean?), so refuse.
    if base_factors.select(["date", "asset_id"]).is_duplicated().any():
        n_dup = int(base_factors.select(["date", "asset_id"]).is_duplicated().sum())
        raise ValueError(
            "orthogonalize_factor: base_factors has duplicate (date, asset_id) "
            f"keys ({n_dup} rows involved). The inner join would fan out and "
            "duplicate panel rows; de-duplicate base_factors first "
            "(e.g. `base_factors.unique(subset=['date', 'asset_id'], keep='first')`)."
        )

    # WHY: join enforces date × asset_id alignment.
    merged = factor_df.join(
        base_factors.select(["date", "asset_id", *base_cols]),
        on=["date", "asset_id"],
        how="inner",
    )

    # WHY: partition_by yields per-date chunks in one pass (O(D × N)).
    # The previous implementation looped over unique dates and re-filtered
    # the merged DataFrame each time, which is O(D² × N) — measurable on
    # full-market panels with thousands of dates.
    residuals_list: list[pl.DataFrame] = []
    all_betas: list[np.ndarray] = []
    all_r2: list[float] = []
    n_rows_non_finite = 0
    n_dates_skipped = 0
    n_rows_orthogonalized = 0
    min_rows = len(base_cols) + 2

    # partition_by groups rows by date value; no pre-sort needed (the
    # earlier `.sort("date")` was dead weight at O(D×N log N)).
    # Per-date OLS is order-invariant; mean_betas / mean_r2 are sums.
    chunks = merged.partition_by("date", maintain_order=True)
    n_dates = len(chunks)

    for chunk in chunks:
        y = chunk[factor_col].to_numpy().astype(np.float64)
        X = chunk.select(base_cols).to_numpy().astype(np.float64)

        # WHY: intercept term de-means the regression.
        ones = np.ones((len(y), 1))
        X_with_intercept = np.hstack([ones, X])

        # WHY: lstsq propagates a single NaN/Inf into every beta, so one bad
        # row used to null out the whole cross-section. Regress on the finite
        # rows; the rest get a null residual (undefined, not "unchanged").
        finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
        n_finite = int(finite.sum())
        n_rows_non_finite += len(y) - n_finite

        residual = np.full(len(y), np.nan)
        orthogonalized = True

        if n_finite < min_rows:
            dt = chunk["date"][0]
            logger.warning(
                "orthogonalize: date %s has %d finite rows (< %d required for "
                "%d base factors + intercept), keeping original values",
                dt,
                n_finite,
                min_rows,
                len(base_cols),
            )
            residual = y
            orthogonalized = False
            n_dates_skipped += 1
        else:
            X_fit = X_with_intercept[finite]
            y_fit = y[finite]
            # WHY: handle collinearity or all-zero columns (e.g. a sector with
            # no observations on that date).
            try:
                beta, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
                residual_fit = y_fit - X_fit @ beta
                residual[finite] = residual_fit
                all_betas.append(beta[1:])  # exclude intercept
                ss_res = float(np.dot(residual_fit, residual_fit))
                centered = y_fit - np.mean(y_fit)
                ss_tot = float(np.dot(centered, centered))
                all_r2.append(1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0)
                n_rows_orthogonalized += n_finite
            except np.linalg.LinAlgError:
                dt = chunk["date"][0]
                logger.warning(
                    "orthogonalize: lstsq failed for date %s, keeping original", dt
                )
                residual = y
                orthogonalized = False
                n_dates_skipped += 1

        residuals_list.append(
            chunk.select("date", "asset_id").with_columns(
                # WHY: NaN → null so the residual column carries the library's
                # single "missing" marker; the `_orthogonalized` flag keeps the
                # "date was skipped" case distinguishable from "row undefined".
                pl.Series(name="_residual", values=residual).fill_nan(None),
                pl.lit(orthogonalized).alias("_orthogonalized"),
            )
        )

    if not residuals_list:
        logger.warning("orthogonalize_factor: no valid dates after join")
        return OrthogonalizeResult(data=factor_df)

    residuals_df = pl.concat(residuals_list)

    # WHY: keep the pre-orthogonalisation values for comparison analysis.
    # Rows on a regressed date take the residual (null when the row itself was
    # non-finite); rows on a skipped date, or missing from the join entirely,
    # keep the original value.
    result = (
        factor_df.with_columns(pl.col(factor_col).alias("factor_pre_ortho"))
        .join(residuals_df, on=["date", "asset_id"], how="left")
        .with_columns(
            pl.when(pl.col("_orthogonalized").fill_null(value=False))
            .then(pl.col("_residual"))
            .otherwise(pl.col(factor_col))
            .alias(factor_col)
        )
        .drop("_residual", "_orthogonalized")
    )

    n_total = len(factor_df)
    n_ortho = n_rows_orthogonalized
    n_base = len(base_cols)
    drop_pct = (n_total - n_ortho) / max(n_total, 1) * 100

    if n_rows_non_finite:
        logger.warning(
            "orthogonalize_factor: %d rows had null / non-finite factor or base "
            "values; excluded from the per-date regression (residual is null "
            "for those rows)",
            n_rows_non_finite,
        )

    if drop_pct > 5:
        logger.warning(
            "orthogonalize_factor: %.1f%% of rows (%d/%d) not orthogonalized "
            "(base factor coverage gap, skipped dates, or non-finite rows — "
            "original values kept except for non-finite rows, which are null)",
            drop_pct,
            n_total - n_ortho,
            n_total,
        )

    logger.info(
        "orthogonalize_factor: processed %d dates (%d skipped), %d base factors, "
        "%.1f%% coverage",
        n_dates,
        n_dates_skipped,
        n_base,
        100 - drop_pct,
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
        data=result,
        mean_betas=mean_betas,
        mean_r_squared=mean_r2,
        n_dates=n_dates,
        coverage=(n_ortho / n_total) if n_total else 0.0,
        n_base=n_base,
        n_rows_non_finite=n_rows_non_finite,
        n_dates_skipped=n_dates_skipped,
    )
