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
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factrix._codes import WarningCode
from factrix._types import EPSILON, MIN_ORTHOGONALIZE_RESIDUAL_ASSETS

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalizeResult:
    """Result of factor orthogonalization with attribution info.

    Attributes:
        data: ``factor_df`` with ``factor_col`` replaced by the residual and
            ``factor_pre_ortho`` preserving the original value.
        mean_betas: Average beta per base factor across regressed dates whose
            design matrix had full column rank. Empty when every regressed
            date was rank deficient — the betas are not identified there, so
            reporting an average of arbitrary minimum-norm solutions would be
            worse than reporting nothing (see ``n_dates_rank_deficient``).
        mean_r_squared: Average raw R² across regressed dates.
        mean_adj_r_squared: Average degrees-of-freedom-adjusted R²,
            ``1 - (1 - R²)(N - 1)/(N - K - 1)``. Raw R² is mechanically
            ≈ ``K/(N - 1)`` even when the true R² is 0, so on a thin
            cross-section the adjusted figure is the one that answers "is the
            factor actually spanned by the base set?".
        n_dates: Number of per-date cross-sections seen after the join.
        coverage: Fraction of input rows that carry an actual residual —
            rows dropped by the join, rows on a skipped date, and rows with
            non-finite inputs are all excluded.
        n_base: Number of base factor columns regressed on.
        n_rows_non_finite: Rows whose factor or base values were null / NaN /
            ±Inf on an otherwise regressed date. These rows are excluded from
            the regression and their residual is **null** in ``data``.
        n_dates_skipped: Dates left un-orthogonalised, for any reason.
            Original factor values are kept for those dates.
        n_dates_insufficient_df: Subset of ``n_dates_skipped`` skipped because
            the cross-section left fewer than ``min_residual_df`` residual
            degrees of freedom.
        n_dates_rank_deficient: Dates that were regressed but whose design
            matrix was rank deficient. The residual is still exact (the
            projection is unique) but the betas are not identified, so those
            dates are excluded from ``mean_betas``.
        restandardized: Whether the residual was rescaled to the per-date
            dispersion of the input factor.
        warning_codes: :class:`~factrix._codes.WarningCode` values raised by
            this run, as strings — the structured twin of the ``UserWarning``
            echoes, for callers that inspect the result rather than trap
            warnings.
    """

    data: pl.DataFrame
    mean_betas: dict[str, float] = field(default_factory=dict)
    mean_r_squared: float = 0.0
    mean_adj_r_squared: float = 0.0
    n_dates: int = 0
    coverage: float = 0.0
    n_base: int = 0
    n_rows_non_finite: int = 0
    n_dates_skipped: int = 0
    n_dates_insufficient_df: int = 0
    n_dates_rank_deficient: int = 0
    restandardized: bool = False
    warning_codes: tuple[str, ...] = ()


def _is_rank_deficient(design: np.ndarray, singular_values: np.ndarray) -> bool:
    """Whether ``design`` is (near-)rank deficient, from ``lstsq``'s own SVD.

    ``np.linalg.lstsq`` returns the singular values it already computed, so
    the rank test costs nothing extra. The tolerance is ``np.linalg.matrix_rank``'s
    default — ``max(M, N) * eps * s_max`` — deliberately looser than lstsq's own
    ``rcond=None`` machine-precision cutoff, which reports full rank for a
    *near*-collinear design and hands back huge unstable betas instead.
    """
    if singular_values.size == 0:
        return design.shape[1] > 0
    tol = max(design.shape) * float(np.finfo(float).eps) * float(singular_values[0])
    return int(np.count_nonzero(singular_values > tol)) < design.shape[1]


def orthogonalize_factor(
    factor_df: pl.DataFrame,
    base_factors: pl.DataFrame,
    factor_col: str = "factor",
    base_cols: list[str] | None = None,
    *,
    min_residual_df: int = MIN_ORTHOGONALIZE_RESIDUAL_ASSETS,
    restandardize: bool = False,
) -> OrthogonalizeResult:
    """Orthogonalize factor against base factors via per-date ordinary least squares (OLS).

    Args:
        factor_df: Panel with ``date, asset_id, {factor_col}``.
            factor_col should already be z-scored (Step 5 output).
        base_factors: Panel with ``date, asset_id`` and base factor columns.
            Industry dummies must be pre-encoded as 0/1 columns **with one
            category dropped as the reference level**: an intercept is always
            prepended, so a full dummy set is exactly singular (the classic
            dummy trap). See the rank-deficiency note below for what happens
            if it is not.
        factor_col: Column name of the factor to orthogonalize.
        base_cols: List of column names in ``base_factors`` to regress on.
            If None, uses all columns except ``date`` and ``asset_id``.
        min_residual_df: Minimum residual degrees of freedom
            (``n_finite - len(base_cols) - 1``) a date must leave to be
            regressed. Defaults to
            :data:`~factrix._types.MIN_ORTHOGONALIZE_RESIDUAL_ASSETS` (10), the
            ``N >= K + 10`` form of the Fama-MacBeth convention; pass
            ``4 * len(base_cols)`` for the ``N >= 5K`` form, or ``1`` to
            restore the old behaviour of fitting anything that is
            arithmetically solvable.
        restandardize: Rescale the residual to the per-date dispersion of the
            input factor. ``False`` (default) returns the raw residual, whose
            scale is ``sqrt(1 - R²)`` times the input — and since R² varies by
            date, so does the output scale. Rank metrics are unaffected; any
            magnitude-based use (weights ``w ~ f``, a spread in factor units)
            is otherwise quietly running on a time-varying scale.

    Returns:
        OrthogonalizeResult with: ``data`` (factor_df with ``factor_col``
        replaced by the residual and ``factor_pre_ortho`` preserving the
        original value), ``mean_betas`` (average beta per base factor
        across full-rank dates), ``mean_r_squared`` and
        ``mean_adj_r_squared``, plus the per-reason skip counts and
        ``warning_codes``.

    Raises:
        ValueError: ``base_factors`` carries duplicate ``(date, asset_id)``
            keys, which would fan the inner join out and duplicate rows.

    Notes:
        **Minimum residual degrees of freedom.** A cross-sectional OLS needs
        residual df to mean anything. The old floor was ``len(base_cols) + 2``
        rows — a single residual df — and raw R² is mechanically ≈
        ``K/(N - 1)`` even when the true R² is zero, so a regional book of six
        names regressed on size / value / momentum plus an industry dummy
        passed the guard, fitted noise, and came back with
        ``mean_r_squared = 0.79`` (adjusted: ``-0.03``) after removing 83% of
        the factor's variance — the user concludes the factor is redundant
        when it is orthogonal by construction. Dates below ``min_residual_df``
        are now skipped, counted in ``n_dates_insufficient_df``, and flagged
        with ``WarningCode.INSUFFICIENT_REGRESSION_DF``.
        ``mean_adj_r_squared`` is reported alongside the raw figure for the
        same reason.

        **Rank deficiency.** ``np.linalg.lstsq`` does **not** raise on a
        rank-deficient design — it returns the minimum-norm solution — so the
        ``except LinAlgError`` branch never fired for the dummy trap. The
        residual stays correct either way (the projection onto the column
        space is unique), but ``mean_betas`` was an arbitrary point in the
        solution space, reported without qualification: on a 4-dummy panel
        the full set gave ``{ind0: 0.008, ind1: -0.007, ...}`` and dropping one
        category gave ``{ind1: -0.015, ...}``, with identical residuals.
        Rank is now read from the singular values ``lstsq`` already computes
        (the ``matrix_rank`` tolerance, so *near*-collinear designs are caught
        too, not only exactly singular ones); deficient dates are counted in
        ``n_dates_rank_deficient``, excluded from ``mean_betas``, and flagged
        with ``WarningCode.RANK_DEFICIENT_DESIGN``.

        **Residual scale.** See ``restandardize`` above; the default leaves
        the residual on its own per-date scale rather than the input's.

        **Non-finite rows.** A single null / NaN / ±Inf in ``factor_col`` or
        in any base column used to make ``np.linalg.lstsq`` return all-NaN
        betas, which turned *every* asset on that date into NaN while the
        date still counted as orthogonalised. The regression now runs on the
        finite rows only. Rows that were not finite come back **null**, not
        with their original value: the residual is undefined for them, and
        keeping the raw value would mix two scales (raw factor and residual)
        inside one column — a silent, unrecoverable contamination. Their
        count is reported as ``n_rows_non_finite`` and logged.

        **Skipped dates.** A date that leaves fewer than ``min_residual_df``
        residual degrees of freedom (or one where ``lstsq`` raises) cannot
        support the regression; the original values are kept for that date and
        it is counted in ``n_dates_skipped``. Such dates are excluded from
        ``coverage``, ``mean_betas`` and both R² figures.

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
    all_adj_r2: list[float] = []
    n_rows_non_finite = 0
    n_dates_skipped = 0
    n_dates_insufficient_df = 0
    n_dates_rank_deficient = 0
    n_rows_orthogonalized = 0
    n_params = len(base_cols) + 1  # base columns + the prepended intercept
    # WHY: residual df, not a bare row count. ``len(base_cols) + 2`` rows left
    # exactly one residual df, where raw R2 is ~K/(N-1) at a true R2 of 0.
    min_rows = n_params + max(int(min_residual_df), 1)

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
                "orthogonalize: date %s has %d finite rows, leaving %d residual "
                "df for %d base factors + intercept (min_residual_df=%d); "
                "keeping original values",
                dt,
                n_finite,
                max(n_finite - n_params, 0),
                len(base_cols),
                min_residual_df,
            )
            residual = y
            orthogonalized = False
            n_dates_skipped += 1
            n_dates_insufficient_df += 1
        else:
            X_fit = X_with_intercept[finite]
            y_fit = y[finite]
            # WHY: handle collinearity or all-zero columns (e.g. a sector with
            # no observations on that date).
            try:
                beta, _, _, sv = np.linalg.lstsq(X_fit, y_fit, rcond=None)
                residual_fit = y_fit - X_fit @ beta
                residual[finite] = residual_fit
                # WHY: lstsq does not raise on a rank-deficient design — it
                # returns the minimum-norm solution — so the residual is exact
                # but the betas are an arbitrary point in the solution space
                # (the dummy trap). Read the rank off the singular values
                # lstsq already computed, at the ``matrix_rank`` tolerance so
                # near-collinear designs are caught too, and keep those betas
                # out of the attribution average rather than reporting them.
                if _is_rank_deficient(X_fit, sv):
                    n_dates_rank_deficient += 1
                else:
                    all_betas.append(beta[1:])  # exclude intercept
                ss_res = float(np.dot(residual_fit, residual_fit))
                centered = y_fit - np.mean(y_fit)
                ss_tot = float(np.dot(centered, centered))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0
                all_r2.append(r2)
                # Raw R2 is mechanically ~K/(N-1) even at a true R2 of 0; the
                # df-adjusted figure is the one that answers "is the factor
                # spanned?" on a thin cross-section.
                df_resid = n_finite - n_params
                all_adj_r2.append(
                    1.0 - (1.0 - r2) * (n_finite - 1) / df_resid
                    if df_resid > 0
                    else float("nan")
                )
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

    if restandardize:
        # The raw residual's scale is sqrt(1 - R2) times the input's, and R2
        # varies by date, so the output scale varies by date too. Rescale each
        # regressed date's residual back to the pre-orthogonalisation
        # dispersion; skipped dates already carry the original values.
        pre_std = pl.col("factor_pre_ortho").std(ddof=1).over("date")
        post_std = pl.col(factor_col).std(ddof=1).over("date")
        result = result.with_columns(
            pl.when(post_std > EPSILON)
            .then(pl.col(factor_col) * pre_std / post_std)
            .otherwise(pl.col(factor_col))
            .alias(factor_col)
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

    # Attribution: average betas across full-rank dates, R² across all fitted
    # ones. A rank-deficient date still contributes a valid residual and R²;
    # only its betas are unidentified.
    mean_betas: dict[str, float] = {}
    mean_r2 = 0.0
    mean_adj_r2 = 0.0
    if all_betas:
        beta_matrix = np.array(all_betas)
        for i, col in enumerate(base_cols):
            mean_betas[col] = float(np.mean(beta_matrix[:, i]))
    if all_r2:
        mean_r2 = float(np.mean(all_r2))
        mean_adj_r2 = float(np.nanmean(all_adj_r2))

    warning_codes: list[str] = []
    if n_dates_insufficient_df:
        warning_codes.append(WarningCode.INSUFFICIENT_REGRESSION_DF.value)
        warnings.warn(
            f"orthogonalize_factor: "
            f"{WarningCode.INSUFFICIENT_REGRESSION_DF.value} — "
            f"{n_dates_insufficient_df}/{n_dates} dates left fewer than "
            f"{min_residual_df} residual degrees of freedom for {n_base} base "
            "factors + intercept and were skipped (original values kept). Raw "
            "R2 is mechanically ~K/(N-1) at a true R2 of 0, so fitting there "
            "reports noise as explanatory power while stripping most of the "
            "factor's variance. Reduce base_cols, or lower min_residual_df if "
            "the thin fit is wanted anyway.",
            UserWarning,
            stacklevel=2,
        )
    if n_dates_rank_deficient:
        warning_codes.append(WarningCode.RANK_DEFICIENT_DESIGN.value)
        warnings.warn(
            f"orthogonalize_factor: {WarningCode.RANK_DEFICIENT_DESIGN.value} "
            f"— {n_dates_rank_deficient}/{n_dates} dates had a rank-deficient "
            "design matrix. The residual is still exact (the projection is "
            "unique) but the betas are not identified there, so those dates "
            "are excluded from mean_betas. The usual cause is a full industry "
            "dummy set alongside the always-prepended intercept — drop one "
            "category as the reference level.",
            UserWarning,
            stacklevel=2,
        )

    return OrthogonalizeResult(
        data=result,
        mean_betas=mean_betas,
        mean_r_squared=mean_r2,
        mean_adj_r_squared=mean_adj_r2,
        n_dates=n_dates,
        coverage=(n_ortho / n_total) if n_total else 0.0,
        n_base=n_base,
        n_rows_non_finite=n_rows_non_finite,
        n_dates_skipped=n_dates_skipped,
        n_dates_insufficient_df=n_dates_insufficient_df,
        n_dates_rank_deficient=n_dates_rank_deficient,
        restandardized=restandardize,
        warning_codes=tuple(warning_codes),
    )
