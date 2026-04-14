"""Preprocessing Step 4-5: factor value normalization.

Step 4 — MAD Winsorize: per-date robust outlier clipping
Step 5 — Cross-sectional Z-score: MAD-based robust standardization

All functions are stateless: DataFrame in → DataFrame out.
Each function is independently importable.
"""

import polars as pl

from factorlib.tools._typing import MAD_CONSISTENCY_CONSTANT


def _mad_expressions(
    factor_col: str,
    date_col: str,
) -> tuple[pl.Expr, pl.Expr]:
    """Compute median and MAD expressions for a factor column.

    Returns:
        (median_expr, mad_expr) — both are per-date window expressions.
    """
    median_expr = pl.col(factor_col).median().over(date_col)
    deviation = (pl.col(factor_col) - median_expr).abs()
    mad_expr = deviation.median().over(date_col)
    return median_expr, mad_expr


def mad_winsorize(
    df: pl.DataFrame,
    date_col: str = "datetime",
    factor_col: str = "factor",
    n_mad: float = 3.0,
) -> pl.DataFrame:
    """Step 4: Per-date MAD-based winsorization on factor values.

    Clips factor values to ``[median ± n_mad × 1.4826 × MAD]`` within each
    cross-section.  Applied before z-score so that extreme factor values
    do not distort the standardization.

    Args:
        n_mad: Number of MAD units for clipping (default 3.0).
            Set to 0 to disable.

    Returns:
        DataFrame with ``factor_col`` clipped in-place.
    """
    if n_mad <= 0:
        return df

    median_expr, mad_expr = _mad_expressions(factor_col, date_col)
    half_width = mad_expr * MAD_CONSISTENCY_CONSTANT * n_mad

    return df.with_columns(
        pl.col(factor_col)
        .clip(median_expr - half_width, median_expr + half_width)
        .alias(factor_col)
    )


def cross_sectional_zscore(
    df: pl.DataFrame,
    date_col: str = "datetime",
    factor_col: str = "factor",
) -> pl.DataFrame:
    """Step 5: MAD-robust z-score within each cross-section (date).

    ``z = (x - median(x)) / (1.4826 × MAD(x))``

    Uses median for centering and MAD for scaling — both resistant to
    outliers.  The consistency constant 1.4826 makes MAD an unbiased
    estimator of σ for normally distributed data.

    Returns:
        DataFrame with ``factor_zscore`` column appended.
    """
    median_expr, mad_expr = _mad_expressions(factor_col, date_col)

    return df.with_columns(
        (
            (pl.col(factor_col) - median_expr)
            / (mad_expr * MAD_CONSISTENCY_CONSTANT)
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("factor_zscore")
    )
