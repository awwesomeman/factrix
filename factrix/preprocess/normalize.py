"""Preprocessing Step 4-5: factor value normalization.

Step 4 — MAD Winsorize: per-date robust outlier clipping
Step 5 — Cross-sectional Z-score: MAD-based robust standardization

All functions expect canonical column names (date, asset_id).
"""

import polars as pl

from factrix._types import EPSILON, MAD_CONSISTENCY_CONSTANT


def _finite(factor_col: str) -> pl.Expr:
    """``factor_col`` with non-finite entries (NaN / ±Inf) blanked to null.

    polars aggregations skip nulls but *propagate* float NaN, so a single
    NaN on a date would otherwise poison that date's median / MAD / std and
    silently take the whole cross-section out of the pipeline. Blanking
    non-finite values to null makes the per-date statistics ignore them, in
    line with the library convention (consumers use
    ``drop_nulls().drop_nans()``; producers never impute).
    """
    col = pl.col(factor_col)
    return pl.when(col.is_finite()).then(col).otherwise(None)


def _scale_expressions(factor_col: str) -> tuple[pl.Expr, pl.Expr]:
    """Per-date centre and dispersion expressions for a factor column.

    Returns:
        ``(median_expr, scale_expr)`` — both per-date window expressions.
        ``scale_expr`` is ``1.4826 × MAD`` when the MAD is non-degenerate,
        the per-date sample standard deviation (``ddof=1``) when the MAD
        collapses to zero, and ``0.0`` when the cross-section carries no
        dispersion at all.
    """
    clean = _finite(factor_col)
    median_expr = clean.median().over("date")
    mad_expr = (clean - median_expr).abs().median().over("date")
    std_expr = clean.std(ddof=1).over("date")

    scale_expr = (
        pl.when(mad_expr > EPSILON)
        .then(mad_expr * MAD_CONSISTENCY_CONSTANT)
        .when(std_expr > EPSILON)
        .then(std_expr)
        .otherwise(pl.lit(0.0))
    )
    return median_expr, scale_expr


def mad_winsorize(
    data: pl.DataFrame,
    factor_col: str = "factor",
    n_mad: float = 3.0,
) -> pl.DataFrame:
    """Step 4: Per-date MAD-based winsorization on factor values.

    Clips factor values to ``[median ± n_mad × 1.4826 × MAD]`` within each
    cross-section.

    Args:
        n_mad: Number of MAD units for clipping (default 3.0).
            Set to 0 to disable.

    Returns:
        DataFrame with ``factor_col`` clipped in-place.

    Notes:
        **Zero-MAD fallback.** More than 50% ties on a date (bucketed,
        binary or heavily discretised factors are the common case) drive
        the MAD to exactly 0, which collapses the clip interval to
        ``[median, median]`` and flattens the entire cross-section to its
        median — the factor is destroyed rather than winsorised. When the
        MAD is 0 we fall back to the per-date sample standard deviation
        (``ddof=1``) as the scale; the MAD branch already carries the
        1.4826 consistency constant precisely so the two scales are
        comparable at the Gaussian. Mainstream robust-scale implementations
        (statsmodels ``mad``, scipy ``median_abs_deviation``) leave the
        zero-MAD case to the caller — the alternatives are an IQR fallback
        (still zero for a two-bucket factor) or returning NaN (drops the
        date). We prefer the std fallback because it keeps a bucketed
        factor finite and rank-preserving. A cross-section with no
        dispersion at all (every value identical) has ``scale = 0`` and is
        left untouched by the clip.

        **Non-finite input.** NaN / ±Inf values are excluded from the
        per-date median / MAD / std (polars aggregations do not skip float
        NaN on their own), so one bad tick no longer voids the date. They
        are still clipped like any other value.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import mad_winsorize
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> clipped = mad_winsorize(raw, n_mad=3.0)
        >>> clipped.columns == raw.columns
        True
        >>> clipped.height == raw.height
        True
    """
    if n_mad <= 0:
        return data

    median_expr, scale_expr = _scale_expressions(factor_col)
    half_width = scale_expr * n_mad

    return data.with_columns(
        pl.when(scale_expr > EPSILON)
        .then(
            pl.col(factor_col).clip(median_expr - half_width, median_expr + half_width)
        )
        .otherwise(pl.col(factor_col))
        .alias(factor_col)
    )


def cross_sectional_zscore(
    data: pl.DataFrame,
    factor_col: str = "factor",
) -> pl.DataFrame:
    """Step 5: MAD-robust z-score within each cross-section (date).

    ``z = (x - median(x)) / (1.4826 × MAD(x))``

    Returns:
        DataFrame with ``factor_zscore`` column appended.

    Notes:
        **Zero-MAD fallback.** With more than 50% ties on a date the MAD is
        exactly 0, so the naive ratio is ``±inf`` for every non-median value
        (and ``0/0 = NaN`` at the median). We fall back to the per-date
        sample standard deviation (``ddof=1``) as the scale, matching
        :func:`mad_winsorize`. The alternative conventions are to return NaN
        for the date (drops bucketed factors entirely) or to fall back to a
        scaled IQR (still zero for a two-bucket factor); the std fallback
        keeps the z finite and rank-preserving, which is what the downstream
        rank-based metrics need. When the cross-section is fully constant
        there is genuinely no dispersion: every non-null value gets ``0.0``,
        an honest "no spread", rather than ``inf`` or NaN.

        **Nulls stay null.** The previous implementation ended with
        ``fill_nan(0.0).fill_null(0.0)``, which imputed every missing factor
        to *exactly the cross-sectional median* — a silent, maximally
        "average" fabrication that inflated coverage. Per the library
        convention (producers never impute; consumers drop with
        ``drop_nulls().drop_nans()``), a null input now yields a null
        z-score, and so does a non-finite (NaN / ±Inf) input.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import cross_sectional_zscore
        >>> raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
        >>> standardized = cross_sectional_zscore(raw)
        >>> "factor_zscore" in standardized.columns
        True
    """
    clean = _finite(factor_col)
    median_expr, scale_expr = _scale_expressions(factor_col)

    zscore = (
        pl.when(clean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(scale_expr > EPSILON)
        .then((clean - median_expr) / scale_expr)
        .otherwise(pl.lit(0.0))
    )

    return data.with_columns(zscore.alias("factor_zscore"))
