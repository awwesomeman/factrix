"""
Layer 1: Data Engine — Pure Polars preprocessing.
All functions are stateless: DataFrame in → DataFrame out.
"""

import polars as pl

# WHY: 1.4826 = 1/Φ⁻¹(0.75)，使 MAD 成為常態分佈下 σ 的無偏估計量
_MAD_CONSISTENCY_CONSTANT: float = 1.4826


def prepare_factor_data(
    df: pl.DataFrame,
    date_col: str = "datetime",
    asset_col: str = "ticker",
    factor_col: str = "factor",
    price_col: str = "close",
    forward_periods: int = 5,
    return_clip_pct: tuple[float, float] = (0.01, 0.99),
    mad_n: float = 3.0,
) -> pl.DataFrame:
    """
    Full preprocessing pipeline:
    1. Forward return computation.
    2. Forward return cross-sectional percentile winsorization.
    3. Abnormal return computation (cross-sectional market mean).
    4. Factor value cross-sectional MAD winsorization.
    5. Cross-sectional MAD-based z-score normalization.

    Args:
        return_clip_pct: (lower, upper) percentile bounds for forward return
            winsorization. Set to (0.0, 1.0) to disable.
        mad_n: Number of MAD units for factor value winsorization.
            Set to 0 to disable.
    """
    out = compute_forward_return(df, date_col, asset_col, price_col, forward_periods)
    out = winsorize_forward_return(out, date_col, lower=return_clip_pct[0], upper=return_clip_pct[1])
    out = compute_abnormal_return(out, date_col)
    
    # Preserve raw factor before any modification
    out = out.with_columns(pl.col(factor_col).alias("factor_raw"))
    
    out = mad_winsorize(out, date_col, factor_col, n_mad=mad_n)
    out = cross_sectional_zscore(out, date_col, factor_col)

    # WHY: 強制輸出為 Datetime("ms") 確保 Pandera schema 無論上游時間精度為何都能通過
    return out.select(
        pl.col(date_col).cast(pl.Datetime("ms")).alias("date"),
        pl.col(asset_col).alias("asset_id"),
        pl.col("factor_raw"),
        pl.col("factor_zscore").alias("factor"),
        pl.col("forward_return"),
        pl.col("abnormal_return"),
    )


def compute_abnormal_return(
    df: pl.DataFrame,
    date_col: str,
) -> pl.DataFrame:
    """Computes cross-sectional abnormal return (forward_return - market_mean)."""
    return df.with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over(date_col))
        .alias("abnormal_return")
    )


def compute_forward_return(
    df: pl.DataFrame,
    date_col: str,
    asset_col: str,
    price_col: str,
    forward_periods: int,
) -> pl.DataFrame:
    return (
        df.sort([asset_col, date_col])
        .with_columns(
            (
                pl.col(price_col).shift(-forward_periods).over(asset_col)
                / pl.col(price_col)
                - 1
            ).alias("forward_return")
        )
        .filter(pl.col("forward_return").is_not_null())
    )


def winsorize_forward_return(
    df: pl.DataFrame,
    date_col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Per-date percentile clip on forward returns.

    Caps extreme return observations at the (lower, upper) quantiles within
    each cross-section to reduce the influence of outliers on downstream
    aggregations that use ``.mean()`` (e.g. Long_Alpha, MDD, NAV).

    # WHY: .mean() 彙總對極端收益率脆弱，percentile clip 是業界標準前處理
    """
    if lower <= 0.0 and upper >= 1.0:
        return df

    lb = pl.col("forward_return").quantile(lower).over(date_col)
    ub = pl.col("forward_return").quantile(upper).over(date_col)

    return df.with_columns(
        pl.col("forward_return").clip(lb, ub).alias("forward_return")
    )


def mad_winsorize(
    df: pl.DataFrame,
    date_col: str,
    factor_col: str,
    n_mad: float = 3.0,
) -> pl.DataFrame:
    """Per-date MAD-based winsorization on factor values.

    Clips factor values to [median - n × MAD × 1.4826, median + n × MAD × 1.4826]
    within each cross-section. Applied before z-score so that extreme factor
    values do not distort the standardization.

    # WHY: 在 z-score 之前先截斷，避免極端因子值拉偏 z-score 分佈
    """
    if n_mad <= 0:
        return df

    median_expr = pl.col(factor_col).median().over(date_col)
    deviation = (pl.col(factor_col) - median_expr).abs()
    mad_expr = deviation.median().over(date_col)
    half_width = mad_expr * _MAD_CONSISTENCY_CONSTANT * n_mad

    return df.with_columns(
        pl.col(factor_col)
        .clip(median_expr - half_width, median_expr + half_width)
        .alias(factor_col)
    )


def cross_sectional_zscore(
    df: pl.DataFrame,
    date_col: str,
    factor_col: str,
) -> pl.DataFrame:
    """MAD-robust z-score within each cross-section (date).

    Uses median for centering and MAD (Median Absolute Deviation) for scaling,
    both resistant to outliers. The consistency constant 1.4826 makes MAD an
    unbiased estimator of std for normally distributed data.

    z = (x - median(x)) / (1.4826 * MAD(x))
    """
    median_expr = pl.col(factor_col).median().over(date_col)
    deviation = (pl.col(factor_col) - median_expr).abs()
    mad_expr = deviation.median().over(date_col)

    return df.with_columns(
        (
            (pl.col(factor_col) - median_expr)
            / (mad_expr * _MAD_CONSISTENCY_CONSTANT)
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("factor_zscore")
    )
