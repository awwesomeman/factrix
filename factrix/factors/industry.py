"""Industry classification encoding for orthogonalization."""

import polars as pl


def encode_industry_dummies(
    df: pl.DataFrame,
    industry_col: str = "industry",
    drop_last: bool = True,
) -> pl.DataFrame:
    """One-hot encode industry classification into dummy columns.

    Output columns are named ``industry_{name}`` with values 0/1.
    Used as control variables in ``orthogonalize_factor``, not as
    a standalone evaluable factor.

    Args:
        df: Raw data with ``date``, ``asset_id``, and ``industry_col``.
        industry_col: Column containing industry labels (string).
        drop_last: Drop the alphabetically last category to avoid the
            dummy variable trap (perfect multicollinearity in OLS).
            Default True.

    Returns:
        Input DataFrame with ``industry_{name}`` columns appended.
        The original ``industry_col`` is preserved.
    """
    categories = df[industry_col].drop_nulls().unique().sort().to_list()

    if drop_last and len(categories) > 1:
        categories = categories[:-1]

    return df.with_columns([
        (pl.col(industry_col) == cat).cast(pl.Int8).alias(f"industry_{cat}")
        for cat in categories
    ])
