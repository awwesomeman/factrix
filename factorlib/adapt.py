"""Column name adapter — maps user column names to factorlib canonical names.

Canonical names used throughout factorlib:
    - ``date``: trading date
    - ``asset_id``: asset identifier (ticker, permno, symbol, etc.)
    - ``price``: price column (close, adj close, VWAP, etc.)

Optional columns (not renamed — user keeps their own names):
    market, market_cap, volume, industry, etc.

Usage::

    from factorlib import adapt

    raw = adapt(
        pl.read_parquet("data.parquet"),
        date="date",
        asset_id="ticker",
        price="close_adj",
    )

    # pandas DataFrame also accepted — automatically converted to polars
    raw = adapt(pd_df, date="Date", asset_id="symbol", price="close")

    # forward-fill NaN/null in numeric columns per asset
    raw = adapt(df, price="close_adj", fill_forward=True)
"""

from __future__ import annotations

import polars as pl
import polars.selectors as cs


def _to_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to polars if needed; pass through polars as-is."""
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except ImportError:
        pass
    if isinstance(df, pl.DataFrame):
        return df
    raise TypeError(
        f"adapt: expected polars or pandas DataFrame, got {type(df).__name__}"
    )


def adapt(
    df: pl.DataFrame,
    *,
    date: str = "date",
    asset_id: str = "asset_id",
    price: str = "close",
    fill_forward: bool = False,
) -> pl.DataFrame:
    """Rename user columns to factorlib canonical names.

    Accepts both polars and pandas DataFrames (pandas is converted
    to polars automatically).  Only renames columns that differ from
    the canonical name.  All other columns are passed through unchanged.

    Args:
        df: Input DataFrame (polars or pandas).
        date: User's date column name.
        asset_id: User's asset identifier column name.
        price: User's price column name.
        fill_forward: If True, replace NaN with null then forward-fill
            all numeric columns per asset.  Useful for raw OHLCV data
            that may contain sporadic missing values.

    Returns:
        Polars DataFrame with canonical column names.

    Raises:
        TypeError: If *df* is neither polars nor pandas DataFrame.
        ValueError: If any specified source column does not exist.
    """
    df = _to_polars(df)

    mapping = {}
    for canonical, source in [("date", date), ("asset_id", asset_id), ("price", price)]:
        if source != canonical:
            if source not in df.columns:
                raise ValueError(
                    f"adapt: column '{source}' not found. "
                    f"Available: {df.columns}"
                )
            if canonical in df.columns:
                raise ValueError(
                    f"adapt: cannot rename '{source}' → '{canonical}' "
                    f"because '{canonical}' already exists in the DataFrame. "
                    f"Drop or rename the existing '{canonical}' column first."
                )
            mapping[source] = canonical

    if mapping:
        df = df.rename(mapping)

    if fill_forward:
        df = (
            df.sort(["asset_id", "date"])
            .with_columns(cs.numeric().fill_nan(None))
            .with_columns(cs.numeric().forward_fill().over("asset_id"))
        )

    return df
