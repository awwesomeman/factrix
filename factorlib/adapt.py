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
"""

from __future__ import annotations

import polars as pl


def adapt(
    df: pl.DataFrame,
    *,
    date: str = "date",
    asset_id: str = "asset_id",
    price: str = "price",
) -> pl.DataFrame:
    """Rename user columns to factorlib canonical names.

    Only renames columns that differ from the canonical name.
    All other columns are passed through unchanged.

    Args:
        df: Input DataFrame.
        date: User's date column name.
        asset_id: User's asset identifier column name.
        price: User's price column name.

    Returns:
        DataFrame with canonical column names.

    Raises:
        ValueError: If any specified source column does not exist.
    """
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

    return df.rename(mapping) if mapping else df
