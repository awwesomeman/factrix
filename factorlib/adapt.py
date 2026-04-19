"""Column name adapter — maps user column names to factorlib canonical names.

Canonical names used throughout factorlib:
    - ``date``: trading date
    - ``asset_id``: asset identifier (ticker, permno, symbol, etc.)
    - ``price``: price column (close, adj close, VWAP, etc.)

Optional OHLCV canonicals (renamed when a source column is provided):
    ``open``, ``high``, ``low``, ``volume`` — required by some factor
    generators (``factors.technical``, ``factors.liquidity``).

Other columns (market_cap, industry, etc.) pass through unchanged;
factorlib does not prescribe names for those.

Usage::

    from factorlib import adapt

    # Minimal: just price panel
    raw = adapt(df, date="date", asset_id="ticker", price="close_adj")

    # Full OHLCV — unlocks technical / liquidity factor generators
    raw = adapt(
        df,
        date="date", asset_id="ticker", price="close_adj",
        open="open_adj", high="high_adj", low="low_adj", volume="volume",
    )
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
    open: str | None = None,
    high: str | None = None,
    low: str | None = None,
    volume: str | None = None,
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
        open: User's open column name. If set, renamed to ``open``.
            Required by ``factors.technical.generate_overnight_return``.
        high: User's high column name. Renamed to ``high``. Required
            by ``generate_52w_high_ratio`` / ``generate_intraday_range``.
        low: User's low column name. Renamed to ``low``. Required by
            ``generate_intraday_range``.
        volume: User's volume column name. Renamed to ``volume``.
            Required by ``generate_amihud`` / ``generate_volume_price_trend``.
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

    renames: list[tuple[str, str | None]] = [
        ("date", date),
        ("asset_id", asset_id),
        ("price", price),
        ("open", open),
        ("high", high),
        ("low", low),
        ("volume", volume),
    ]
    mapping = {}
    for canonical, source in renames:
        if source is None or source == canonical:
            continue
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

    # Promote pl.Date → pl.Datetime("ms") losslessly so downstream joins
    # (regime_labels / spanning_base_spreads / user panels) can share a
    # common datetime dtype without the user writing an explicit cast.
    # Other Datetime variants (any time_unit, any TZ) pass through — the
    # library is TZ-agnostic and trusts the caller's precision choice.
    if "date" in df.columns and df.schema["date"] == pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Datetime("ms")))

    if fill_forward:
        df = (
            df.sort(["asset_id", "date"])
            .with_columns(cs.numeric().fill_nan(None))
            .with_columns(cs.numeric().forward_fill().over("asset_id"))
        )

    return df
