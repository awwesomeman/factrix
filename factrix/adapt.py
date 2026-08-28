"""Column name adapter — maps user column names to factrix canonical names.

Canonical names used throughout factrix:
    - ``date``: trading date
    - ``asset_id``: asset identifier (ticker, permno, symbol, etc.)
    - ``price``: price column (close, adj close, VWAP, etc.)

Optional OHLCV canonicals (renamed when a source column is provided):
    ``open``, ``high``, ``low``, ``volume`` — nothing in factrix reads
    these; ``adapt`` canonicalises them so your own factor construction
    can rely on stable names.

Other columns (market_cap, industry, etc.) pass through unchanged;
factrix does not prescribe names for those.

Usage::

    from factrix.adapt import adapt

    # Minimal: just price panel
    raw = adapt(data, date="date", asset_id="ticker", price="close_adj")

    # Full OHLCV — canonical open/high/low/volume for your own factors
    raw = adapt(
        data,
        date="date", asset_id="ticker", price="close_adj",
        open="open_adj", high="high_adj", low="low_adj", volume="volume",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from factrix._data_input import _is_pandas_dataframe

if TYPE_CHECKING:
    import pandas as pd

type AdaptInput = pl.DataFrame | pl.LazyFrame | pd.DataFrame


def _to_polars(data: AdaptInput) -> pl.DataFrame | pl.LazyFrame:
    """Coerce ``adapt`` input to polars, preserving ``LazyFrame``.

    ``pl.DataFrame`` / ``pl.LazyFrame`` pass through unchanged.
    ``pd.DataFrame`` is converted via ``pl.from_pandas`` (pandas has
    no lazy equivalent).
    """
    if isinstance(data, pl.DataFrame | pl.LazyFrame):
        return data
    if _is_pandas_dataframe(data):
        return pl.from_pandas(data)
    raise TypeError(
        f"adapt: expected pl.DataFrame, pl.LazyFrame, or pd.DataFrame; got {type(data).__name__}"
    )


def adapt(
    data: AdaptInput,
    *,
    date: str = "date",
    asset_id: str = "asset_id",
    price: str = "close",
    open: str | None = None,
    high: str | None = None,
    low: str | None = None,
    volume: str | None = None,
    fill_forward: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Rename user columns to factrix canonical names.

    Type-preserving for polars inputs: a ``pl.LazyFrame`` stays lazy
    (rename / cast / fill happen inside the lazy chain, no implicit
    ``.collect()``), a ``pl.DataFrame`` stays eager. ``pd.DataFrame``
    is converted to ``pl.DataFrame`` (pandas has no lazy equivalent).
    Only renames columns that differ from the canonical name; all other
    columns pass through unchanged.

    Args:
        data: Input frame — ``pl.DataFrame``, ``pl.LazyFrame``, or
            ``pd.DataFrame``.
        date: User's date column name. Renamed to ``date`` — the panel's
            ordering and alignment key; every factrix horizon, window and
            lag counts periods on its distinct-date grid.
        asset_id: User's asset identifier column name. Renamed to
            ``asset_id`` — the cross-sectional key every metric groups by.
        price: User's price column name. Renamed to ``price`` — the level
            series ``compute_forward_return`` differences into
            ``forward_return``, and the volatility source the event-study
            metrics (``caar``, ``event_horizon``) prefer when present.
        open: User's opening-price column name. Renamed to ``open``.
            Not read by factrix; passed through for downstream use.
        high: User's period-high price column name. Renamed to ``high``.
            Not read by factrix; passed through for downstream use.
        low: User's period-low price column name. Renamed to ``low``.
            Not read by factrix; passed through for downstream use.
        volume: User's traded-volume column name. Renamed to ``volume``.
            Not read by factrix; passed through for downstream use.
        fill_forward: If True, map every non-finite value (NaN and
            ±inf) to null, then forward-fill all numeric columns per
            asset. Useful for raw OHLCV data that may contain sporadic
            missing or non-finite values. Mapping ±inf is deliberate:
            inf is not null, so it would otherwise survive the tail
            drop in ``compute_forward_return`` and leak into return math.

    Returns:
        Same polars type as input (``pl.DataFrame`` → ``pl.DataFrame``,
        ``pl.LazyFrame`` → ``pl.LazyFrame``) with canonical column
        names. ``pd.DataFrame`` input returns ``pl.DataFrame``.

    Raises:
        TypeError: If *data* is none of ``pl.DataFrame``, ``pl.LazyFrame``,
            ``pd.DataFrame``.
        ValueError: If any specified source column does not exist.
    """
    data = _to_polars(data)
    schema = data.collect_schema()
    columns = schema.names()

    renames: list[tuple[str, str | None]] = [
        ("date", date),
        ("asset_id", asset_id),
        ("price", price),
        ("open", open),
        ("high", high),
        ("low", low),
        ("volume", volume),
    ]
    mapping: dict[str, str] = {}
    for canonical, source in renames:
        if source is None or source == canonical:
            continue
        if source not in columns:
            raise ValueError(
                f"adapt: column '{source}' not found. Available: {columns}"
            )
        if canonical in columns:
            raise ValueError(
                f"adapt: cannot rename '{source}' → '{canonical}' because '{canonical}' already exists in the DataFrame. Drop or rename the existing '{canonical}' column first."
            )
        mapping[source] = canonical

    if mapping:
        data = data.rename(mapping)

    # Promote pl.Date → pl.Datetime("ms") losslessly so downstream joins
    # against other user panels share a common datetime dtype without the
    # user writing an explicit cast.
    # Other Datetime variants (any time_unit, any TZ) pass through — the
    # library is TZ-agnostic and trusts the caller's precision choice.
    if schema.get(date) == pl.Date:
        data = data.with_columns(pl.col("date").cast(pl.Datetime("ms")))

    if fill_forward:
        # Map every non-finite value (NaN and ±inf) to null in one pass, then
        # forward-fill per asset. fill_nan alone leaves ±inf untouched, and inf
        # survives the downstream is_not_null() drop in compute_forward_return,
        # leaking into return math (e.g. a zero entry price yields inf return).
        data = (
            data.sort(["asset_id", "date"])
            .with_columns(
                pl.when(cs.numeric().is_finite()).then(cs.numeric()).otherwise(None)
            )
            .with_columns(cs.numeric().forward_fill().over("asset_id"))
        )

    return data
