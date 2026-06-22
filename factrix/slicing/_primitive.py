"""Slice-partition primitive shared across the slicing package."""

from __future__ import annotations

import polars as pl


def _slice_by(
    df: pl.DataFrame,
    by: str,
) -> dict[str, pl.DataFrame]:
    """Partition ``df`` by the values of an existing column.

    Returns ``{value: sub_df}`` with the ``by`` column dropped from each
    sub-frame (it is constant within a partition and consumers do not
    need it). Raises ``ValueError`` if ``by`` is not a column of
    ``df`` or ``df`` is empty.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"_slice_by expects a polars DataFrame; got {type(df).__name__}."
        )
    if by not in df.columns:
        raise ValueError(
            f"_slice_by: column {by!r} not found in df; "
            f"got columns {df.columns}. Compose the column upstream "
            "(e.g. df.with_columns(pl.lit(...).alias(...)) or a join) "
            "before calling by_slice."
        )
    if df.is_empty():
        raise ValueError(f"_slice_by: df is empty; nothing to slice on {by!r}.")
    if df.get_column(by).null_count() > 0:
        raise ValueError(
            f"_slice_by: column {by!r} contains nulls; "
            f"drop or impute before slicing (e.g. df.drop_nulls({by!r}))."
        )
    return {
        str(key): sub_df
        for (key,), sub_df in df.partition_by(
            by, as_dict=True, include_key=False
        ).items()
    }
