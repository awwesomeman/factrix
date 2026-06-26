"""Slice-partition primitive shared across the slicing package."""

from __future__ import annotations

import polars as pl


def _slice_by(
    data: pl.DataFrame,
    by: str,
) -> dict[str, pl.DataFrame]:
    """Partition ``data`` by the values of an existing column.

    Returns ``{value: sub_df}`` with the ``by`` column dropped from each
    sub-frame (it is constant within a partition and consumers do not
    need it). Raises ``ValueError`` if ``by`` is not a column of
    ``data`` or ``data`` is empty.
    """
    if not isinstance(data, pl.DataFrame):
        raise TypeError(
            f"_slice_by expects a polars DataFrame; got {type(data).__name__}."
        )
    if by not in data.columns:
        raise ValueError(
            f"_slice_by: column {by!r} not found in data; "
            f"got columns {data.columns}. Compose the column upstream "
            "(e.g. data.with_columns(pl.lit(...).alias(...)) or a join) "
            "before calling by_slice."
        )
    if data.is_empty():
        raise ValueError(f"_slice_by: data is empty; nothing to slice on {by!r}.")
    if data.get_column(by).null_count() > 0:
        raise ValueError(
            f"_slice_by: column {by!r} contains nulls; "
            f"drop or impute before slicing (e.g. data.drop_nulls({by!r}))."
        )
    return {
        str(key): sub_df
        for (key,), sub_df in data.partition_by(
            by, as_dict=True, include_key=False
        ).items()
    }
