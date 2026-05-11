"""Slice-partition primitive shared across the slicing package."""

from __future__ import annotations

import polars as pl


def _slice_by_label(
    df: pl.DataFrame,
    label: str,
) -> dict[str, pl.DataFrame]:
    """Partition ``df`` by the values of an existing column.

    Returns ``{value: sub_df}`` with the label column dropped from each
    sub-frame (it is constant within a partition and consumers do not
    need it). Raises ``ValueError`` if ``label`` is not a column of
    ``df`` or ``df`` is empty.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"_slice_by_label expects a polars DataFrame; got {type(df).__name__}."
        )
    if label not in df.columns:
        raise ValueError(
            f"_slice_by_label: label column {label!r} not found in df; "
            f"got columns {df.columns}. Compose the label upstream "
            "(e.g. df.with_columns(pl.lit(...).alias(...)) or a join) "
            "before calling by_slice."
        )
    if df.is_empty():
        raise ValueError(
            f"_slice_by_label: df is empty; nothing to slice on {label!r}."
        )
    if df.get_column(label).null_count() > 0:
        raise ValueError(
            f"_slice_by_label: label column {label!r} contains nulls; "
            f"drop or impute before slicing (e.g. df.drop_nulls({label!r}))."
        )
    return {
        str(key): sub_df
        for (key,), sub_df in df.partition_by(
            label, as_dict=True, include_key=False
        ).items()
    }
