"""Slice-partition primitive shared across the slicing package."""

from __future__ import annotations

import polars as pl

from factrix._errors import UserInputError

_DOCS_BY_SLICE = "api/by-slice"


def _slice_by(
    data: pl.DataFrame,
    by: str,
    *,
    func_name: str = "by_slice",
) -> dict[str, pl.DataFrame]:
    """Partition ``data`` by the values of an existing column.

    Returns ``{value: sub_df}`` with the ``by`` column dropped from each
    sub-frame (it is constant within a partition and consumers do not
    need it). Raises :class:`~factrix._errors.UserInputError` (naming
    ``func_name`` — the public entry point the caller reached this through)
    if ``by`` is not a column of ``data``, ``data`` is empty, or ``by``
    carries nulls.
    """
    if not isinstance(data, pl.DataFrame):
        raise TypeError(
            f"_slice_by expects a polars DataFrame; got {type(data).__name__}."
        )
    if by not in data.columns:
        raise UserInputError(
            func_name=func_name,
            field="by",
            value=by,
            candidates=data.columns,
            expected=(
                "a column of data. Compose it upstream (e.g. "
                "data.with_columns(pl.lit(...).alias(...)) or a join) before "
                "slicing."
            ),
            docs_path=_DOCS_BY_SLICE,
        )
    if data.is_empty():
        raise UserInputError(
            func_name=func_name,
            field="data",
            value="empty DataFrame",
            expected=f"a non-empty frame; there is nothing to slice on {by!r}",
            docs_path=_DOCS_BY_SLICE,
        )
    if data.get_column(by).null_count() > 0:
        raise UserInputError(
            func_name=func_name,
            field="by",
            value=by,
            expected=(
                f"a column with no nulls; drop or impute before slicing "
                f"(e.g. data.drop_nulls({by!r}))"
            ),
            docs_path=_DOCS_BY_SLICE,
        )
    return {
        str(key): sub_df
        for (key,), sub_df in data.partition_by(
            by, as_dict=True, include_key=False
        ).items()
    }
