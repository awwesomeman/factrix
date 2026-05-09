"""Axis-agnostic slice dispatcher.

Public :func:`by_slice` partitions a metric's date-keyed input on an
existing column and applies the metric per slice. Private
:func:`_slice_by_label` is the partition primitive shared with
curated wrappers. Universe-overlap composition is user-side; see
``docs/api/by-slice.md`` for reference patterns.

(Previously called Layer-A; renamed per #157.)

Matrix-row: by_slice | (*, *, *, *) | dispatcher | none (no cross-slice test) | _slice_by_label
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from factrix._types import MetricOutput


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


def by_slice(
    metric: Callable[..., MetricOutput],
    df: pl.DataFrame,
    *,
    label: str,
    **kwargs: Any,
) -> dict[str, MetricOutput]:
    """Apply ``metric`` to each value-partition of ``df`` keyed by ``label``.

    Args:
        metric: Callable returning a ``MetricOutput`` (e.g.
            :func:`factrix.metrics.ic`, :func:`factrix.metrics.caar`).
        df: Metric's primary DataFrame; must already contain ``label``
            as a column. If you have a separate labels DataFrame, join
            it in upstream — :func:`by_slice` deliberately does not
            ingest a separate labels argument.
        label: Column name in ``df`` whose distinct values define the
            slices. Must already exist in ``df.columns``; for
            cross-product slicing (e.g. market × sector) compose a
            single composite column upstream
            (``pl.concat_str([...]).alias("...")``).
        **kwargs: Forwarded unchanged to ``metric`` on every per-slice
            call.

    Returns:
        ``{label_value: metric(slice, **kwargs)}``. Keys are
        stringified (an ``Int64`` decile column yields ``"1".."10"``);
        dict order matches polars ``partition_by(as_dict=True)``. No
        cross-slice statistical inference — see API page.

    Raises:
        TypeError: ``df`` is not a polars DataFrame.
        ValueError: ``label`` not in ``df.columns``, or ``df`` is empty.

    Example:
        >>> import polars as pl
        >>> from factrix.metrics import by_slice, ic, compute_ic
        >>> ic_df = compute_ic(panel)              # already has 'sector'
        >>> per_sector = by_slice(ic, ic_df, label="sector")

    Universe overlap (superset / multi-membership / hierarchical /
    sliding window / cross-product) is composed by the caller — see
    the API page for reference patterns. ``by_slice`` does no
    cross-slice statistical inference; for IC use
    :func:`factrix.metrics.regime_ic` (or a future ``by_<scope>_<metric>``
    curated wrapper) instead.
    """
    sliced = _slice_by_label(df, label)
    return {key: metric(sub_df, **kwargs) for key, sub_df in sliced.items()}
