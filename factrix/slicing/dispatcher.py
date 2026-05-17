"""Axis-agnostic slice dispatcher.

Public :func:`by_slice` partitions a metric's date-keyed input on an
existing column and applies the metric per slice. Universe-overlap
composition is user-side; see ``docs/api/by-slice.md`` for reference
patterns.

Matrix-row: by_slice | (*, *, *, *) | dispatcher | none (no cross-slice test) | _slice_by_label
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from factrix._types import MetricOutput
from factrix.slicing._primitive import _slice_by_label
from factrix.slicing.result import SliceResult


def by_slice(
    metric: Callable[..., MetricOutput],
    df: pl.DataFrame,
    *,
    label: str,
    **kwargs: Any,
) -> SliceResult:
    """Apply ``metric`` to each value-partition of ``df`` keyed by ``label``.

    Universe overlap (superset / multi-membership / hierarchical /
    sliding window / cross-product) is composed by the caller — see
    the API page for reference patterns. ``by_slice`` does no
    cross-slice statistical inference; for paired comparison see
    :func:`factrix.slice_pairwise_test` / :func:`factrix.slice_joint_test`.

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
        :class:`SliceResult` — ``Mapping[str, MetricOutput]`` (so every
        ``dict``-shaped consumer keeps working) plus a
        :meth:`SliceResult.to_frame` long-form renderer. Keys are
        stringified (an ``Int64`` decile column yields ``"1".."10"``);
        iteration order matches polars
        ``partition_by(as_dict=True)``. No cross-slice statistical
        inference — see API page.

    Raises:
        TypeError: ``df`` is not a polars DataFrame.
        ValueError: ``label`` not in ``df.columns``, or ``df`` is empty.

    Examples:
        Per-year information coefficient (IC) on a synthetic cross-sectional panel — attach the
        slice label to the metric's primary DataFrame upstream, then
        dispatch one metric call per slice value:

        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics import ic, compute_ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=500)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> ic_df = compute_ic(panel)["factor"].with_columns(
        ...     pl.col("date").dt.year().alias("year")
        ... )
        >>> per_year = fx.by_slice(ic, ic_df, label="year")
    """
    sliced = _slice_by_label(df, label)
    return SliceResult(
        {key: metric(sub_df, **kwargs) for key, sub_df in sliced.items()}
    )
