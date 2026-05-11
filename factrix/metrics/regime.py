"""Regime analysis: legacy dispatcher pending removal.

:func:`by_regime` is deprecated since v0.10.0 and removed in #217. It
slices a metric input by regime label and applies a metric callable
per slice, returning ``dict[str, MetricOutput]`` — **no cross-slice
statistical test**. Callers wanting inferential contrasts should join
labels upstream and call :func:`factrix.slice_pairwise_test` or
:func:`factrix.slice_joint_test`.

Matrix-row: by_regime | (*, *, *, *) | dispatcher | none (no cross-regime test) | _slice_by_regime
"""

from __future__ import annotations

import warnings as _warnings
from collections.abc import Callable
from typing import Any

import polars as pl

from factrix._types import MetricOutput


def _slice_by_regime(
    df: pl.DataFrame,
    regime_labels: pl.DataFrame | None,
    date_col: str = "date",
) -> pl.DataFrame:
    """Annotate ``df`` with a ``regime`` column.

    With ``regime_labels``: inner-join on ``date_col`` (rows without a
    label are dropped — they cannot be assigned to a regime).
    Without: time-bisection fallback labelled ``first_half`` /
    ``second_half`` and a ``UserWarning`` is emitted so callers do not
    silently rely on what is structurally a break test, not a regime
    test. The label format is fixed and greppable.
    """
    if regime_labels is not None:
        if "regime" not in regime_labels.columns:
            raise ValueError(
                "regime_labels missing required column 'regime'; "
                f"got columns {regime_labels.columns}"
            )
        return df.join(
            regime_labels.select(date_col, "regime"), on=date_col, how="inner"
        )
    _warnings.warn(
        "by_regime / regime_<metric>: no regime_labels supplied — "
        "falling back to time-bisection (first_half / second_half). "
        "This is a structural-break check, not a regime test. Pass "
        "regime_labels with a domain-driven classification for any "
        "decision-grade analysis.",
        UserWarning,
        stacklevel=3,
    )
    sorted_df = df.sort(date_col)
    mid = len(sorted_df) // 2
    return (
        sorted_df.with_row_index("_idx")
        .with_columns(
            pl.when(pl.col("_idx") < mid)
            .then(pl.lit("first_half"))
            .otherwise(pl.lit("second_half"))
            .alias("regime")
        )
        .drop("_idx")
    )


def by_regime(
    metric: Callable[..., MetricOutput],
    df: pl.DataFrame,
    *,
    regime_labels: pl.DataFrame | None = None,
    **kwargs: Any,
) -> dict[str, MetricOutput]:
    """Apply a metric to each regime slice of ``df``.

    Convention-based dispatch — pass the metric callable and its
    primary date-keyed DataFrame; ``by_regime`` slices the DataFrame
    by regime and forwards the rest of ``kwargs`` to ``metric`` on
    every call. Works with any metric whose first positional argument
    is a DataFrame carrying a ``date`` column.

    Args:
        metric: Callable returning a ``MetricOutput`` (e.g.
            :func:`factrix.metrics.ic`, :func:`factrix.metrics.caar`).
        df: The metric's primary date-keyed input.
        regime_labels: ``(date, regime)`` DataFrame. ``None`` triggers
            the time-bisection fallback inside :func:`_slice_by_regime`.
        **kwargs: Forwarded unchanged to ``metric`` on every per-regime
            call.

    Returns:
        ``{regime_label: metric(slice, **kwargs)}``. **No cross-regime
        statistic is computed** — for inferential contrasts call
        :func:`factrix.slice_pairwise_test` (or
        :func:`factrix.slice_joint_test`) on a labels-joined frame.

    Raises:
        ValueError: ``df`` has no ``date`` column, or no rows survive
            the regime-label join.

    Example:
        ```python
        from factrix.metrics import by_regime, compute_ic, ic

        ic_df = compute_ic(panel)
        per_regime = by_regime(ic, ic_df, regime_labels=labels)
        ```
    """
    _warnings.warn(
        "factrix.metrics.by_regime is deprecated since v0.10.0; use "
        "factrix.metrics.by_slice instead. With regime_labels, replace "
        "by_regime(metric, df, regime_labels=labels) with "
        "by_slice(metric, df.join(labels, on='date'), label='regime'). "
        "Without regime_labels (time-bisection fallback), compose the "
        "label yourself — see docs/api/by-regime.md migration block.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"by_regime expects a polars DataFrame as the second arg; "
            f"got {type(df).__name__}. Scalar-input metrics like "
            f"breakeven_cost / net_spread are not regime-dispatchable — "
            f"compute their per-regime inputs (gross_spread, turnover) "
            f"via by_regime first, then combine the scalars."
        )
    if "date" not in df.columns:
        raise ValueError(f"by_regime: df must have a 'date' column; got {df.columns}")
    annotated = _slice_by_regime(df, regime_labels)
    if annotated.is_empty():
        raise ValueError(
            "by_regime: no rows survived the regime-label join — "
            "likely a date-range or dtype mismatch between df and regime_labels"
        )
    from factrix.slicing import by_slice

    return by_slice(metric, annotated, label="regime", **kwargs)
