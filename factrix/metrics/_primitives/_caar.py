from __future__ import annotations

import warnings

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import cell
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _is_sparse_magnitude_weighted


@metric(
    cell=cell(
        None, FactorDensity.SPARSE, DataStructure.PANEL, raw="(*, SPARSE, PANEL)"
    ),
    aggregation=Aggregation.EVENT_TIME,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_caar(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    r"""Per-event-date weighted abnormal return series.

    Magnitude is preserved — no ``.sign()`` coercion.

    Output columns:
        date: event date (one row per date carrying at least one event).
        caar: cross-asset mean of the (signed/magnitude-weighted)
            abnormal return on that date.
        n_events: number of events (non-zero factor rows) collapsed into
            this date's ``caar``. The downstream ``caar()`` test is an
            equal-weight calendar-time portfolio across event *dates*, so
            this count is the per-date portfolio breadth — surfaced for
            transparency (a date built on 1 event vs 500 is otherwise
            indistinguishable), not used to weight or drop dates.
        date_ordinal: 0-based position of the date on the *full* input
            calendar (dense rank over every date in ``df``, including
            non-event dates). Consumers that sub-sample for non-overlap
            independence measure the gap between kept event dates in
            these calendar steps rather than in event-index steps —
            the rank is computed before the ``factor != 0`` filter, so a
            gap of ``k`` means ``k`` underlying periods elapsed, not
            ``k`` events. On an event-only series the two diverge under
            sparse or clustered events, so the ordinal is what makes the
            forward-return overlap window measurable downstream.
    """
    if _is_sparse_magnitude_weighted(df, factor_col):
        warnings.warn(
            "compute_caar: factor column is mixed-sign and not a clean ±1 "
            "ternary. The result is the Sefcik-Thompson (1986) "
            "magnitude-weighted CAAR, not the textbook MacKinlay (1997) "
            "signed CAAR; apply .sign() to the column before calling for "
            "sign-flip semantics.",
            UserWarning,
            stacklevel=2,
        )
    return (
        df.with_columns((pl.col("date").rank(method="dense") - 1).alias("date_ordinal"))
        .filter(pl.col(factor_col) != 0)
        .with_columns((pl.col(return_col) * pl.col(factor_col)).alias("_signed_car"))
        .group_by("date")
        .agg(
            pl.col("_signed_car").mean().alias("caar"),
            pl.len().alias("n_events"),
            pl.col("date_ordinal").first().alias("date_ordinal"),
        )
        .sort("date")
    )
