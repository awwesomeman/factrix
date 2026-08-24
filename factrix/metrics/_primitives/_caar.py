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
    data: pl.DataFrame,
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
            calendar (dense rank over every date in ``data``, including
            non-event dates). Consumers that sub-sample for non-overlap
            independence measure the gap between kept event dates in
            these calendar steps rather than in event-index steps —
            the rank is computed before the ``factor != 0`` filter, so a
            gap of ``k`` means ``k`` underlying periods elapsed, not
            ``k`` events. On an event-only series the two diverge under
            sparse or clustered events, so the ordinal is what makes the
            forward-return overlap window measurable downstream.
        n_events_dropped_non_finite: total number of event rows removed
            before aggregation because ``return_col`` or ``factor_col``
            was null / NaN. Broadcast as a constant on every row (the
            count is a whole-frame diagnostic, not a per-date one: a date
            whose events were *all* non-finite leaves the output entirely,
            so a per-date column could not carry it). Consumers surface it
            as ``metadata["n_events_dropped_non_finite"]``.

    Non-finite handling:
        polars' ``mean`` propagates float ``NaN`` (it only skips nulls), so a
        single NaN ``return_col`` on one event used to poison that whole
        date's ``caar`` — and a NaN caar then reaches ``_calc_t_stat``
        downstream, which returns NaN and makes the metric withhold its test
        as ``degenerate_variance``, mislabelling missing data as
        degeneracy. Event rows are therefore
        filtered to finite ``return_col`` **and** finite ``factor_col``
        before the ``group_by`` (this is the producer boundary that the
        project convention makes responsible for dropping non-finite values),
        the surviving per-date mean is taken over finite events only, and the
        dropped count is both reported on the frame and warned about. Note
        ``factor_col`` needs its own guard: ``NaN != 0`` evaluates to *True*
        in polars, so a NaN factor survives the event filter and would make
        ``_signed_car`` NaN.
    """
    if _is_sparse_magnitude_weighted(data, factor_col):
        warnings.warn(
            "compute_caar: factor column is mixed-sign and not a clean ±1 "
            "ternary. The result is the Sefcik-Thompson (1986) "
            "magnitude-weighted CAAR, not the textbook MacKinlay (1997) "
            "signed CAAR; apply .sign() to the column before calling for "
            "sign-flip semantics.",
            UserWarning,
            stacklevel=2,
        )
    events = data.with_columns(
        (pl.col("date").rank(method="dense") - 1).alias("date_ordinal")
    ).filter(pl.col(factor_col) != 0)
    n_events_in = events.height

    finite = (
        pl.col(return_col).is_not_null()
        & pl.col(return_col).is_not_nan()
        & pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
    )
    events = events.filter(finite)
    n_dropped = n_events_in - events.height
    if n_dropped > 0:
        warnings.warn(
            f"compute_caar: dropped {n_dropped} of {n_events_in} event rows with "
            f"a non-finite '{return_col}' or '{factor_col}' before aggregating. "
            f"Each per-date caar is the mean over the surviving finite events; "
            f"the count is reported as the 'n_events_dropped_non_finite' column.",
            UserWarning,
            stacklevel=2,
        )

    return (
        events.with_columns(
            (pl.col(return_col) * pl.col(factor_col)).alias("_signed_car")
        )
        .group_by("date")
        .agg(
            pl.col("_signed_car").mean().alias("caar"),
            pl.len().alias("n_events"),
            pl.col("date_ordinal").first().alias("date_ordinal"),
        )
        .sort("date")
        .with_columns(
            pl.lit(n_dropped, dtype=pl.Int64).alias("n_events_dropped_non_finite")
        )
    )
