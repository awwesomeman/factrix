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
from factrix._codes import WarningCode
from factrix._metric_index import cell
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _attach_abnormal_return,
    _is_sparse_magnitude_weighted,
)


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
    estimation_window: int = 60,
    overlap_periods: int = 5,
    expected_warnings: tuple[str, ...] = (),
) -> pl.DataFrame:
    r"""Per-event-period weighted abnormal return series.

    Magnitude is preserved — no ``.sign()`` coercion.

    The abnormal return is a real abnormal return: an ``abnormal_return``
    column on the input is used as-is (market-adjusted), and otherwise the
    asset's estimation-window mean, lagged by ``overlap_periods``, is
    subtracted from ``return_col`` — see
    :func:`~factrix.metrics._helpers._attach_abnormal_return`. Events without
    enough history for that estimate carry a null abnormal return and are
    dropped with the other non-finite rows.

    Output columns:
        date: event period (one row per period carrying at least one event).
        caar: cross-asset mean of the (signed/magnitude-weighted)
            abnormal return on that date.
        n_events: number of events (non-zero factor rows) collapsed into
            this date's ``caar``. The downstream ``caar()`` test is an
            equal-weight calendar-time portfolio across event *periods*, so
            this count is the per-period portfolio breadth — surfaced for
            transparency (a date built on 1 event vs 500 is otherwise
            indistinguishable), not used to weight or drop dates.
        date_ordinal: 0-based position of the date on the *full* input
            grid (dense rank over every date in ``data``, including
            non-event periods). Consumers that sub-sample for non-overlap
            independence measure the gap between kept event periods in
            these grid steps rather than in event-index steps —
            the rank is computed before the ``factor != 0`` filter, so a
            gap of ``k`` means ``k`` underlying periods elapsed, not
            ``k`` events. On an event-only series the two diverge under
            sparse or clustered events, so the ordinal is what makes the
            forward-return overlap window measurable downstream.
        n_events_dropped_non_finite: total number of event rows removed
            before aggregation because ``return_col`` or ``factor_col``
            was null / NaN. Broadcast as a constant on every row (the
            count is a whole-frame diagnostic, not a per-period one: a date
            whose events were *all* non-finite leaves the output entirely,
            so a per-period column could not carry it). Consumers surface it
            as ``metadata["n_events_dropped_non_finite"]``.
        sparse_magnitude_weighted: whether the input factor is mixed-sign and
            not a clean ``{-1, 0, +1}`` ternary. Broadcast as a constant so
            downstream event tests can attach the structured warning code.

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
        the surviving per-period mean is taken over finite events only, and the
        dropped count is both reported on the frame and warned about. Note
        ``factor_col`` needs its own guard: ``NaN != 0`` evaluates to *True*
        in polars, so a NaN factor survives the event filter and would make
        ``_signed_car`` NaN.
    """
    sparse_magnitude_weighted = _is_sparse_magnitude_weighted(data, factor_col)
    if (
        sparse_magnitude_weighted
        and WarningCode.SPARSE_MAGNITUDE_WEIGHTED.value not in expected_warnings
    ):
        warnings.warn(
            "compute_caar: factor column is mixed-sign and not a clean ±1 "
            "ternary. The result is the Sefcik-Thompson (1986) "
            "magnitude-weighted CAAR, not the textbook MacKinlay (1997) "
            "signed CAAR; apply .sign() to the column before calling for "
            "sign-flip semantics.",
            UserWarning,
            stacklevel=2,
        )
    # The expected return is estimated on the FULL panel (non-event rows are
    # the estimation window), so the adjustment happens before the event
    # filter.
    adjusted, ar_diagnostics = _attach_abnormal_return(
        data,
        metric_name="caar",
        expected_warnings=expected_warnings,
        return_col=return_col,
        estimation_window=estimation_window,
        overlap_periods=overlap_periods,
        factor_col=factor_col,
    )
    events = adjusted.with_columns(
        (pl.col("date").rank(method="dense") - 1).alias("date_ordinal")
    ).filter(pl.col(factor_col) != 0)
    n_events_in = events.height

    # Two reasons an event leaves the sample, kept apart: a hole in the input
    # columns (a data-quality fact) versus an asset with too little history for
    # the estimation-window mean (a sample-design fact).
    raw_finite = (
        pl.col(return_col).is_not_null()
        & pl.col(return_col).is_not_nan()
        & pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
    )
    ar_finite = (
        pl.col("_abnormal_return").is_not_null()
        & pl.col("_abnormal_return").is_not_nan()
    )
    n_dropped = events.filter(~raw_finite).height
    n_dropped_no_window = events.filter(raw_finite & ~ar_finite).height
    events = events.filter(raw_finite & ar_finite)
    if (
        n_dropped > 0
        and WarningCode.NON_FINITE_INPUT_DROPPED.value not in expected_warnings
    ):
        warnings.warn(
            f"compute_caar: dropped {n_dropped} of {n_events_in} event rows with "
            f"a non-finite '{return_col}' or '{factor_col}' before aggregating. "
            f"Each per-period caar is the mean over the surviving finite events; "
            f"the count is reported as the 'n_events_dropped_non_finite' column.",
            UserWarning,
            stacklevel=2,
        )

    return (
        events.with_columns(
            (pl.col("_abnormal_return") * pl.col(factor_col)).alias("_signed_car")
        )
        .group_by("date")
        .agg(
            pl.col("_signed_car").mean().alias("caar"),
            pl.len().alias("n_events"),
            pl.col("date_ordinal").first().alias("date_ordinal"),
        )
        .sort("date")
        .with_columns(
            pl.lit(n_dropped, dtype=pl.Int64).alias("n_events_dropped_non_finite"),
            pl.lit(sparse_magnitude_weighted).alias("sparse_magnitude_weighted"),
            pl.lit(n_dropped_no_window, dtype=pl.Int64).alias(
                "n_events_dropped_no_estimation_window"
            ),
            pl.lit(str(ar_diagnostics["abnormal_return_model"])).alias(
                "abnormal_return_model"
            ),
            pl.lit(
                ar_diagnostics["estimation_window_event_share"], dtype=pl.Float64
            ).alias("estimation_window_event_share"),
        )
    )
