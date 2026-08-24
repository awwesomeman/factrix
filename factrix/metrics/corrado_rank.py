"""Corrado nonparametric rank test on event abnormal returns.

Standalone metric in cell ``(*, SPARSE, *, PANEL)`` — not part of the
default profile. Available via
``from factrix.metrics import corrado_rank``.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    FactorDensity,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat
from factrix._types import (
    DDOF,
    EPSILON,
    MIN_EVENTS_HARD,
    MIN_EVENTS_WARN,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _enforce_min_floor, _short_circuit_output

__all__ = [
    "corrado_rank",
]


@metric(
    # structure=None (event-axis): the rank test runs across events, so a single
    # name with enough events is a valid sample. Density stays SPARSE; the event
    # floor guards thin samples.
    cell=cell(None, FactorDensity.SPARSE, structure=None),
    aggregation=Aggregation.EVENT_TIME,
    slice_boundary_sensitive=True,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def corrado_rank(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Corrado nonparametric rank test for event abnormal returns.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the rank test on the count of non-zero (event) observations.

    A non-parametric alternative to the CAAR t-test. Robust to extreme
    returns, non-normal distributions, cross-asset heteroscedasticity,
    **and same-date event clustering** — the regime where the parametric
    ``caar`` t-test is unreliable and this metric is the recommended
    fallback. Direction-adjusted for two-sided signals (extension of the
    original one-directional test).

    Formula:
        For each asset $i$, rank ``return`` across the full sample
        (event + non-event, **finite observations only**), transform to
        $U_{i,t} = \mathrm{rank} / (T+1) - 0.5$, and on event rows
        form $U_{\text{event,signed}} = U_{\text{event}} \cdot \mathrm{sign}(\text{factor})$.
        Test statistic
        Same-date events are then averaged into one observation per event
        date, $\bar{U}_d$, and the test runs on that date series:

        $z = \mathrm{mean}(\bar{U}_d) / (\mathrm{std}(\bar{U}_d) / \sqrt{D})$,
        $D$ = number of event dates.

        ``p_value`` is **one-sided** (``H0: mean_u <= 0``): the direction
        adjustment already folds sign into $z$, so a factor that
        anti-predicts returns a negative $z$ rather than a small two-sided
        $p$ — mirroring :func:`~factrix.metrics.directional_hit_rate.directional_hit_rate`.

    Args:
        data: Full panel with ``date, asset_id, factor, forward_return``.
            Must include non-event rows for ranking.

    Returns:
        MetricResult with value=mean rank deviation, stat=z.

    Notes:
        **The event date is the unit of inference.** Events sharing a date
        share whatever moved the market that day, so they are not
        independent draws. Collapsing each date's cross-section to
        $\bar{U}_d$ before taking the time-series SD folds that
        within-date correlation into the denominator; ``n_obs`` therefore
        counts event *dates* (axis ``"periods"``), with the raw event count
        kept in ``metadata["n_events"]`` alongside
        ``events_per_date_mean`` / ``events_per_date_max``.

        An earlier version divided by the **pooled** std of ``U_all`` over
        every ``(asset, date)`` cell and scaled by $\sqrt{N_{events}}$.
        That ignores clustering entirely: piling k correlated events onto
        one date grew $N_{events}$ without growing the denominator, so $z$
        scaled with $\sqrt{k}$ for no new information. Under a simulated
        null with 6 events per date it rejected at 22% against a nominal
        one-sided 5%; the date-clustered denominator gives 3%. The metric
        was liberal in precisely the regime it is recommended for.

        This is the *intent* of Corrado (1989) eq. (5) — a time-series SD
        of a cross-sectional mean — but not its literal form. Corrado's
        design is event-time aligned, so every date's cross-section is the
        full set of N names and the SD may be taken over the whole sample
        period. In a calendar-time sparse panel the event dates carry a
        handful of names while a generic date carries the full universe, so
        an SD taken over all dates is the SD of a much better-averaged
        quantity: on the demo panel (50 names/day, 1.57 events/day) it
        understates the relevant dispersion ~5.7x, i.e. it would be *more*
        liberal than the pooled std it replaced. The SD is therefore taken
        over the event-date series itself, whose scale matches the
        numerator by construction.

        **Non-finite returns.** Ranks are formed over the finite
        ``return_col`` values only (per asset), and $T$ in the
        $\mathrm{rank}/(T+1)$ normalisation is the count of those finite
        values — so a gap in the return series shifts neither the ranks of
        its neighbours nor the uniform scaling. Non-finite event rows (and
        rows with a non-finite factor, which survive the ``!= 0`` filter
        because ``NaN != 0`` is True) are excluded from
        $U_{\text{event}}$ and from the pooled ``std(U_all)``, and the
        excluded event count is reported as
        ``metadata["n_events_dropped_non_finite"]``;
        ``metadata["n_total_obs"]`` counts the finite cells actually behind
        the denominator. This matters because polars ranks a float ``NaN``
        as the largest value rather than treating it as missing: left
        unmasked, every NaN return would enter the test as a maximal
        positive rank deviation.

        Short-circuits to ``MetricResult`` with
        ``metadata["reason"]="insufficient_events"`` when
        ``N_events < MIN_EVENTS_HARD``;
        ``"insufficient_event_dates"`` when fewer than
        ``MIN_EVENTS_HARD`` distinct event dates survive (the time-series
        SD cannot be estimated from a handful of dates, however many events
        sit on them) — the same floor ``caar`` applies to its own event-date
        series, and ``WarningCode.FEW_EVENTS`` fires in
        ``[MIN_EVENTS_HARD, MIN_EVENTS_WARN)``; and
        ``"degenerate_rank_variance"`` when ``std(U_bar_d) < EPSILON``.

    References:
        - [Corrado (1989)][corrado-1989]. "A Nonparametric Test for
          Abnormal Security-price Performance in Event Studies."
          *Journal of Financial Economics* 23(2), 385–395. The
          The nonparametric rank test factrix implements, with the
          denominator taken over the event-date series rather than the
          full sample period (see Notes).
        - [Corrado & Zivney (1992)][corrado-zivney-1992]. "The
          Specification and Power of the Sign Test in Event Study
          Hypothesis Tests Using Daily Stock Returns." *Journal of
          Financial and Quantitative Analysis* 27(3), 465–478. Source
          of the direction-adjustment idea applied to two-sided signals.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.corrado_rank import corrado_rank
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = corrado_rank(panel)
        >>> result.name == ""
        True
    """
    # Rank only the finite returns. Ranking `return_col` directly is wrong
    # twice over: a null return produces a null rank, which turns std(u_all)
    # into NaN and hands _calc_t_stat a NaN (NaN z, NaN p); and a
    # float NaN is not a null to polars, so it ranks as the *largest* value
    # in the asset and is quietly kept as a genuine top-decile observation.
    # Masking to null first makes both cases explicit and excludable, and
    # `count()` (non-null count) then supplies the correct T for the
    # rank / (T + 1) normalisation.
    finite_return = pl.col(return_col).is_not_null() & pl.col(return_col).is_not_nan()
    ranked = data.with_columns(
        pl.when(finite_return).then(pl.col(return_col)).alias("_finite_return")
    ).with_columns(
        (
            pl.col("_finite_return").rank(method="average").over("asset_id")
            / (pl.col("_finite_return").count().over("asset_id") + 1)
            - 0.5
        ).alias("_rank_u")
    )

    all_events = ranked.filter(pl.col(factor_col) != 0)
    # `NaN != 0` is True in polars, so a NaN factor survives the event filter;
    # guard it here as well or sign(factor) turns u_event into NaN.
    events = all_events.filter(
        pl.col("_rank_u").is_not_null()
        & pl.col("_rank_u").is_not_nan()
        & pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
    )
    n_events = len(events)
    n_events_dropped_non_finite = len(all_events) - n_events

    sc = _enforce_min_floor(
        corrado_rank,
        "corrado_rank",
        n_events,
        "insufficient_events",
        axis="events",
        alternative="greater",
    )
    if sc is not None:
        return sc

    # Collapse each event date's cross-section to ONE observation before
    # testing. Same-date events share whatever moved the market that day, so
    # they are not independent draws; averaging them first makes the event
    # DATE the unit of inference and folds the within-date correlation into
    # the denominator, which is the whole point of reaching for this metric
    # when ``clustering_hhi`` is high.
    per_date = (
        events.select(
            "date",
            (pl.col("_rank_u") * pl.col(factor_col).sign()).alias("_u_signed"),
        )
        .group_by("date")
        .agg(
            pl.col("_u_signed").mean().alias("_u_bar"),
            pl.len().alias("_k"),
        )
        .sort("date")
    )
    u_bar = per_date["_u_bar"].to_numpy()
    n_event_dates = len(u_bar)
    events_per_date = per_date["_k"].to_numpy()

    u_all = ranked["_rank_u"].drop_nulls().drop_nans().to_numpy()

    # The date series carries the test, so the floor moves onto dates: 396
    # events spread over 3 days estimate the time-series SD from 3 points.
    # Same constants as ``caar``, deliberately — ``caar`` also tests an
    # event-DATE series, so the two share an axis and stay comparable. A
    # private floor here would have made a quarterly-earnings factor (a few
    # event dates a year) short-circuit under corrado_rank while caar
    # happily reported on the same sample.
    if n_event_dates < MIN_EVENTS_HARD:
        return _short_circuit_output(
            "corrado_rank",
            "insufficient_event_dates",
            alternative="greater",
            n_obs=n_event_dates,
            n_obs_axis="periods",
            min_required=MIN_EVENTS_HARD,
            n_events=n_events,
        )

    warning_codes: list[str] = []
    if n_event_dates < MIN_EVENTS_WARN:
        warning_codes.append(WarningCode.FEW_EVENTS.value)
        warnings.warn(
            f"corrado_rank: n_event_dates={n_event_dates} below "
            f"MIN_EVENTS_WARN={MIN_EVENTS_WARN}. The denominator is the "
            f"time-series SD of the per-event-date mean rank, so this counts "
            f"event *dates*, not events (n_events={n_events}): piling more "
            f"same-date events on does not add sample. z is returned but the "
            f"normal approximation is power-thin here.",
            UserWarning,
            stacklevel=2,
        )

    mean_u = float(np.mean(u_bar))
    std_u = float(np.std(u_bar, ddof=DDOF))

    if std_u < EPSILON:
        return _short_circuit_output(
            "corrado_rank",
            "degenerate_rank_variance",
            alternative="greater",
            n_obs=n_event_dates,
            n_obs_axis="periods",
            std_u=std_u,
            n_events=n_events,
            n_event_dates=n_event_dates,
        )

    z = _calc_t_stat(mean_u, std_u, n_event_dates)
    # One-sided: u_bar is already direction-adjusted by sign(factor), so
    # z > 0 signals genuine directional skill and z < 0 signals a factor that
    # anti-predicts — a two-sided p would read the latter as "significant".
    p = float(sp_stats.norm.sf(z))

    return MetricResult(
        p_value=p,
        alternative="greater",
        value=mean_u,
        n_obs=n_event_dates,
        n_obs_axis="periods",
        stat=z,
        warning_codes=tuple(warning_codes),
        metadata={
            "n_event_dates": n_event_dates,
            "n_events": n_events,
            "events_per_date_mean": float(np.mean(events_per_date)),
            "events_per_date_max": int(np.max(events_per_date)),
            "n_total_obs": len(u_all),
            "n_events_dropped_non_finite": n_events_dropped_non_finite,
            "stat_type": "z",
            "h0": "mu_rank<=0",
            "method": "Corrado (1989) rank test",
        },
    )
