"""Corrado nonparametric rank test on event abnormal returns.

Standalone metric in cell ``(*, SPARSE, *, PANEL)`` — not part of the
default profile. Available via
``from factrix.metrics import corrado_rank``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    FactorDensity,
)
from factrix._codes import WarningCode
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat
from factrix._types import (
    DDOF,
    EPSILON,
    MIN_EVENTS_HARD,
    MIN_EVENTS_WARN,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _attach_abnormal_return,
    _enforce_min_floor,
    _event_sample_threshold,
    _is_sparse_magnitude_weighted,
    _sample_events_non_overlapping,
    _scaled_min_periods,
    _short_circuit_output,
    _warn_below_scaled_floor,
    _warn_estimation_window_contamination,
    _warn_event_window_overlap,
)

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
    sample_threshold=_event_sample_threshold,
)
def corrado_rank(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    overlap_periods: int = 5,
    estimation_window: int = 60,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Corrado nonparametric rank test for event abnormal returns.

    The static event floor (MIN_EVENTS_HARD) gates the rank test on the count of
    non-zero (event) observations that survive the event-axis spacing pass; the
    warn floor scales with the overlap_periods parameter (the spacing stride),
    so the threshold is declared as a resolver (a callable sample_threshold)
    rather than a constant.

    A non-parametric alternative to the CAAR t-test. Robust to extreme
    returns, non-normal distributions, cross-asset heteroscedasticity,
    **and same-period event clustering** — the regime where the parametric
    ``caar`` t-test is unreliable and this metric is the recommended
    fallback. Direction-adjusted for two-sided signals (extension of the
    original one-directional test).

    Formula:
        For each asset $i$, rank ``return`` across the full sample
        (event + non-event, **finite observations only**), transform to
        $U_{i,t} = \mathrm{rank} / (T+1) - 0.5$, and on event rows
        form $U_{\text{event,signed}} = U_{\text{event}} \cdot \mathrm{sign}(\text{factor})$.
        Test statistic
        Same-period events are then averaged into one observation per event
        period, $\bar{U}_d$, and the test runs on that period series:

        $z = \mathrm{mean}(\bar{U}_d) / (\mathrm{std}(\bar{U}_d) / \sqrt{D})$,
        $D$ = number of event periods.

        ``p_value`` is **one-sided** (``H0: mean_u <= 0``): the direction
        adjustment already folds sign into $z$, so a factor that
        anti-predicts returns a negative $z$ rather than a small two-sided
        $p$ — mirroring :func:`~factrix.metrics.directional_hit_rate.directional_hit_rate`.

    Args:
        data: Full panel with ``date, asset_id, factor, forward_return``.
            Must include non-event rows for ranking.
        overlap_periods: Forward-return horizon, injected by ``evaluate`` from
            the panel metadata; standalone calls may pass it directly. Sets the
            minimum calendar gap between two kept events on one asset, so the
            tested events no longer share overlapping return windows.

    Returns:
        MetricResult with value=mean rank deviation, stat=z.

    Notes:
        **The event period is the unit of inference.** Events sharing a period
        share whatever moved the market then, so they are not
        independent draws. Collapsing each period's cross-section to
        $\bar{U}_d$ before taking the time-series SD folds that
        within-period correlation into the denominator; ``n_obs`` therefore
        counts event *periods* — reported on the battery's shared axis token
        ``"events"`` (every member's sample is a set of non-overlapping event
        observations) with the count itself in
        ``metadata["n_event_periods"]`` — and the raw event count is
        kept in ``metadata["n_events"]`` alongside
        ``events_per_period_mean`` / ``events_per_period_max``.

        An earlier version divided by the **pooled** std of ``U_all`` over
        every ``(asset, date)`` cell and scaled by $\sqrt{N_{events}}$.
        That ignores clustering entirely: piling k correlated events onto
        one period grew $N_{events}$ without growing the denominator, so $z$
        scaled with $\sqrt{k}$ for no new information. Under a simulated
        null with 6 events per period it rejected at 22% against a nominal
        one-sided 5%; the period-clustered denominator gives 3%. The metric
        was liberal in precisely the regime it is recommended for.

        This is the *intent* of Corrado (1989) eq. (5) — a time-series SD
        of a cross-sectional mean — but not its literal form. Corrado's
        design is event-time aligned, so every period's cross-section is the
        full set of N names and the SD may be taken over the whole sample
        period. In a calendar-time sparse panel the event periods carry a
        handful of names while a non-event period carries the full universe,
        so an SD taken over all periods is the SD of a much better-averaged
        quantity: on the demo panel (50 names per period, 1.57 events per period) it
        understates the relevant dispersion ~5.7x, i.e. it would be *more*
        liberal than the pooled std it replaced. The SD is therefore taken
        over the event-period series itself, whose scale matches the
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

        **Missing data.** $T_i$ is the count of non-missing returns for asset
        $i$, so the $\mathrm{rank} / (T_i + 1)$ scaling is the missing-data
        denominator of [Corrado & Zivney (1992)][corrado-zivney-1992] rather
        than a fixed window length: a gap in one asset's series shifts neither
        its neighbours' ranks nor the uniform scaling, and the excluded event
        count is reported.

        **Time-axis overlap.** Before the per-period collapse, events are
        strided per asset so no two kept events on one name sit inside one
        forward-return window — [Brown-Warner (1985)][brown-warner-1985]
        non-overlap sampling on the event axis, the same treatment ``caar``
        applies to its event-period series. The collapse and the stride are
        complements: the collapse removes same-period dependence across names,
        the stride removes same-asset dependence across time. Without it the
        rank test over-rejected a bursty single-asset trigger at 21% against a
        nominal 5%; ``EVENT_WINDOW_OVERLAP`` reports what was removed.

        Short-circuits to ``MetricResult`` with
        ``metadata["reason"]="insufficient_events"`` when
        ``N_events < MIN_EVENTS_HARD``;
        ``"insufficient_event_periods"`` when fewer than
        ``MIN_EVENTS_HARD`` distinct event periods survive (the time-series
        SD cannot be estimated from a handful of periods, however many events
        sit on them) — the same floor ``caar`` applies to its own event-period
        series, and ``WarningCode.FEW_EVENTS`` fires in
        ``[MIN_EVENTS_HARD, MIN_EVENTS_WARN)``; and
        ``"degenerate_rank_variance"`` when ``std(U_bar_d) < EPSILON``.

    References:
        - [Corrado (1989)][corrado-1989]. "A Nonparametric Test for
          Abnormal Security-price Performance in Event Studies."
          *Journal of Financial Economics* 23(2), 385–395. The
          The nonparametric rank test factrix implements, with the
          denominator taken over the event-period series rather than the
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
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=0),
        ...     forward_periods=5,
        ... )
        >>> result = corrado_rank(panel)
        >>> result.name == ""
        True
    """
    sparse_magnitude_weighted = _is_sparse_magnitude_weighted(data, factor_col)

    # Rank only the finite returns. Ranking `return_col` directly is wrong
    # twice over: a null return produces a null rank, which propagates into
    # that period's mean and turns the event-period SD into NaN, handing
    # _calc_t_stat a NaN (NaN z, NaN p); and a
    # float NaN is not a null to polars, so it ranks as the *largest* value
    # in the asset and is quietly kept as a genuine top-decile observation.
    # Masking to null first makes both cases explicit and excludable, and
    # `count()` (non-null count) then supplies the correct T for the
    # rank / (T + 1) normalisation.
    # Corrado ranks ABNORMAL returns. Ranking the raw forward return makes the
    # rank of an event depend on the asset's drift as much as on the event.
    data, ar_diagnostics = _attach_abnormal_return(
        data,
        return_col=return_col,
        estimation_window=estimation_window,
        overlap_periods=overlap_periods,
        factor_col=factor_col,
    )
    ar_col = "_abnormal_return"
    finite_return = pl.col(ar_col).is_not_null() & pl.col(ar_col).is_not_nan()
    ranked = data.with_columns(
        pl.when(finite_return).then(pl.col(ar_col)).alias("_finite_return")
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
    #
    # Two drop reasons, reported apart: a hole in the input columns, versus an
    # event whose asset has too little history for the estimation-window mean
    # (so its abnormal return, and therefore its rank, does not exist). The
    # second is a sample-design fact — the same one bmp_z reports as
    # n_dropped_no_vol — not a data-quality one.
    raw_finite = (
        pl.col(return_col).is_not_null()
        & pl.col(return_col).is_not_nan()
        & pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
    )
    rank_finite = pl.col("_rank_u").is_not_null() & pl.col("_rank_u").is_not_nan()
    events = all_events.filter(raw_finite & rank_finite)
    n_events = len(events)
    n_events_dropped_non_finite = all_events.filter(~raw_finite).height
    n_events_dropped_no_estimation_window = all_events.filter(
        raw_finite & ~rank_finite
    ).height

    # Overlap discipline on the TIME axis, applied before the cross-section is
    # collapsed. The per-period collapse below removes SAME-PERIOD dependence;
    # it cannot see two events on one asset a few periods apart, whose
    # ``(t, t+h]`` windows overlap. On a single-asset panel that is the only
    # clustering axis there is, so without this pass the rank test over-rejects.
    events = _sample_events_non_overlapping(
        events, overlap_periods, grid_dates=data["date"]
    )
    n_events_sampled = len(events)

    # The floor is enforced on the SAMPLED count — the sample the rank test
    # actually runs on. Pre-flight reads the raw non-zero factor count as the
    # documented loose upper bound.
    sc = _enforce_min_floor(
        corrado_rank,
        "corrado_rank",
        n_events_sampled,
        "insufficient_events",
        axis="events",
        alternative="greater",
        overlap_periods=overlap_periods,
        n_events_raw=n_events,
    )
    if sc is not None:
        return sc

    # Collapse each event period's cross-section to ONE observation before
    # testing. Same-period events share whatever moved the market then, so
    # they are not independent draws; averaging them first makes the event
    # PERIOD the unit of inference and folds the within-period correlation into
    # the denominator, which is the whole point of reaching for this metric
    # when ``clustering_hhi`` is high.
    per_period = (
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
    u_bar = per_period["_u_bar"].to_numpy()
    n_event_periods = len(u_bar)
    events_per_period = per_period["_k"].to_numpy()

    n_total_obs = int(ranked["_rank_u"].drop_nulls().drop_nans().len())

    # The period series carries the test, so the floor moves onto periods:
    # 396 events spread over 3 periods estimate the time-series SD from 3
    # points.
    # Same constants as ``caar``, deliberately — ``caar`` also tests an
    # event-period series, so the two share an axis and stay comparable. A
    # private floor here would have made a quarterly-earnings factor (a few
    # event periods a year) short-circuit under corrado_rank while caar
    # happily reported on the same sample.
    if n_event_periods < MIN_EVENTS_HARD:
        return _short_circuit_output(
            "corrado_rank",
            "insufficient_event_periods",
            alternative="greater",
            n_obs=n_event_periods,
            n_obs_axis="events",
            min_required=MIN_EVENTS_HARD,
            n_events=n_events,
        )

    warning_codes: list[str] = []
    if sparse_magnitude_weighted:
        warning_codes.append(WarningCode.SPARSE_MAGNITUDE_WEIGHTED.value)
    metadata: dict = {
        "n_event_periods": n_event_periods,
        "n_events": n_events,
        "events_per_period_mean": float(np.mean(events_per_period)),
        "events_per_period_max": int(np.max(events_per_period)),
        "n_total_obs": n_total_obs,
        "n_events_dropped_non_finite": n_events_dropped_non_finite,
        "n_events_dropped_no_estimation_window": n_events_dropped_no_estimation_window,
        "stat_type": "z",
        "h0": "mu_rank<=0",
        "method": "Corrado (1989) rank test",
        "overlap_periods": overlap_periods,
        **ar_diagnostics,
    }
    _warn_event_window_overlap(
        "corrado_rank",
        n_events,
        n_events_sampled,
        overlap_periods,
        metadata,
        warning_codes,
        expected_warnings=expected_warnings,
    )
    _warn_estimation_window_contamination(
        "corrado_rank", metadata, warning_codes, expected_warnings=expected_warnings
    )
    raw_min_warn = _scaled_min_periods(MIN_EVENTS_WARN, overlap_periods)
    warn_code = _warn_below_scaled_floor(
        n_events,
        MIN_EVENTS_WARN,
        overlap_periods,
        f"corrado_rank: n_events={n_events} below the floor of {raw_min_warn} "
        f"(= MIN_EVENTS_WARN {MIN_EVENTS_WARN} x overlap_periods "
        f"{overlap_periods}); {n_event_periods} event periods survive "
        f"non-overlap sampling at stride h={overlap_periods}, which keeps "
        f"up to one event in h per asset. The denominator is the time-series "
        f"SD of the per-event-period mean rank, so the sample is those "
        f"periods, not the raw events: piling more same-period events on does "
        f"not add sample. z is returned but the normal approximation is "
        f"power-thin here.",
        WarningCode.FEW_EVENTS,
        expected_warnings=expected_warnings,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    mean_u = float(np.mean(u_bar))
    std_u = float(np.std(u_bar, ddof=DDOF))

    if std_u < EPSILON:
        return _short_circuit_output(
            "corrado_rank",
            "degenerate_rank_variance",
            alternative="greater",
            n_obs=n_event_periods,
            n_obs_axis="events",
            std_u=std_u,
            n_events=n_events,
            n_event_periods=n_event_periods,
        )

    z = _calc_t_stat(mean_u, std_u, n_event_periods)
    # One-sided: u_bar is already direction-adjusted by sign(factor), so
    # z > 0 signals genuine directional skill and z < 0 signals a factor that
    # anti-predicts — a two-sided p would read the latter as "significant".
    p = float(sp_stats.norm.sf(z))

    return MetricResult(
        p_value=p,
        alternative="greater",
        value=mean_u,
        n_obs=n_event_periods,
        n_obs_axis="events",
        stat=z,
        warning_codes=tuple(warning_codes),
        metadata=metadata,
    )
