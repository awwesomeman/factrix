"""Per-event quality descriptive statistics for event signals.

All metrics operate on the ``signed_car`` (return x sign(factor)) of
individual events. They describe the quality and shape of per-event
outcomes — distinct from significance testing (caar.py) and path
analysis (mfe_mae.py).

Metrics:
    event_hit_rate — fraction of correct-direction events (binomial test)
    event_ic       — density strength → return correlation (Spearman)
    signal_density — average time gap between events
    profit_factor  — sum(gains) / sum(losses)
    event_skewness — skewness of signed_car distribution

Notes:
    **Pipeline.** Per-event scalar (hit / information coefficient (IC) / skew / density) computed
    on `signed_car`, then cross-event aggregation; binomial inference
    for hit rate, nonparametric for IC / skewness, descriptive
    elsewhere.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    FactorDensity,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _binomial_two_sided_p
from factrix._types import EPSILON, MIN_EVENTS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _enforce_min_floor,
    _event_signal_is_discrete,
    _finite_expr,
    _short_circuit_output,
    _signed_car,
)

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "event_hit_rate",
    "event_ic",
    "profit_factor",
    "event_skewness",
    "signal_density",
]

# structure=None (event-axis): event_hit_rate / event_ic / profit_factor aggregate
# over events, so a single name with enough events is valid. Density stays SPARSE;
# the event floor guards thin samples.
_EQ_CELL = cell(None, FactorDensity.SPARSE, structure=None)


def _finite_events(
    data: pl.DataFrame, factor_col: str, return_col: str
) -> tuple[pl.DataFrame, int]:
    """Event rows (``factor != 0``) restricted to finite factor *and* return.

    Single owner for the non-finite boundary in this module. Every metric here
    reduces ``signed_car = return x sign(factor)`` to a scalar, and each one
    used to mishandle a hole in that column differently: ``event_hit_rate``
    scored a NaN as ``not > 0`` and therefore as a *miss* (biasing the rate
    down and the binomial p towards spurious significance in the negative
    direction), ``profit_factor`` excluded NaNs from both sums but still
    divided the story by the unfiltered ``n_obs``, and ``event_skewness``
    propagated NaN into ``skewtest`` and raised out of ``MetricResult``.
    Filtering once, immediately after the ``factor != 0`` filter, makes
    ``value`` / ``stat`` / ``p_value`` / ``n_obs`` share one sample.

    ``factor_col`` needs its own finiteness guard: polars evaluates
    ``NaN != 0`` as True, so a NaN factor is *not* removed by the event
    filter and would make ``sign(factor)`` — and hence ``signed_car`` — NaN.

    Returns:
        ``(events, n_dropped)`` — the filtered frame and the number of event
        rows removed, for ``metadata["n_events_dropped_non_finite"]``.
    """
    events = data.filter(pl.col(factor_col) != 0)
    n_all = events.height
    finite = (
        pl.col(return_col).is_not_null()
        & pl.col(return_col).is_not_nan()
        & pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
    )
    events = events.filter(finite)
    return events, n_all - events.height


@metric(
    cell=_EQ_CELL,
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def event_hit_rate(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Fraction of events with return in expected direction.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the hit-rate binomial test on the count of non-zero (event) observations.

    Args:
        data: Panel with event density and forward return.

    Returns:
        MetricResult with value=hit rate, stat=hit count (the exact
        binomial test statistic).

    Notes:
        ``hits = sum_i 1{signed_car_i > 0}``, ``rate = hits / N``.
        Two-sided **exact** binomial test against ``H0: p = 0.5``
        (``scipy.stats.binomtest``) at every sample size.

        factrix uses the exact test unconditionally rather than switching
        to the normal approximation ``z = (hits - N/2) / (sqrt(N)/2)`` on
        large samples. The approximation is the mainstream shortcut and is
        cheaper, but it is anti-conservative in the tails — precisely where
        the p-value is read — and the exact test costs nothing at
        event-study sample sizes. ``stat`` is therefore always the raw hit
        count (``stat_type="binomial_hits"``); no Gaussian z is published,
        so an exact p can never be paired with an approximate statistic.

        Events whose ``return_col`` or ``factor_col`` is null / NaN are
        dropped before counting (see :func:`_finite_events`) and reported
        as ``metadata["n_events_dropped_non_finite"]``. Previously such an
        event failed ``signed_car > 0`` and was scored as a **miss**, which
        both depressed the rate and inflated ``N``.

        ``return_col`` must be sign-symmetric around zero — ``signed_car =
        return_col * sign(factor_col)``, so an always-positive magnitude
        target (realised volatility, turnover) collapses ``sign(signed_car)``
        to ``sign(factor_col)`` and silently turns the hit rate into a count
        of positive-factor events, with no error raised. Use
        :func:`~factrix.metrics.ic.ic` or
        :func:`~factrix.metrics.monotonicity.monotonicity` for those targets
        instead.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.event_quality import event_hit_rate
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = event_hit_rate(panel)
        >>> result.name == ""
        True
    """
    events, n_dropped = _finite_events(data, factor_col, return_col)

    n = len(events)
    sc = _enforce_min_floor(
        event_hit_rate, "event_hit_rate", n, "insufficient_events", axis="events"
    )
    if sc is not None:
        return sc

    signed = _signed_car(events, factor_col, return_col)
    hits = int(np.sum(signed > 0))
    rate = hits / n
    p = _binomial_two_sided_p(hits, n, p0=0.5)

    return MetricResult(
        p_value=p,
        alternative="two-sided",
        value=rate,
        n_obs=n,
        n_obs_axis="events",
        stat=float(hits),
        metadata={
            "n_events": n,
            "n_hits": hits,
            "n_events_dropped_non_finite": n_dropped,
            "stat_type": "binomial_hits",
            "h0": "p=0.5",
            "method": "binomial exact test",
        },
    )


@metric(
    cell=_EQ_CELL,
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
    requires_continuous_magnitude=True,
)
def event_ic(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Spearman correlation between factor value and realised forward return.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the rank correlation on the count of non-zero (event) observations.

    Spearman correlation between ``|factor|`` and ``signed_car``
    (``return × sign(factor)``), computed only on event rows.

    Unlike standard information coefficient (IC) (full cross-section per date), this measures
    whether density **magnitude** predicts return magnitude among
    triggered events. Direction is already accounted for via sign().

    Only meaningful when density values have magnitude variance
    (not all ±1). The metric returns a not-applicable result when
    variance is absent.

    Args:
        data: Panel with event density and forward return.

    Returns:
        MetricResult with value=Spearman rho, stat=z from Fisher transform.

    Notes:
        ``rho = Spearman(|factor|, signed_car)`` over event rows; Fisher
        z-transform ``z = atanh(rho) * sqrt(N - 3)`` against ``H0: rho =
        0``. Direction is already absorbed into ``signed_car`` so this
        isolates the magnitude-of-density → magnitude-of-return link.

        factrix short-circuits ``"not_applicable_discrete_signal"`` when
        ``|factor|`` lacks variance (e.g. ``{0, ±1}`` events): event-IC
        is undefined without magnitude variation, distinct from "too few
        events".

        Events with a null / NaN ``return_col`` or ``factor_col`` are
        dropped before the correlation (``scipy.stats.spearmanr`` would
        otherwise return NaN for both ρ and p, and a NaN p raises out of
        ``MetricResult``); the count is reported as
        ``metadata["n_events_dropped_non_finite"]``.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.event_quality import event_ic
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = event_ic(panel)
        >>> result.name == ""
        True
    """
    from scipy import stats as sp_stats

    events, n_dropped = _finite_events(data, factor_col, return_col)
    n = len(events)

    sc = _enforce_min_floor(
        event_ic, "event_ic", n, "insufficient_events", axis="events"
    )
    if sc is not None:
        return sc

    if _event_signal_is_discrete(data, factor_col):
        # Signal is discrete {±1}: event_ic is not defined (no magnitude variance).
        # Flagged as "not_applicable" rather than "insufficient" — this is by
        # design, not a shortfall; profiles suppress the field (→ None).
        # Same predicate drives inspect_data's pre-flight verdict (declared via
        # MetricSpec.requires_continuous_magnitude) so the two cannot diverge.
        return _short_circuit_output(
            "event_ic",
            "not_applicable_discrete_signal",
            n_events=n,
        )

    abs_signal = np.abs(events[factor_col].to_numpy())
    signed = _signed_car(events, factor_col, return_col)

    rho, p = sp_stats.spearmanr(abs_signal, signed)
    rho = float(rho)
    p = float(p)

    z = np.arctanh(rho) * np.sqrt(n - 3) if abs(rho) < 1.0 - 1e-10 and n > 3 else 0.0

    return MetricResult(
        p_value=p,
        alternative="two-sided",
        value=rho,
        n_obs=n,
        n_obs_axis="events",
        stat=z,
        metadata={
            "n_events": n,
            "n_events_dropped_non_finite": n_dropped,
            "stat_type": "z",
            "h0": "rho=0",
            "method": "Spearman rank correlation (|density| vs signed_car)",
        },
    )


@metric(
    cell=_EQ_CELL,
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def profit_factor(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Profit factor = sum(gains) / sum(|losses|) across events.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the ratio on the count of non-zero (event) observations.

    Per-event aggregate — no strategy assumptions. A profit factor > 1
    means gross gains exceed gross losses across all events.

    Args:
        data: Panel with event density and forward return.

    Returns:
        MetricResult with value=profit_factor.

    Notes:
        ``PF = sum(signed_car_i * 1{signed_car_i > 0}) /
        |sum(signed_car_i * 1{signed_car_i < 0})|``. Descriptive only;
        no formal H0 (the ratio's sampling distribution lacks a
        clean closed-form null without distributional assumptions).
        ``PF > 1`` means gross gains exceed gross losses across all
        events; the metric ignores per-event variance.

        When gains are positive and losses are zero, factrix returns
        ``inf``: the gross gain/loss ratio is unbounded, not zero. When
        both gains and losses are zero, the ratio is undefined and the
        metric returns ``NaN`` with ``metadata["profit_factor_status"]``.

        Events with a non-finite ``return_col`` / ``factor_col`` are
        dropped up front and counted in
        ``metadata["n_events_dropped_non_finite"]``. They were already
        absent from both sums (``NaN > 0`` and ``NaN < 0`` are both
        False), but used to remain inside ``n_obs`` / ``n_events`` — so
        ``n_wins + n_losses`` silently failed to reconcile with the
        advertised sample size.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.event_quality import profit_factor
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = profit_factor(panel)
        >>> result.name == ""
        True
    """
    events, n_dropped = _finite_events(data, factor_col, return_col)
    n = len(events)

    sc = _enforce_min_floor(
        profit_factor, "profit_factor", n, "insufficient_events", axis="events"
    )
    if sc is not None:
        return sc

    signed = _signed_car(events, factor_col, return_col)

    gains = float(np.sum(signed[signed > 0]))
    losses = float(np.abs(np.sum(signed[signed < 0])))

    no_gains = gains <= EPSILON
    no_losses = losses <= EPSILON
    if no_losses and no_gains:
        pf = float("nan")
        status = "undefined_no_gains_or_losses"
    elif no_losses:
        pf = float("inf")
        status = "unbounded_no_losses"
    else:
        pf = gains / losses
        status = "finite"

    return MetricResult(
        value=pf,
        n_obs=n,
        n_obs_axis="events",
        metadata={
            "total_gains": gains,
            "total_losses": losses,
            "n_events": n,
            "n_events_dropped_non_finite": n_dropped,
            "n_wins": int(np.sum(signed > 0)),
            "n_losses": int(np.sum(signed < 0)),
            "no_gains": no_gains,
            "no_losses": no_losses,
            "profit_factor_status": status,
        },
    )


@metric(
    cell=_EQ_CELL,
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def event_skewness(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Skewness of signed event return distribution.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the descriptive skewness on the count of non-zero (event) observations.

    Positive skew = occasional large gains, frequent small losses
    (desirable for event strategies). Uses scipy's Fisher skewness
    (bias-corrected).

    Also tests H₀: skewness = 0 via D'Agostino's skew test.

    Args:
        data: Panel with event density and forward return.

    Returns:
        MetricResult with value=skewness, stat=z from D'Agostino test.

    Notes:
        ``skew = m_3 / m_2^(3/2)`` (Fisher, bias-corrected via
        ``scipy.stats.skew(bias=False)``); D'Agostino skew test gives
        ``z`` with ``H0: skew = 0`` when ``n_events >= 20``. Below 20 events,
        the test is not produced (``stat=None``) but the descriptive
        skewness is still returned.

        factrix gates the inference branch at ``n_events >= 20`` because the
        D'Agostino-Pearson normal approximation degrades sharply on
        small samples; reporting an unreliable z would invite
        false-positive significance.

        Events with a non-finite ``return_col`` / ``factor_col`` are
        dropped before the moments are taken and counted in
        ``metadata["n_events_dropped_non_finite"]``. A single NaN used to
        make ``skew`` and the ``skewtest`` p NaN, and a NaN ``p_value``
        raises ``ValueError`` out of ``MetricResult`` — the metric crashed
        rather than degrading.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.event_quality import event_skewness
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = event_skewness(panel)
        >>> result.name == ""
        True
    """
    from scipy import stats as sp_stats

    events, n_dropped = _finite_events(data, factor_col, return_col)
    n = len(events)

    sc = _enforce_min_floor(
        event_skewness, "event_skewness", n, "insufficient_events", axis="events"
    )
    if sc is not None:
        return sc

    signed = _signed_car(events, factor_col, return_col)

    skew = float(sp_stats.skew(signed, bias=False))

    if n >= 20:
        z, p = sp_stats.skewtest(signed)
        z = float(z)
        p = float(p)
    else:
        z = None
        p = None

    return MetricResult(
        p_value=p,
        alternative="two-sided" if p is not None else None,
        value=skew,
        n_obs=n,
        n_obs_axis="events",
        stat=z,
        metadata={
            "n_events": n,
            "n_events_dropped_non_finite": n_dropped,
            **(
                {
                    "stat_type": "z",
                    "h0": "skew=0",
                    "method": "D'Agostino skew test",
                }
                if p is not None
                else {}
            ),
        },
    )


@metric(
    cell=_EQ_CELL,
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(),
)
def signal_density(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
) -> MetricResult:
    """Average bars per event (inverse frequency).

    No event-axis sample_threshold is declared (sample_threshold=SampleThreshold()) because the in-body floor is a degeneracy guard (>= 2 events overall, then >= 2 per asset) needed to define the per-asset bars-per-event ratio, not the statistical MIN_EVENTS_HARD floor the other event metrics gate on; like other math-degeneracy guards it stays in the body.

    Answers: "how frequently does this density fire?"

    Computed per-asset as ``total_bars / n_events`` (inverse event
    frequency), then averaged across assets. This is **not** the mean
    of actual inter-event gaps: bars-per-event depends only on counts,
    so clustered events and evenly-spaced events yield the same value.
    See ``clustering_hhi`` for event-date concentration.

    Low density (large gaps) means the density is selective; high
    density (small gaps) means the density fires often — capacity is
    higher but independence may be weaker.

    Args:
        data: Panel with ``date, asset_id, factor``.

    Returns:
        MetricResult with value = mean bars-per-event across assets.

    Notes:
        Per asset ``i``: ``bars_per_event_i = total_bars_i / n_events_i``;
        the headline is the cross-asset mean of this ratio. This is an
        inverse-frequency measure, **not** the mean of inter-event gaps:
        clustered and evenly-spaced events at the same total count map to
        the same value.

        factrix exposes ``clustering_hhi`` for event-date
        concentration; pair the two when independence assumptions matter.

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics.event_quality import signal_density
        >>> panel = fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0)
        >>> result = signal_density(panel)
        >>> result.name == ""
        True
    """
    # ``NaN != 0`` is True in polars, so a non-finite factor would count as an
    # event; use the shared finite predicate.
    events = data.filter(_finite_expr(factor_col) & (pl.col(factor_col) != 0)).sort(
        ["asset_id", "date"]
    )
    n_events = len(events)

    if n_events < 2:
        return _short_circuit_output(
            "signal_density",
            "insufficient_events",
            n_obs=n_events,
            n_obs_axis="events",
            min_required=2,
        )

    # Per-asset: count events and date span
    per_asset = (
        events.group_by("asset_id")
        .agg(
            pl.col("date").count().alias("n"),
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
        )
        .filter(pl.col("n") >= 2)
    )

    if per_asset.is_empty():
        return _short_circuit_output(
            "signal_density",
            "no_asset_has_min_two_events",
            n_obs=n_events,
            n_obs_axis="events",
            min_required_per_asset=2,
        )

    # Total bars per asset (from full panel, not just events)
    bars_per_asset = data.group_by("asset_id").agg(
        pl.col("date").count().alias("total_bars")
    )
    per_asset = per_asset.join(bars_per_asset, on="asset_id", how="left")

    # Mean gap = total_bars / n_events per asset, then average
    per_asset = per_asset.with_columns(
        (pl.col("total_bars") / pl.col("n")).alias("bars_per_event")
    )

    mean_gap = float(per_asset["bars_per_event"].mean())  # type: ignore[arg-type]
    events_per_asset = float(per_asset["n"].mean())  # type: ignore[arg-type]

    return MetricResult(
        value=mean_gap,
        n_obs=n_events,
        n_obs_axis="events",
        metadata={
            "n_events_total": n_events,
            "n_assets_with_events": len(per_asset),
            "mean_events_per_asset": events_per_asset,
            "mean_bars_between_events": mean_gap,
        },
    )
