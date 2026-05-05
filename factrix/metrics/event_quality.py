"""Per-event quality descriptive statistics for event signals.

Aggregation: per-event scalar (hit / IC / skew / density) computed
on `signed_car`, then cross-event aggregation; binomial inference for
hit rate, nonparametric for IC / skewness, descriptive elsewhere.

All metrics operate on the ``signed_car`` (return x sign(factor)) of
individual events. They describe the quality and shape of per-event
outcomes — distinct from significance testing (caar.py) and path
analysis (mfe_mae.py).

Metrics:
    event_hit_rate — fraction of correct-direction events (binomial test)
    event_ic       — signal strength → return correlation (Spearman)
    signal_density — average time gap between events
    profit_factor  — sum(gains) / sum(losses)
    event_skewness — skewness of signed_car distribution

Matrix-row: event_hit_rate, event_ic, profit_factor, event_skewness, signal_density | (*, SPARSE, *, PANEL) | per-event | binomial / nonparametric rank | _binomial_two_sided_p, _significance_marker, _short_circuit_output, _signed_car
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import EPSILON, MIN_EVENTS, MetricOutput
from factrix._stats import (
    _BINOMIAL_EXACT_CUTOFF,
    _binomial_test_method_name,
    _binomial_two_sided_p,
    _significance_marker,
)
from factrix.metrics._helpers import _short_circuit_output, _signed_car


def event_hit_rate(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Fraction of events where signed abnormal return > 0.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=hit_rate, stat=z from binomial test.

    Notes:
        ``hits = sum_i 1{signed_car_i > 0}``, ``rate = hits / N``.
        Two-sided binomial test against ``H0: p = 0.5``: exact below
        ``_BINOMIAL_EXACT_CUTOFF``, normal-approximation z above
        (``z = (hits - N/2) / (sqrt(N) / 2)``).

        factrix publishes ``stat`` consistent with the test branch
        (raw hit count for the exact path, z for the normal path) so an
        exact-binomial p is never paired with a Gaussian z label.
    """
    events = df.filter(pl.col(factor_col) != 0)

    n = len(events)
    if n < MIN_EVENTS:
        return _short_circuit_output(
            "event_hit_rate",
            "insufficient_events",
            n_observed=n,
            min_required=MIN_EVENTS,
        )

    signed = _signed_car(events, factor_col, return_col)
    hits = int(np.sum(signed > 0))
    rate = hits / n
    p = _binomial_two_sided_p(hits, n, p0=0.5)

    # Keep stat / stat_type consistent with the test that produced p.
    if n < _BINOMIAL_EXACT_CUTOFF:
        stat: float = float(hits)
        stat_type = "binomial_hits"
    else:
        stat = (hits - n * 0.5) / (np.sqrt(n) * 0.5)
        stat_type = "z"

    return MetricOutput(
        name="event_hit_rate",
        value=rate,
        stat=stat,
        significance=_significance_marker(p),
        metadata={
            "n_events": n,
            "n_hits": hits,
            "p_value": p,
            "stat_type": stat_type,
            "h0": "p=0.5",
            "method": _binomial_test_method_name(n),
        },
    )


def event_ic(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Signal strength → directional return correlation among events.

    Spearman correlation between ``|factor|`` and ``signed_car``
    (``return × sign(factor)``), computed only on event rows.

    Unlike standard IC (full cross-section per date), this measures
    whether signal **magnitude** predicts return magnitude among
    triggered events. Direction is already accounted for via sign().

    Only meaningful when signal values have magnitude variance
    (not all ±1). Profile auto-skips when variance is absent.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=Spearman rho, stat=z from Fisher transform.

    Notes:
        ``rho = Spearman(|factor|, signed_car)`` over event rows; Fisher
        z-transform ``z = atanh(rho) * sqrt(N - 3)`` against ``H0: rho =
        0``. Direction is already absorbed into ``signed_car`` so this
        isolates the magnitude-of-signal → magnitude-of-return link.

        factrix short-circuits ``"not_applicable_discrete_signal"`` when
        ``|factor|`` lacks variance (e.g. ``{0, ±1}`` events): event-IC
        is undefined without magnitude variation, distinct from "too few
        events".
    """
    from scipy import stats as sp_stats

    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return _short_circuit_output(
            "event_ic",
            "insufficient_events",
            n_observed=n,
            min_required=MIN_EVENTS,
        )

    abs_signal = np.abs(events[factor_col].to_numpy())

    if np.ptp(abs_signal) < EPSILON:
        # Signal is discrete {±1}: event_ic is not defined (no magnitude variance).
        # Flagged as "not_applicable" rather than "insufficient" — this is by
        # design, not a shortfall; profiles suppress the field (→ None).
        return _short_circuit_output(
            "event_ic",
            "not_applicable_discrete_signal",
            n_events=n,
        )

    signed = _signed_car(events, factor_col, return_col)

    rho, p = sp_stats.spearmanr(abs_signal, signed)
    rho = float(rho)
    p = float(p)

    if abs(rho) < 1.0 - 1e-10 and n > 3:
        z = np.arctanh(rho) * np.sqrt(n - 3)
    else:
        z = 0.0

    return MetricOutput(
        name="event_ic",
        value=rho,
        stat=z,
        significance=_significance_marker(p),
        metadata={
            "n_events": n,
            "p_value": p,
            "stat_type": "z",
            "h0": "rho=0",
            "method": "Spearman rank correlation (|signal| vs signed_car)",
        },
    )


def profit_factor(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """sum(positive signed_car) / sum(negative signed_car).

    Per-event aggregate — no strategy assumptions. A profit factor > 1
    means gross gains exceed gross losses across all events.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=profit_factor.

    Notes:
        ``PF = sum(signed_car > 0) / |sum(signed_car < 0)|``. Descriptive
        only; no formal H0 (the ratio's sampling distribution lacks a
        clean closed-form null without distributional assumptions).
        ``PF > 1`` means gross gains exceed gross losses across all
        events; the metric ignores per-event variance.

        factrix returns ``0.0`` rather than infinity when total losses
        are below ``EPSILON`` so downstream aggregators do not propagate
        non-finite floats.
    """
    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return _short_circuit_output(
            "profit_factor",
            "insufficient_events",
            n_observed=n,
            min_required=MIN_EVENTS,
        )

    signed = _signed_car(events, factor_col, return_col)

    gains = float(np.sum(signed[signed > 0]))
    losses = float(np.abs(np.sum(signed[signed < 0])))

    pf = gains / losses if losses > EPSILON else 0.0

    return MetricOutput(
        name="profit_factor",
        value=pf,
        metadata={
            "total_gains": gains,
            "total_losses": losses,
            "n_events": n,
            "n_wins": int(np.sum(signed > 0)),
            "n_losses": int(np.sum(signed < 0)),
        },
    )


def event_skewness(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Skewness of signed_car distribution.

    Positive skew = occasional large gains, frequent small losses
    (desirable for event strategies). Uses scipy's Fisher skewness
    (bias-corrected).

    Also tests H₀: skewness = 0 via D'Agostino's skew test.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=skewness, stat=z from D'Agostino test.

    Notes:
        ``skew = m_3 / m_2^(3/2)`` (Fisher, bias-corrected via
        ``scipy.stats.skew(bias=False)``); D'Agostino skew test gives
        ``z`` with ``H0: skew = 0`` when ``N >= 20``. Below 20 events,
        the test is not produced (``stat=None``) but the descriptive
        skewness is still returned.

        factrix gates the inference branch at ``N >= 20`` because the
        D'Agostino-Pearson normal approximation degrades sharply on
        small samples; reporting an unreliable z would invite
        false-positive significance.
    """
    from scipy import stats as sp_stats

    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return _short_circuit_output(
            "event_skewness",
            "insufficient_events",
            n_observed=n,
            min_required=MIN_EVENTS,
        )

    signed = _signed_car(events, factor_col, return_col)

    skew = float(sp_stats.skew(signed, bias=False))

    if n >= 20:
        z, p = sp_stats.skewtest(signed)
        z = float(z)
        p = float(p)
    else:
        z = None
        p = None

    return MetricOutput(
        name="event_skewness",
        value=skew,
        stat=z,
        significance=_significance_marker(p) if p is not None else None,
        metadata={
            "n_events": n,
            **(
                {
                    "p_value": p,
                    "stat_type": "z",
                    "h0": "skew=0",
                    "method": "D'Agostino skew test",
                }
                if p is not None
                else {}
            ),
        },
    )


def signal_density(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
) -> MetricOutput:
    """Average bars per event (inverse frequency).

    Answers: "how frequently does this signal fire?"

    Computed per-asset as ``total_bars / n_events`` (inverse event
    frequency), then averaged across assets. This is **not** the mean
    of actual inter-event gaps: bars-per-event depends only on counts,
    so clustered events and evenly-spaced events yield the same value.
    See ``clustering_diagnostic`` for event-date concentration.

    Low density (large gaps) means the signal is selective; high
    density (small gaps) means the signal fires often — capacity is
    higher but independence may be weaker.

    Args:
        df: Panel with ``date, asset_id, factor``.

    Returns:
        MetricOutput with value = mean bars-per-event across assets.

    Notes:
        Per asset ``i``: ``bars_per_event_i = total_bars_i / n_events_i``;
        the headline is the cross-asset mean of this ratio. This is an
        inverse-frequency measure, **not** the mean of inter-event gaps:
        clustered and evenly-spaced events at the same total count map to
        the same value.

        factrix exposes ``clustering_diagnostic`` for event-date
        concentration; pair the two when independence assumptions matter.
    """
    events = df.filter(pl.col(factor_col) != 0).sort(["asset_id", "date"])
    n_events = len(events)

    if n_events < 2:
        return _short_circuit_output(
            "signal_density",
            "insufficient_events",
            n_observed=n_events,
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
            n_observed=n_events,
            min_required_per_asset=2,
        )

    # Total bars per asset (from full panel, not just events)
    bars_per_asset = df.group_by("asset_id").agg(
        pl.col("date").count().alias("total_bars")
    )
    per_asset = per_asset.join(bars_per_asset, on="asset_id", how="left")

    # Mean gap = total_bars / n_events per asset, then average
    per_asset = per_asset.with_columns(
        (pl.col("total_bars") / pl.col("n")).alias("bars_per_event")
    )

    mean_gap = float(per_asset["bars_per_event"].mean())
    events_per_asset = float(per_asset["n"].mean())

    return MetricOutput(
        name="signal_density",
        value=mean_gap,
        metadata={
            "n_events_total": n_events,
            "n_assets_with_events": len(per_asset),
            "mean_events_per_asset": events_per_asset,
            "mean_bars_between_events": mean_gap,
        },
    )
