"""Event clustering diagnostic for event signals.

When events cluster on the same dates, the independence assumption
underlying the CAAR t-test is violated, potentially inflating the
test statistic. The Herfindahl-Hirschman Index (HHI) on event dates
quantifies this concentration.

Only meaningful for multi-asset panels (n_assets > 1). For single-asset
event studies, clustering across assets is not applicable.

Notes:
    **Pipeline.** Static cross-section — single HHI computed once over
    the event-date histogram; no time-axis aggregation, no formal H₀
    (descriptive concentration index).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import MIN_EVENTS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _enforce_min_floor, _finite_expr

__all__ = [
    "clustering_hhi",
]


@metric(
    # structure=PANEL (kept, unlike the other event metrics): HHI measures
    # same-date event clustering, which needs a cross-section of assets so that
    # multiple events can share a date. A single name has at most one event per
    # date, so HHI degenerates to 1/n_events (uninformative) — hence this stays
    # multi-asset rather than relaxing to structure=None like caar / bmp_z / etc.
    cell=cell(None, FactorDensity.SPARSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.CS_SNAPSHOT,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def clustering_hhi(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    cluster_window: int = 3,
) -> MetricResult:
    r"""Event clustering Herfindahl index on event dates.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates this descriptive diagnostic on the count of non-zero (event) observations.

    Computes $\mathrm{HHI} = \sum_d s_d^2$ where
    $s_d = (\text{events on date } d) / (\text{total events})$. Herfindahl-Hirschman index (HHI)
    ranges from $1/D$ (uniform) to $1.0$ (all events on one date).

    **Read the three numbers together — HHI alone cannot answer "do my events
    cluster?"** By construction $\mathrm{HHI} \ge 1/D$ over $D$ event dates,
    and it is *invariant to how many assets fire on each date*: 20 assets all
    firing on the same 40 dates — the maximum-cross-sectional-clustering design
    — scores $\mathrm{HHI} = 0.025$ and $\texttt{hhi\_normalized} = 0.000$,
    identical to one asset firing once on each of those dates. Any rule of the
    form "HHI above 0.3 means clustered" is therefore unreachable on a panel
    with more than a handful of event dates, and factrix no longer states one.

    What HHI does measure is whether the events are spread evenly *across the
    dates that have events*. The two axes it misses are reported alongside it:

    - ``events_per_period_mean`` — the cross-sectional axis, and the one the
      Kolari-Pynnönen adjustment acts on. Above 1, events share periods and
      ``bmp_z``'s adjustment (on by default) has something to correct.
    - ``share_events_in_bursts`` — the temporal axis: the share of events whose
      same-asset predecessor sits within ``cluster_window`` periods. HHI is
      blind to this too, scoring 30 events on 30 consecutive periods exactly as
      it scores 30 events spread over 30 years.

    Args:
        data: Panel with ``date, asset_id, factor``.
        cluster_window: Periods defining a temporal burst. An event counts as
            clustered in time when the same asset's previous event sits fewer
            than this many periods earlier, measured on the full panel
            calendar; the share of such events is reported as
            ``share_events_in_bursts``.

    Returns:
        MetricResult with value=HHI, metadata includes the cross-sectional
        (``events_per_period_mean``) and temporal (``share_events_in_bursts``)
        concentration measures HHI itself cannot see.

    Notes:
        $\mathrm{HHI} = \sum_d s_d^2$ where
        $s_d = (\text{events on date } d) / \text{total}$; ranges from
        $1/D$ (uniform across $D$ event dates) to $1.0$ (all events on
        a single date). An event is a **finite non-zero** ``factor_col``
        observation: null and NaN factors are excluded rather than counted
        as events (a bare ``factor != 0`` predicate is True for NaN in
        polars as in numpy).
        ``effective_n_periods`` $= 1 / \mathrm{HHI}$;
        ``hhi_normalized`` $= (\mathrm{HHI} - 1/D) / (1 - 1/D)$ rescales
        to $[0, 1]$ — 0 when every event date carries the same number of
        events, including under perfect cross-sectional clustering.

        ``events_per_period_mean`` is the Kish effective cluster size
        $n_0 = (N - \sum_d n_d^2 / N) / (D - 1)$, the same quantity
        :func:`~factrix.metrics._helpers._estimate_within_date_icc` feeds to
        the Kolari-Pynnönen deflator, so this diagnostic and the correction it
        points at read one number rather than two.

        factrix reports these as descriptive concentration indices — no
        formal $H_0$ — because the corrections they point at are applied
        automatically: ``bmp_z``, ``event_hit_rate`` and ``event_ic`` deflate
        for same-period clustering by default, and every event significance
        test strides its event axis for the temporal kind.

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics.clustering_hhi import clustering_hhi
        >>> panel = fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0)
        >>> result = clustering_hhi(panel)
        >>> result.name == ""
        True
    """
    # ``factor != 0`` is True for a float NaN, so a bare inequality would count
    # a non-finite factor as an event and inflate the date histogram. Events are
    # the *finite* non-zero rows: a null / NaN factor is "no observation", not
    # "an event", so it is excluded from ``n_events`` and from the per-date
    # shares alike.
    events = data.filter(_finite_expr(factor_col) & (pl.col(factor_col) != 0))
    n_events = len(events)

    sc = _enforce_min_floor(
        clustering_hhi, "clustering_hhi", n_events, "insufficient_events", axis="events"
    )
    if sc is not None:
        return sc

    # Count events per date
    per_date = events.group_by("date").agg(pl.len().alias("count"))
    counts = per_date["count"].to_numpy().astype(float)
    shares = counts / counts.sum()

    # The cross-sectional axis HHI is blind to: the Kish effective cluster
    # size, the same n_eff the Kolari-Pynnonen deflator consumes.
    n_total = float(counts.sum())
    n_periods_with_events = len(counts)
    events_per_period_mean = (
        float((n_total - (counts**2).sum() / n_total) / (n_periods_with_events - 1))
        if n_periods_with_events > 1
        else float(n_total)
    )

    # The temporal axis HHI is blind to: same-asset events inside one window.
    # Gaps are measured on the FULL panel calendar (the dense rank over every
    # date, event or not), so a gap of k means k periods elapsed, not k events.
    calendar = (
        data.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.int_range(pl.len()).alias("_ordinal"))
    )
    ordered = (
        events.join(calendar, on="date", how="left")
        .sort(["asset_id", "_ordinal"])
        .with_columns(
            (pl.col("_ordinal") - pl.col("_ordinal").shift(1).over("asset_id")).alias(
                "_gap"
            )
        )
    )
    in_burst = ordered.filter(
        pl.col("_gap").is_not_null() & (pl.col("_gap") < cluster_window)
    ).height
    share_events_in_bursts = in_burst / n_events if n_events else 0.0

    hhi = float(np.sum(shares**2))

    # Effective number of independent dates = 1/HHI
    effective_n = 1.0 / hhi if hhi > 0 else 0.0

    n_dates = len(per_date)
    # Normalized HHI: (HHI - 1/D) / (1 - 1/D), ranges 0 to 1
    hhi_min = 1.0 / n_dates if n_dates > 0 else 0.0
    hhi_normalized = (hhi - hhi_min) / (1.0 - hhi_min) if n_dates > 1 else 0.0

    return MetricResult(
        value=hhi,
        n_obs=n_events,
        n_obs_axis="events",
        metadata={
            "n_events": n_events,
            "n_event_periods": n_dates,
            "effective_n_periods": effective_n,
            "hhi_normalized": hhi_normalized,
            "events_per_period_mean": events_per_period_mean,
            "max_events_per_period": int(counts.max()),
            "share_events_in_bursts": share_events_in_bursts,
            "cluster_window": cluster_window,
        },
    )
