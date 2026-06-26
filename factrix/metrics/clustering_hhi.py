"""Event clustering diagnostic for event signals.

When events cluster on the same dates, the independence assumption
underlying the CAAR t-test is violated, potentially inflating the
test statistic. The Herfindahl-Hirschman Index (HHI) on event dates
quantifies this concentration.

Only meaningful for multi-asset panels (N > 1). For single-asset
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
from factrix.metrics._helpers import _enforce_min_floor

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

    High HHI → events concentrate in few dates → cross-event independence
    assumption violated → CAAR $t$-stat may be inflated.

    Args:
        data: Panel with ``date, asset_id, factor``.
        cluster_window: Not used in HHI calculation but preserved for
            future block-bootstrap clustering adjustment.

    Returns:
        MetricResult with value=HHI, metadata includes effective_n_periods
        and concentration ratio.

    Notes:
        $\mathrm{HHI} = \sum_d s_d^2$ where
        $s_d = (\text{events on date } d) / \text{total}$; ranges from
        $1/D$ (uniform across $D$ event dates) to $1.0$ (all events on
        a single date).
        ``effective_n_periods`` $= 1 / \mathrm{HHI}$;
        ``hhi_normalized`` $= (\mathrm{HHI} - 1/D) / (1 - 1/D)$ rescales
        to $[0, 1]$.

        factrix reports HHI as a descriptive concentration index — no
        formal $H_0$ — because the natural follow-up correction
        (cross-sectional dependence in CAAR / BMP) is delegated to
        ``bmp_z(kolari_pynnonen_adjust=True)``.

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics.clustering_hhi import clustering_hhi
        >>> panel = fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0)
        >>> result = clustering_hhi(panel)
        >>> result.name == ""
        True
    """
    events = data.filter(pl.col(factor_col) != 0)
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
            "cluster_window": cluster_window,
        },
    )
