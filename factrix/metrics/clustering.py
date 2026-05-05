"""Event clustering diagnostic for event signals.

Aggregation: static cross-section — single HHI computed once over the
event-date histogram; no time-axis aggregation, no formal H₀
(descriptive concentration index).

When events cluster on the same dates, the independence assumption
underlying the CAAR t-test is violated, potentially inflating the
test statistic. The Herfindahl-Hirschman Index (HHI) on event dates
quantifies this concentration.

Only meaningful for multi-asset panels (N > 1). For single-asset
event studies, clustering across assets is not applicable.

Matrix-row: clustering_diagnostic | (*, SPARSE, *, PANEL) | static CS | no formal H₀ | _short_circuit_output
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import MIN_EVENTS, MetricOutput
from factrix.metrics._helpers import _short_circuit_output


def clustering_diagnostic(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    cluster_window: int = 3,
) -> MetricOutput:
    r"""Event clustering Herfindahl index on event dates.

    Computes $\mathrm{HHI} = \sum_d s_d^2$ where
    $s_d = (\text{events on date } d) / (\text{total events})$. HHI
    ranges from $1/D$ (uniform) to $1.0$ (all events on one date).

    High HHI → events concentrate in few dates → cross-event independence
    assumption violated → CAAR $t$-stat may be inflated.

    Args:
        df: Panel with ``date, asset_id, factor``.
        cluster_window: Not used in HHI calculation but preserved for
            future block-bootstrap clustering adjustment.

    Returns:
        MetricOutput with value=HHI, metadata includes effective_n_dates
        and concentration ratio.

    Notes:
        $\mathrm{HHI} = \sum_d s_d^2$ where
        $s_d = (\text{events on date } d) / \text{total}$; ranges from
        $1/D$ (uniform across $D$ event dates) to $1.0$ (all events on
        a single date).
        ``effective_n_dates`` $= 1 / \mathrm{HHI}$;
        ``hhi_normalized`` $= (\mathrm{HHI} - 1/D) / (1 - 1/D)$ rescales
        to $[0, 1]$.

        factrix reports HHI as a descriptive concentration index — no
        formal $H_0$ — because the natural follow-up correction
        (cross-sectional dependence in CAAR / BMP) is delegated to
        ``bmp_test(kolari_pynnonen_adjust=True)``.
    """
    events = df.filter(pl.col(factor_col) != 0)
    n_events = len(events)

    if n_events < MIN_EVENTS:
        return _short_circuit_output(
            "clustering_hhi",
            "insufficient_events",
            n_observed=n_events,
            min_required=MIN_EVENTS,
        )

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

    return MetricOutput(
        name="clustering_hhi",
        value=hhi,
        metadata={
            "n_events": n_events,
            "n_event_dates": n_dates,
            "effective_n_dates": effective_n,
            "hhi_normalized": hhi_normalized,
            "cluster_window": cluster_window,
        },
    )
