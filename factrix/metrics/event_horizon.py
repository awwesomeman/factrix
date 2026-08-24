"""Multi-horizon event analysis — how does the density behave across time?

Answers:
    - Is there pre-event leakage? (T-6..T-1 should be ~0)
    - At which horizon is the density strongest?
    - Does the alpha persist or decay quickly?

Metrics:
    compute_event_returns — per-event, per-offset raw return data
    event_around_return   — return profile summary at each offset

Notes:
    **Pipeline.** Per-event return profile across `k` offsets
    (per-event step); a descriptive curve only — no hypothesis test, so
    ``p_value`` is ``None`` and the per-horizon ``hit_rate`` in
    ``per_offset`` is a raw fraction, not a tested statistic.
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
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _short_circuit_output
from factrix.metrics._primitives import compute_event_returns

__all__ = [
    "event_around_return",
]

# structure=None (event-axis): event-window returns aggregate over events, so a
# single name with enough events is valid. Density stays SPARSE; the event floor
# guards thin samples.
_EH_CELL = cell(None, FactorDensity.SPARSE, structure=None)


@metric(
    cell=_EH_CELL,
    aggregation=Aggregation.EVENT_TIME,
    slice_boundary_sensitive=True,
    sample_threshold=SampleThreshold(),
)
def event_around_return(
    data: pl.DataFrame,
    *,
    offsets: list[int] | None = None,
    factor_col: str = "factor",
    price_col: str = "price",
) -> MetricResult:
    """Return profile at multiple offsets around event date.

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because this is a multi-horizon summary metric whose available offsets and event counts are factor-context-dependent.

    Summarizes per-offset: mean, median, p25, p75, hit_rate, n.

    The primary value is the pre-event leakage score: the mean over
    pre-event offsets of ``|cross-event mean single-bar return|``
    (should be ~0). It is a mean of per-offset means, not a pre-event
    CAR — see Notes. High leakage → density may be reactive, not
    predictive.

    Args:
        data: Panel with ``date, asset_id, factor, price``.
        offsets: Defaults to ``[-6, -3, -1, 1, 6, 12, 24]``.

    Returns:
        MetricResult with per-offset stats in metadata. When price data is
        unavailable, returns a short-circuit MetricResult (``value=NaN``,
        ``metadata["reason"]="no_price_data"``) so all metrics share a
        single return contract.

    Notes:
        For each offset ``k``: ``mean, median, p25, p75, hit_rate``
        across events with valid ``signed_return``. The headline
        ``value = mean_{k < 0} |mean_k|`` summarises pre-event leakage —
        a healthy density has flat pre-event means.

        factrix uses ``|mean|`` rather than absolute returns to avoid
        rewarding offsets where positive and negative pre-event drifts
        cancel — leakage with consistent direction would be missed by
        ``mean(|return|)``.

        **What the leakage headline is (and is not).** It is the mean of
        the *absolute values of per-offset cross-event means* of
        **single-bar** pre-event returns. It is **not** a pre-event CAR
        and not a cumulative run-up: ``compute_event_returns`` defines
        each ``k <= 0`` offset as the one-bar return at that lag
        (``prices[idx+k] / prices[idx+k-1] - 1``), deliberately, so a
        single leaking bar stays localised instead of being smeared across
        every longer lag. Consequences to keep in mind when reading the
        number:

        - Its scale is one bar's return regardless of how far back the
          offsets reach, so it does not grow with the pre-event window and
          is not comparable to a post-event cumulative offset (those *are*
          cumulative from a common ``idx + 1`` entry).
        - Averaging ``|mean_k|`` over a sparse offset grid (default
          ``-6, -3, -1``) samples three isolated bars; it neither covers
          nor sums the bars between them. A clean score means "these
          sampled bars show no directional drift", not "the whole
          pre-event window is flat". Pass a dense negative ``offsets``
          list to actually sweep the window.
        - Offsets with fewer than 5 events contribute no ``mean`` and are
          simply skipped in the average (they appear in ``per_offset``
          with ``mean=None``), so the headline can rest on fewer offsets
          than were requested.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.event_horizon import event_around_return
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = event_around_return(panel)
        >>> result.name == ""
        True
    """
    if offsets is None:
        offsets = [-6, -3, -1, 1, 6, 12, 24]

    event_rets = compute_event_returns(
        data,
        offsets=offsets,
        factor_col=factor_col,
        price_col=price_col,
    )

    if event_rets.is_empty():
        return _short_circuit_output(
            "event_around_return",
            "no_price_data",
            descriptive=True,
            per_offset={},
        )

    per_offset: dict[int, dict] = {}
    pre_leakage_vals: list[float] = []

    for k in offsets:
        subset = event_rets.filter(pl.col("offset") == k)["signed_return"]
        n = len(subset)
        if n < 5:
            per_offset[k] = {"mean": None, "n": n}
            continue

        arr = subset.to_numpy()
        mean_v = float(np.mean(arr))
        per_offset[k] = {
            "mean": mean_v,
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "hit_rate": float(np.mean(arr > 0)),
            "n": n,
        }

        if k < 0:
            pre_leakage_vals.append(abs(mean_v))

    # Primary value: pre-event leakage (mean of |pre-event returns|)
    leakage = float(np.mean(pre_leakage_vals)) if pre_leakage_vals else 0.0

    return MetricResult(
        p_value=None,
        value=leakage,
        metadata={
            "per_offset": per_offset,
            "interpretation": (
                "value = mean over pre-event offsets of |mean single-bar "
                "signed return| (not a cumulative pre-event CAR); "
                "high = potential leakage"
            ),
        },
    )
