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
from factrix._types import DDOF, EPSILON
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _densify_on_period_grid,
    _short_circuit_output,
    _warn_ragged_event_grid,
)
from factrix.metrics._primitives import compute_event_returns

__all__ = [
    "event_around_return",
]

# structure=None (event-axis): event-window returns aggregate over events, so a
# single name with enough events is valid. Density stays SPARSE; the event floor
# guards thin samples.
_EH_CELL = cell(None, FactorDensity.SPARSE, structure=None)


def _unconditional_bar_return(data: pl.DataFrame, price_col: str) -> float:
    """Mean single-bar return across the whole panel, per asset then pooled.

    The pre-event offsets are single-bar returns, so an asset that drifts up
    0.1% a bar has pre-event means of 0.1% with no information leakage at all.
    Subtracting this baseline makes the leakage score measure what its name
    says. Returns ``0.0`` when it cannot be computed, which leaves the score
    exactly as it was.

    The bar is one step on the panel's period grid, matching the offsets it is
    subtracted from: on a ragged panel a step across an asset's missing periods
    is not a one-period return and drops out rather than inflating the
    baseline.
    """
    if price_col not in data.columns or "asset_id" not in data.columns:
        return 0.0
    dense, _ = _densify_on_period_grid(data)
    rets = (
        dense.with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1).over("asset_id") - 1).alias(
                "_bar_ret"
            )
        )["_bar_ret"]
        .drop_nulls()
        .drop_nans()
    )
    return float(rets.mean()) if len(rets) else 0.0  # type: ignore[arg-type]


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
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Return profile at multiple offsets around event date.

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because this is a multi-horizon summary metric whose available offsets and event counts are factor-context-dependent.

    Summarizes per-offset: mean, median, p25, p75, hit_rate, n.

    Offsets are steps on the panel's period grid (see
    :func:`~factrix.metrics._primitives.compute_event_returns`), as is the
    unconditional bar return the pre-event means are taken against.

    The primary value is the pre-event leakage score: the mean over
    pre-event offsets of ``|cross-event mean single-bar excess return|``,
    where "excess" is over the panel's unconditional bar return, signed like
    the return it is subtracted from (``sign(factor) * baseline``). It is a
    mean of per-offset means, not a pre-event CAR — see Notes. High leakage →
    density may be reactive, not predictive; read it against
    ``metadata["leakage_null_scale"]``, which is what the score is worth under
    no leakage at all.

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

        **There is no zero to compare against, and the score is not a test.**
        $E|\bar x| \approx 0.8\,\sigma/\sqrt{n} > 0$ under a true null, so the
        headline is strictly positive by construction and *shrinks as the event
        count grows*: the same clean factor scores lower on more events. The
        doc's old "should be ~0" target was therefore unattainable, and two
        further things used to be confounded into it. Unconditional drift is
        now removed — every offset is an excess over
        ``sign(factor) * metadata["baseline_bar_return"]``, so a trending
        asset no longer reads as leaky on either side (the returns are
        signed, so the baseline must be too: an earlier version subtracted
        the unsigned drift and scored a short-signed event at twice the
        drift, $t \approx -400$ on a 0.2%-per-period trend) — and the null
        scale itself is published as
        ``metadata["leakage_null_scale"]``. Per offset, ``se`` and ``t`` say
        how much of a mean is signal. Nothing here is a hypothesis test:
        ``p_value`` stays ``None`` (see the warning on the doc page for why).

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
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=0),
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
            n_obs=0,
            n_obs_axis="events",
            per_offset={},
        )

    per_offset: dict[int, dict] = {}
    pre_leakage_vals: list[float] = []
    pre_leakage_se: list[float] = []
    # Unconditional mean single-bar return across the whole panel: the baseline
    # every offset is measured against.
    baseline = _unconditional_bar_return(data, price_col)

    for k in offsets:
        subset = event_rets.filter(pl.col("offset") == k)
        n = len(subset)
        if n < 5:
            per_offset[k] = {"mean": None, "n": n}
            continue

        arr = subset["signed_return"].to_numpy()
        # Excess over the unconditional bar return: a trending asset's bars are
        # non-zero on average whether or not an event is coming, and that drift
        # entered the leakage score directly. The return is signed, so the
        # baseline is signed the same way: a short event's bar carries -mu,
        # and subtracting +mu from it scored the drift twice over.
        excess = arr - baseline * subset["sign"].to_numpy()
        mean_v = float(np.mean(excess))
        se = float(np.std(excess, ddof=DDOF) / np.sqrt(n)) if n > 1 else float("nan")
        per_offset[k] = {
            "mean": mean_v,
            "se": se,
            # The scale the score has to be read against: |mean| of a true null
            # is about 0.8 * se, never zero.
            "t": float(mean_v / se) if se > EPSILON else None,
            "median": float(np.median(excess)),
            "p25": float(np.percentile(excess, 25)),
            "p75": float(np.percentile(excess, 75)),
            "hit_rate": float(np.mean(excess > 0)),
            "n": n,
        }

        if k < 0:
            pre_leakage_vals.append(abs(mean_v))
            pre_leakage_se.append(se)

    # Primary value: pre-event leakage (mean of |pre-event excess returns|).
    # NaN, not 0.0, when no pre-event offset qualified: 0.0 is the *best*
    # possible leakage score and was being reported for a quantity that was
    # never computed (every negative offset below the 5-event floor, or a
    # caller passing only positive offsets).
    if pre_leakage_vals:
        leakage = float(np.mean(pre_leakage_vals))
        leakage_reason = None
    else:
        leakage = float("nan")
        leakage_reason = "no_pre_event_offset_with_enough_events"

    # Sample size on the event axis: the distinct (date, asset) events behind
    # the curve. Every sibling event metric stamps n_obs / n_obs_axis, and
    # n_obs is the library's single source of truth for sample size, so a null
    # here broke the uniform column in to_frame() for no reason — the count is
    # already in per_offset. One event contributes one row per offset, so the
    # row count is not it; the distinct event count is.
    n_events = event_rets.select("date", "asset_id").n_unique()

    warning_codes: list[str] = []
    _warn_ragged_event_grid(
        "event_around_return", data, warning_codes, expected_warnings=expected_warnings
    )

    return MetricResult(
        p_value=None,
        value=leakage,
        n_obs=n_events,
        n_obs_axis="events",
        warning_codes=tuple(warning_codes),
        metadata={
            "n_events": n_events,
            "per_offset": per_offset,
            "baseline_bar_return": baseline,
            # The null scale of the headline: E|x̄| ≈ 0.8 σ/√n > 0 under no
            # leakage at all, so the score shrinks as events accumulate and
            # cannot be read against a fixed "should be ~0" target.
            "leakage_null_scale": (
                float(np.mean(pre_leakage_se) * np.sqrt(2.0 / np.pi))
                if pre_leakage_se and np.isfinite(pre_leakage_se).all()
                else None
            ),
            **({"reason": leakage_reason} if leakage_reason else {}),
            "interpretation": (
                "value = mean over pre-event offsets of |mean single-bar "
                "signed return| (not a cumulative pre-event CAR); "
                "high = potential leakage"
            ),
        },
    )
