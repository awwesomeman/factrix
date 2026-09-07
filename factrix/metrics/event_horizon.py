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
from factrix._data_input import DataInput, _coerce_price_data
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import DDOF, EPSILON
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _densify_on_period_grid,
    _finite_expr,
    _short_circuit_output,
    _warn_ragged_event_grid,
)

# Re-exported so the primitive and its metric verb share one import path, the
# way ``mfe_mae`` re-exports ``compute_mfe_mae``. The redundant alias is the
# explicit re-export form: nothing in this module body calls it, so a plain
# import would read as dead. It stays out of ``__all__`` on purpose — the
# module's declared surface is the metric verb, and the docs page documents the
# primitive in prose rather than rendering its API a second time.
from factrix.metrics._primitives import compute_event_returns as compute_event_returns
from factrix.metrics._primitives._event_returns import (
    _compute_event_returns_with_audit,
)

__all__ = [
    "event_around_return",
]

# structure=None (event-axis): event-window returns aggregate over events, so a
# single name with enough events is valid. Density stays SPARSE; the event floor
# guards thin samples.
_EH_CELL = cell(None, FactorDensity.SPARSE, structure=None)
_BENCHMARK_WEIGHTING = "event_weighted_mean_signed_per_asset"


def _unconditional_bar_returns(
    data: pl.DataFrame, price_col: str
) -> tuple[pl.DataFrame | None, int]:
    """Mean single-period return for every asset that can form one.

    An asset that drifts up 0.1% a period has offset means of 0.1% with no
    information leakage at all. Subtracting this baseline makes the leakage
    score measure what its name says. Returns ``None`` when it cannot be
    computed. A non-finite or non-positive observed price invalidates the
    baseline rather than being silently dropped: otherwise a contaminated
    denominator can manufacture an infinite baseline and a finite-looking hit
    rate.

    The period is one step on the panel's period grid, matching the offsets
    the baseline is subtracted from: on a ragged panel a step across an
    asset's missing periods is not a one-period return and drops out rather
    than inflating the baseline.

    The caller subtracts the matching asset's mean from every event. It may
    still average this table with equal asset weight for a compact panel-level
    diagnostic, but that scalar is not used to price an event: pooling first
    creates an event-composition residue when assets have different drifts,
    and compounding the pool creates an additional Jensen gap at long offsets.

    Returns:
        The per-asset baselines and the count of invalid observed prices.
        ``None`` means the panel is invalid; an empty frame means that no asset
        could form a finite adjacent-period return.
    """
    if price_col not in data.columns or "asset_id" not in data.columns:
        return None, 0

    price = pl.col(price_col).cast(pl.Float64, strict=False)
    invalid_price = pl.col(price_col).is_not_null() & (
        ~_finite_expr(price_col) | (price <= EPSILON).fill_null(False)
    )
    n_invalid_prices = data.filter(invalid_price).height
    if n_invalid_prices:
        return None, n_invalid_prices

    dense, _ = _densify_on_period_grid(data)
    per_asset = (
        dense.with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1).over("asset_id") - 1).alias(
                "_bar_ret"
            )
        )
        .filter(_finite_expr("_bar_ret"))
        .group_by("asset_id")
        .agg(pl.col("_bar_ret").mean().alias("_baseline_bar_return"))
        .sort("asset_id")
    )
    return per_asset, 0


def _horizon_benchmark(baseline: np.ndarray, offset: int) -> np.ndarray:
    """Per-asset baselines over the periods ``offset`` actually spans.

    Offset ``k > 0`` is a simple return from ``t+1`` to ``t+1+k``, i.e. a
    ratio of two prices ``k`` periods apart, so its benchmark is the same
    ratio built from that event asset's mean single-period return:
    ``(1 + mu_asset)**k - 1``.
    Subtracting one period of ``mu`` from it left roughly ``k - 1`` periods
    of drift in the answer, which reads as event alpha on a panel that
    merely trends.

    Offsets at or below zero are single-period returns and keep the
    single-period benchmark.
    """
    if offset <= 0:
        return baseline
    return np.power(1.0 + baseline, offset) - 1.0


@metric(
    cell=_EH_CELL,
    aggregation=Aggregation.EVENT_TIME,
    slice_boundary_sensitive=True,
    sample_threshold=SampleThreshold(),
)
def event_around_return(
    data: pl.DataFrame,
    *,
    price_data: DataInput | None = None,
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
    per-asset unconditional bar return the pre-event observations are taken
    against.

    The primary value is the pre-event leakage score: the mean over
    pre-event offsets of ``|cross-event mean single-bar excess return|``,
    where "excess" is over each event asset's unconditional bar return,
    signed like the return it is subtracted from
    (``sign(factor) * asset_baseline``). It is a
    mean of per-offset means, not a pre-event CAR — see Notes. High leakage →
    density may be reactive, not predictive; read it against
    ``metadata["leakage_null_scale"]``, which is what the score is worth under
    no leakage at all.

    Args:
        data: Panel with ``date, asset_id, factor, price``.
        price_data: Optional complete ``date, asset_id, price`` panel. Events
            come from ``data``; offsets and the per-asset unconditional
            bar-return baselines use this full price grid. Pass the raw panel when
            ``data`` came from ``compute_forward_return``.
        offsets: Defaults to ``[-6, -3, -1, 1, 6, 12, 24]``.

    Returns:
        MetricResult with per-offset stats and audit counts in metadata.
        Every published ``per_offset[k]`` includes ``eligible``, ``computed``,
        ``censored``, and ``censor_reasons``; ``n`` remains an alias for the
        computed count. When price data is unavailable, the censor audit
        explains why no path was computed. When the unconditional baseline
        cannot be computed from valid positive prices, the whole curve is
        withdrawn and ``per_offset`` is empty rather than publishing counts
        from the discarded raw-path computation. Both cases return a
        short-circuit MetricResult (``value=NaN``) so all metrics share a
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
        ``sign(factor) * metadata["baseline_bar_return_by_asset"][asset_id]``,
        compounded over positive offsets, so neither heterogeneous asset drift
        nor unequal event counts manufacture alpha. The panel-level
        ``baseline_bar_return`` is only an equal-asset-weighted diagnostic.
        The returns are signed, so each asset baseline must be too: an earlier
        version subtracted the unsigned drift and scored a short-signed event
        at twice the drift ($t \approx -400$ on a 0.2%-per-period trend). The
        null scale itself is published as
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
    resolved_offsets = [-6, -3, -1, 1, 6, 12, 24] if offsets is None else offsets
    resolved_prices = _coerce_price_data(
        price_data,
        data=data,
        func_name="event_around_return",
        price_col=price_col,
    )
    event_rets, offset_audit = _compute_event_returns_with_audit(
        data,
        price_data=resolved_prices,
        offsets=resolved_offsets,
        factor_col=factor_col,
        price_col=price_col,
    )
    per_offset: dict[int, dict[str, object]] = {
        offset: {**audit, "mean": None, "n": int(audit["computed"])}
        for offset, audit in offset_audit.items()
    }
    n_events_eligible = (
        int(next(iter(offset_audit.values()))["eligible"]) if offset_audit else 0
    )

    if event_rets.is_empty():
        only_missing_price = bool(offset_audit) and all(
            audit["censor_reasons"] == {"missing_price_column": n_events_eligible}
            for audit in offset_audit.values()
        )
        return _short_circuit_output(
            "event_around_return",
            "no_price_data" if only_missing_price else "no_computed_event_offsets",
            descriptive=True,
            n_obs=0,
            n_obs_axis="events",
            n_events=0,
            n_events_eligible=n_events_eligible,
            per_offset=per_offset,
        )

    n_raw_events = event_rets.select("date", "asset_id").n_unique()
    # Each event is measured against its own asset's unconditional mean
    # single-bar return. A pooled scalar leaves an event-composition residue
    # even at one-bar offsets and a Jensen gap after compounding at long ones.
    # A bad observed price still invalidates the whole baseline: dropping only
    # its affected return would silently change the estimand and still publish
    # a finite-looking hit rate.
    path_data = data if resolved_prices is None else resolved_prices
    baselines, n_invalid_prices = _unconditional_bar_returns(path_data, price_col)
    if baselines is None or baselines.is_empty():
        reason = (
            "invalid_price_data" if n_invalid_prices else "no_finite_baseline_returns"
        )
        return _short_circuit_output(
            "event_around_return",
            reason,
            descriptive=True,
            n_obs=n_raw_events,
            n_obs_axis="events",
            n_events=n_raw_events,
            n_events_eligible=n_events_eligible,
            n_invalid_prices=n_invalid_prices,
            baseline_bar_return=None,
            baseline_bar_return_by_asset={},
            n_assets_in_baseline=0,
            benchmark_weighting=_BENCHMARK_WEIGHTING,
            # The raw paths ran, so n_events remains reportable, but no
            # per-offset curve survived the invalid baseline. Publishing its
            # intermediate counts under the final curve key makes discarded
            # work look like a partially available result.
            per_offset={},
        )

    baseline = float(baselines["_baseline_bar_return"].mean())  # type: ignore[arg-type]
    baseline_by_asset = dict(
        baselines.select("asset_id", "_baseline_bar_return").iter_rows()
    )
    n_assets_in_baseline = baselines.height

    # A ragged asset can form an event path without ever having two adjacent
    # panel periods from which to estimate a one-bar drift. Such a path has no
    # per-asset excess-return estimand, so account for it as censored instead of
    # falling back to another asset's or the panel's drift.
    event_rets = event_rets.join(baselines, on="asset_id", how="left")
    missing_baseline = event_rets.filter(pl.col("_baseline_bar_return").is_null())
    for offset, count in missing_baseline.group_by("offset").len().iter_rows():
        audit = offset_audit[int(offset)]
        n_missing = int(count)
        audit["computed"] -= n_missing
        audit["censored"] += n_missing
        reasons = audit["censor_reasons"]
        reasons["missing_asset_baseline"] = (
            reasons.get("missing_asset_baseline", 0) + n_missing
        )
    event_rets = event_rets.drop_nulls("_baseline_bar_return")
    per_offset = {
        offset: {**audit, "mean": None, "n": int(audit["computed"])}
        for offset, audit in offset_audit.items()
    }
    n_events = event_rets.select("date", "asset_id").n_unique()
    if event_rets.is_empty():
        return _short_circuit_output(
            "event_around_return",
            "no_event_asset_baselines",
            descriptive=True,
            n_obs=0,
            n_obs_axis="events",
            n_events=0,
            n_events_eligible=n_events_eligible,
            n_invalid_prices=0,
            baseline_bar_return=baseline,
            baseline_bar_return_by_asset=baseline_by_asset,
            n_assets_in_baseline=n_assets_in_baseline,
            benchmark_weighting=_BENCHMARK_WEIGHTING,
            per_offset=per_offset,
        )

    pre_leakage_vals: list[float] = []
    pre_leakage_se: list[float] = []

    for k in resolved_offsets:
        subset = event_rets.filter(pl.col("offset") == k)
        n = len(subset)
        if n < 5:
            continue

        arr = subset["signed_return"].to_numpy()
        # Excess over this event's own asset drift, compounded over this
        # offset's horizon. The return is signed, so the benchmark is signed
        # the same way: a short event's period carries -mu, and subtracting
        # +mu from it would score the drift twice over.
        unsigned_benchmarks = _horizon_benchmark(
            subset["_baseline_bar_return"].to_numpy(), k
        )
        signed_benchmarks = unsigned_benchmarks * subset["sign"].to_numpy()
        excess = arr - signed_benchmarks
        # Linearity makes this the exact scalar subtracted from the raw signed
        # *mean*. Individual event baselines can differ, so it is deliberately
        # not described as the benchmark behind the quantiles.
        benchmark = float(np.mean(signed_benchmarks))
        mean_v = float(np.mean(excess))
        se = float(np.std(excess, ddof=DDOF) / np.sqrt(n)) if n > 1 else float("nan")
        per_offset[k] = {
            **offset_audit[k],
            # Event-weighted mean of the signed, per-asset benchmarks actually
            # subtracted. Therefore raw signed mean == mean + benchmark even
            # when event composition and factor direction vary.
            "benchmark": benchmark,
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
    warning_codes: list[str] = []
    _warn_ragged_event_grid(
        "event_around_return",
        path_data,
        warning_codes,
        expected_warnings=expected_warnings,
    )

    return MetricResult(
        p_value=None,
        value=leakage,
        n_obs=n_events,
        n_obs_axis="events",
        warning_codes=tuple(warning_codes),
        metadata={
            "n_events": n_events,
            "n_events_eligible": n_events_eligible,
            "per_offset": per_offset,
            "baseline_bar_return": baseline,
            # Compact panel-level diagnostic only: events are priced from the
            # matching per-asset mapping below, never from this pooled scalar.
            "baseline_bar_return_by_asset": baseline_by_asset,
            "n_assets_in_baseline": n_assets_in_baseline,
            "benchmark_weighting": _BENCHMARK_WEIGHTING,
            # The null scale of the headline: E|x̄| ≈ 0.8 σ/√n > 0 under no
            # leakage at all, so the score shrinks as events accumulate and
            # cannot be read against a fixed "should be ~0" target.
            "leakage_null_scale": (
                float(np.mean(pre_leakage_se) * np.sqrt(2.0 / np.pi))
                if pre_leakage_se and np.isfinite(pre_leakage_se).all()
                else None
            ),
            **({"reason": leakage_reason} if leakage_reason else {}),
        },
    )
