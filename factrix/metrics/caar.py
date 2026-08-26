r"""CAAR (Cumulative Average Abnormal Return) significance tests.

Tests $H_0$: event abnormal return = 0, using two complementary methods:
    compute_caar — per-event-period weighted abnormal return series
    caar         — CAAR t-test (parametric, non-overlapping sampling)
    bmp_z     — BMP standardized AR test (robust to event-induced variance)

Notes:
    `caar` and `bmp_z` are complementary inferential tests on the
    per-event-period abnormal-return series. `caar` is the parametric
    cross-event $t$-test; `bmp_z` is the standardized-AR $z$-test that
    is robust to event-induced variance.

References:
    - [MacKinlay (1997)][mackinlay-1997], "Event Studies in Economics
      and Finance."
    - [Boehmer, Musumeci & Poulsen (1991)][boehmer-musumeci-poulsen-1991],
      "Event-study Methodology Under Conditions of Event-induced
      Variance."
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    FactorDensity,
    InputShape,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import (
    _calc_t_stat,
    _p_value_from_t,
    _p_value_from_z,
)
from factrix._types import (
    DDOF,
    EPSILON,
    MIN_EVENTS_HARD,
    MIN_EVENTS_WARN,
)
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _degenerate_test_fields,
    _enforce_min_floor,
    _enforce_scaled_floor,
    _estimate_within_date_icc,
    _event_sample_threshold,
    _kp_cluster_scale,
    _sample_event_spaced,
    _sample_events_non_overlapping,
    _scaled_min_periods,
    _short_circuit_output,
    _warn_below_scaled_floor,
    _warn_event_window_overlap,
)
from factrix.metrics._metric_capabilities import per_date_series_rename
from factrix.metrics._primitives import compute_caar

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "compute_caar",
    "caar",
    "bmp_z",
]

# structure=None (event-axis): caar/bmp_z aggregate over the event cross-section
# (event periods / events), not the asset cross-section, so they run on single-asset
# multi-event data too. Density stays SPARSE — the event-shaped signal — and the
# event-count floor guards thin samples.
_CAAR_CELL = cell(None, FactorDensity.SPARSE, structure=None)

# Slice-test contract: CAAR is event-driven; the
# cross-section is the event sample, not a bucketed asset universe,
# so slice tests skip the `n_groups` downscale step. Minimum event
# count for the cross-event t-test (FEW_EVENTS threshold)
# lives in the procedure short-circuit and is parallel to (not
# exposed via) this attribute.
min_assets_per_group: int | None = None
per_date_series = per_date_series_rename("caar")


def _caar_sample_threshold(self: MetricBase) -> SampleThreshold:
    """Dynamic event floor for ``caar``: the raw event-period count scales with
    ``forward_periods`` because the t-test runs on a non-overlap subsample
    (stride ``forward_periods``). Delegates to the same ``_scaled_min_periods``
    the in-body short-circuit reads, so pre-flight and run-time floors agree.

    Unlike the sibling event tests (:func:`_event_sample_threshold`), the *hard*
    floor scales too: ``caar`` consumes an already-aggregated event-period
    series, so its pre-flight count and its tested sample live on one axis and
    can be gated on the same scaled number.
    """
    return SampleThreshold(
        min_events=_scaled_min_periods(MIN_EVENTS_HARD, self.forward_periods),
        warn_events=_scaled_min_periods(MIN_EVENTS_WARN, self.forward_periods),
    )


@metric(
    cell=_CAAR_CELL,
    aggregation=Aggregation.EVENT_TIME,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"caar_df": compute_caar},
    sample_threshold=_caar_sample_threshold,
)
def caar(
    caar_df: pl.DataFrame,
    *,
    forward_periods: int = 5,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""CAAR significance: is mean CAAR significantly different from zero?

    The event floor is dynamic — the minimum event-period count scales with the
    forward_periods parameter (non-overlapping stride) — so it is declared as a
    resolver (a callable sample_threshold) rather than a constant. Pre-flight
    counts non-zero factor rows as a loose upper bound; this in-body short-circuit
    on event periods stays authoritative.

    Args:
        caar_df: Output of ``compute_caar()`` with columns ``date, caar,
            n_events, date_ordinal, n_events_dropped_non_finite``. The last
            column is optional (hand-built frames omit it); when present its
            row-0 value is echoed into
            ``metadata["n_events_dropped_non_finite"]``.
        forward_periods: Sampling interval for non-overlapping dates.
            Maps to ``config.forward_periods`` — the return horizon used
            in ``compute_forward_return``. Distinct from
            ``EventConfig.event_window_post`` which controls MFE/MAE.

    Returns:
        MetricResult with value=mean CAAR **on the non-overlap subsample**,
        stat=t from that same subsample.

    Notes:
        $t = \mathrm{mean}(\mathrm{CAAR}) / (\mathrm{std}(\mathrm{CAAR}) / \sqrt{n})$
        on a non-overlap subsample of the per-event-period $\mathrm{CAAR}$
        series; $H_0: \mathbb{E}[\mathrm{CAAR}] = 0$.

        **One sample behind every field.** ``value``, ``stat``, ``p_value``
        and ``n_obs`` are all computed on the event-spaced subsample. Earlier
        versions reported ``value`` as the mean of the *full* event-period
        series while ``stat``/``p_value``/``n_obs`` came from the subsample,
        so the published effect size did not belong to the published test —
        a headline CAAR could be positive while the t-stat that "supports"
        it was estimated on a different (often much smaller) sample. The
        full-series mean is still available as
        ``metadata["mean_caar_full"]`` with its size
        ``metadata["n_event_periods_full"]``; use it as a descriptive
        summary only, never as the effect size for the reported $p$.

        Null / NaN ``caar`` rows are dropped **before** the spacing pass.
        Order matters: the greedy walk keeps the first event of every
        admissible calendar gap, so a null-caar date filtered afterwards
        would still have consumed its slot and blocked the next usable
        event — silently shrinking the subsample and shifting which dates
        it contains. Dropping first lets a usable neighbour take the slot.

        The subsample is drawn **calendar-aware**: the CAAR series is
        event-period-indexed (``compute_caar`` keeps only ``factor != 0``
        rows), so its dates are calendar-irregular. Sampling every
        ``forward_periods``-th *row* (index distance) would mis-handle both
        regimes — sparse events get further thinned (power loss), clustered
        events inside one forward-return window are admitted as independent
        (iid violated, $t$ inflated). Instead a greedy pass over
        ``date_ordinal`` (each event's position on the full calendar) keeps
        an event only when its calendar gap to the previously kept event is
        ``>= forward_periods``, so consecutive kept observations no longer
        share overlapping forward-return windows. The alternative —
        reindexing to a dense calendar with zero-fill before fixed-stride
        sampling — was rejected: the zero padding would dominate the
        subsample and distort the iid mean estimator this path is built
        around; the greedy calendar walk keeps the event-only mean intact.

        ``caar`` is an **equal-weight calendar-time portfolio** test: the
        inference unit is the event *period*. Same-period events are collapsed
        to one cross-asset mean (which absorbs same-period cross-sectional
        correlation by construction), and the t-test runs across those
        periods — so ``n`` counts event *periods* (the number of periods with
        an event), not events, and ``metadata["n_event_periods_sampled"]``
        names it precisely. ``n_obs_axis`` is ``"events"``, the one token the
        whole event battery reports on: every member's sample is a set of
        non-overlapping event observations, and stacking their ``to_frame()``
        rows must not yield two axis labels for one quantity. It uses
        non-overlap resampling rather than Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent
        (HAC), the same convention as ``ic``.

        The across-events siblings are complementary, not redundant:
        ``bmp_z`` is the across-events standardized-AR z-test with the
        Kolari-Pynnönen clustering correction on by default — use it when
        events are heavily clustered or across-events power is wanted; and
        ``corrado_rank`` is the non-parametric rank test robust to
        heavy-tailed event returns, and — because it tests the event-period
        series rather than pooled event rows — to same-period clustering as
        well. The per-period portfolio breadth behind
        this test is surfaced as ``n_events`` (the ``compute_caar`` series)
        and ``total_events`` (this result's metadata).

    References:
        - [Brown & Warner (1985)][brown-warner-1985]. "Using Daily Stock
          Returns: The Case of Event Studies." Journal of Financial
          Economics, 14(1), 3–31. Daily event-study t-test specification
          at standard sample sizes.
        - [MacKinlay (1997)][mackinlay-1997]. "Event Studies in Economics
          and Finance." Journal of Economic Literature, 35(1), 13–39.
          Event-window vocabulary.

    Examples:
        Chain from :func:`compute_caar` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.caar import compute_caar, caar
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> caar_df = compute_caar(panel)
        >>> result = caar(caar_df, forward_periods=5)
        >>> result.name == ""
        True
    """
    # Drop non-finite caar dates up front so every downstream step — the
    # floor check, the spacing pass, the headline mean and the t-test — sees
    # the same usable sample. polars' drop_nulls does not remove float NaN,
    # hence the paired drop_nans (project convention for SERIES consumers).
    n_event_periods_full = caar_df.height
    caar_df = caar_df.filter(pl.col("caar").is_not_null() & pl.col("caar").is_not_nan())
    vals = caar_df["caar"]
    n = len(vals)
    # Total underlying events behind the event-period portfolio. compute_caar
    # supplies the per-period n_events; a hand-built caar_df without it falls
    # back to one-event-per-period (n).
    total_events = (
        int(caar_df["n_events"].sum()) if "n_events" in caar_df.columns else n
    )
    raw_min_warn = _scaled_min_periods(MIN_EVENTS_WARN, forward_periods)
    sc = _enforce_scaled_floor(
        "caar",
        n,
        MIN_EVENTS_HARD,
        forward_periods,
        "insufficient_event_periods",
        axis="events",
    )
    if sc is not None:
        return sc

    warning_codes: list[str] = []
    warn_code = _warn_below_scaled_floor(
        n,
        MIN_EVENTS_WARN,
        forward_periods,
        f"caar: n_event_periods={n} below the floor of {raw_min_warn} "
        f"(= MIN_EVENTS_WARN {MIN_EVENTS_WARN} x forward_periods "
        f"{forward_periods}). The t-test runs on the subsample left after "
        f"non-overlap sampling at stride h={forward_periods}, which keeps "
        f"about one period in h, so the raw series must carry h times "
        f"MIN_EVENTS_WARN periods to land on {MIN_EVENTS_WARN} independent "
        f"observations. caar is an equal-weight calendar-time portfolio "
        f"across event *periods*, so this counts the number of periods with "
        f"an event, not events. t-stat returned but read p-values cautiously.",
        WarningCode.FEW_EVENTS,
        expected_warnings=expected_warnings,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    mean_caar_full = float(vals.mean())  # type: ignore[arg-type]
    # Normal input arrives from compute_caar carrying date_ordinal (the
    # full-calendar position). A hand-built caar_df that bypasses
    # compute_caar lacks it; fall back to the dense rank of the event periods
    # themselves — event-index spacing is less calendar-aware, but never raises.
    if "date_ordinal" not in caar_df.columns:
        caar_df = caar_df.with_columns(
            (pl.col("date").rank(method="dense") - 1).alias("date_ordinal")
        )
    # caar_df is already free of null/NaN caar (filtered above), so the
    # spacing pass allocates its slots to usable dates only.
    sampled_df = _sample_event_spaced(caar_df, forward_periods)
    sampled = sampled_df["caar"]
    n_sampled = len(sampled)

    # Headline value comes from the *tested* sample, not the full series.
    mean_caar = float(sampled.mean()) if n_sampled else float("nan")  # type: ignore[arg-type]
    t = (
        _calc_t_stat(mean_caar, float(sampled.std()), n_sampled)  # type: ignore[arg-type]
        if n_sampled >= 2
        else float("nan")
    )
    p = _p_value_from_t(t, n_sampled)

    metadata: dict = {
        "n_event_periods": n,
        "total_events": total_events,
        "n_event_periods_sampled": n_sampled,
        # Full (pre-spacing) event-period series, kept for description only:
        # it is NOT the sample behind stat / p_value / n_obs.
        "mean_caar_full": mean_caar_full,
        "n_event_periods_full": n_event_periods_full,
        "n_event_periods_dropped_non_finite": n_event_periods_full - n,
        "stat_type": "t",
        "h0": "mu=0",
        "method": "non-overlapping t-test",
    }
    _warn_event_window_overlap(
        "caar",
        n,
        n_sampled,
        forward_periods,
        metadata,
        warning_codes,
        expected_warnings=expected_warnings,
    )
    if "n_events_dropped_non_finite" in caar_df.columns and caar_df.height:
        metadata["n_events_dropped_non_finite"] = int(
            caar_df["n_events_dropped_non_finite"][0]
        )

    # Fewer than two spaced event periods, or an identical CAAR on all of them:
    # ``mean_caar`` still stands, the t does not exist.
    stat, p_out, alternative = _degenerate_test_fields(
        t, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=mean_caar,
        n_obs=n_sampled,
        n_obs_axis="events",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


@metric(
    cell=_CAAR_CELL,
    aggregation=Aggregation.EVENT_TIME,
    slice_boundary_sensitive=True,
    sample_threshold=_event_sample_threshold,
)
def bmp_z(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    estimation_window: int = 60,
    forward_periods: int = 5,
    kolari_pynnonen_adjust: bool = True,
    include_prediction_error_variance: bool = False,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    r"""Boehmer-Musumeci-Poulsen Standardized Abnormal Return test.

    The static event floor (MIN_EVENTS_HARD) gates the standardized-AR z-test on
    the count of events with a usable estimation-window volatility that survive
    the event-axis spacing pass; the warn floor scales with the forward_periods
    parameter (the spacing stride), so the threshold is declared as a resolver
    (a callable sample_threshold) rather than a constant.

    Standardizes each event's abnormal return by the asset's pre-event
    residual volatility, making the test robust to event-induced variance
    inflation that biases the ordinary CAAR $t$-test.

    Uses ``price`` column for estimation-window volatility if available;
    falls back to per-asset historical ``forward_return`` std otherwise.
    The fallback std is lagged by ``forward_periods`` so the estimation
    window ends before each event's own forward return (which spans
    ``(t, t+h]``) rather than leaking the event AR into its own
    standardiser; it remains a coarser, horizon-overlapping vol proxy
    than a price-derived one-period std and raises
    ``WarningCode.BMP_RETURN_VOL_FALLBACK``
    (``metadata["vol_source"]`` records which path ran).

    Steps:
        1. For each event ($\text{factor} \neq 0$), look back
           ``estimation_window`` periods of the same asset's returns to
           estimate $\sigma_i$.
        2. Scale $\sigma_i$ to match the forward_return horizon.
        3. $\mathrm{SAR}_i = \mathrm{AR}^{\mathrm{signed}}_i / \sigma^{\text{scaled}}_i$.
        4. $z = \mathrm{mean}(\mathrm{SAR}) / (\mathrm{std}(\mathrm{SAR}) / \sqrt{N})$.

    Args:
        data: Full panel (including non-event rows) with ``date, asset_id,
            factor, forward_return``. Must include enough history for
            estimation window.
        estimation_window: Number of periods before each event for
            volatility estimation (default 60).
        forward_periods: Return horizon for vol scaling (default 5),
            counted in panel rows — not calendar days. When using
            price-derived one-period vol, scales by
            ``1/sqrt(forward_periods)`` to match per-period forward_return.
        kolari_pynnonen_adjust: On by default. Apply the
            [Kolari-Pynnönen (2010)][kolari-pynnonen-2010] adjustment for
            cross-sectional correlation of SAR. BMP assumes events are
            cross-sectionally independent; same-period events (earnings
            season, index rebalances, macro releases) share a common shock
            and break that. Measured on a true null with ρ = 0.5 (nominal
            5%): 1 event per period 6.0% either way (the adjustment is the
            identity when no two events share a period —
            ``kolari_pynnonen_applied`` is ``False``); 4 per period
            **21.5% → 5.0%**; 10 per period **40% → 4.5%**. What it cannot
            do is manufacture independent periods: once same-period events
            are correlated the effective sample is closer to the number of
            distinct event *periods* than to the event count, and the
            adjusted test's residual tracks that — 4 events per period over
            4 / 8 / 15 / 30 periods: 15.2% / 10.7% / 5.5% / 5.8%, and 10
            events per period over 8 / 15 / 30: 9.2% / 5.8% / 6.8% (400
            reps, SE ≈ 1.1–1.8 pp). The residual depends on the period
            count, not on how many events share each period, and is gone
            by ~15 periods either way. So when events share periods,
            ``FEW_EVENTS`` fires on ``metadata["n_event_periods"]``
            (distinct event periods) below ``MIN_EVENTS_WARN``, not on
            ``n_events``; with one event per period the two coincide and
            the hard floor on ``n_events`` is the only gate. The measured
            edge is ~15; ``MIN_EVENTS_WARN`` (30) is the floor ``caar`` and
            ``corrado_rank`` already apply on this axis, reused so the three
            event-study tests share one grammar — inside [15, 30) the
            warning is conservative, not a measured miscalibration.
            ``False`` gives the unadjusted BMP for reproducing a
            source that reports it; ``metadata["kolari_pynnonen_applied"]``
            / ``["kolari_pynnonen_r"]`` disclose which statistic ran.
            Formula:
            $z_{\mathrm{KP}} = z_{\mathrm{BMP}} / \sqrt{1 + (N_{\mathrm{eff}} - 1) \cdot \hat r}$
            where $\hat r$ is the one-way-ANOVA ICC(1) estimate of the
            within-period correlation of SAR and
            ``N_eff`` is the average events per event period. This is the
            plain design-effect deflator. The published K-P statistic
            carries an extra $(1 - \hat r)$ in the numerator, which
            factrix deliberately omits — see Notes for why. Vanilla BMP
            overstates significance when events cluster on the same
            date (earnings season, macro release), inflating z by
            factors of 1.5-2×. Enable this when the event-study
            ``clustering_hhi`` diagnostic is high (≥ 0.3) or when you
            otherwise expect same-period shock sharing.
        include_prediction_error_variance: When True, inflate the
            per-event standardiser by $\sqrt{1 + 1/T_{\mathrm{est}}}$
            (with $T_{\mathrm{est}}$ = ``estimation_window``) to absorb
            the prediction-error variance of the mean-adjusted residual
            forecast — the strict [Boehmer-Musumeci-Poulsen (1991)][boehmer-musumeci-poulsen-1991]
            denominator. Default is
            False, preserving the prior factrix denominator (residual
            std only). Under mean-adjusted residuals + a single
            ``estimation_window`` the correction scales every SAR by
            the same constant, so ``mean_SAR`` and ``std_SAR`` shrink
            by $1/\sqrt{1 + 1/T_{\mathrm{est}}}$ but the $z$ statistic
            is invariant: the flag documents the strict standardiser,
            it does not move inference in this regime. Per-event $T_i$
            variation (which would move $z$) requires a market-model
            extension and is out of scope here.

            Caveat: ``rolling_std(min_samples=20)`` accepts events with
            as few as 20 prior returns, so the effective $T_i$ for
            early-history events can be smaller than ``estimation_window``.
            The constant correction is therefore an approximation in
            that regime; ensure every event has at least
            ``estimation_window`` prior returns when the strict denominator
            matters.

    Returns:
        MetricResult(value=mean_SAR, p_value=p_bmp, stat=z_bmp, ...).

    Notes:
        For each event $i$: estimate pre-event vol $\sigma_i$ over the
        ``estimation_window``, scaled to the forward horizon by
        $1/\sqrt{h}$ (with $h$ = ``forward_periods``) when a ``price`` column is
        available;
        $\mathrm{SAR}_i = \mathrm{AR}^{\mathrm{signed}}_i / \sigma_i$; aggregate to
        $z = \mathrm{mean}(\mathrm{SAR}) / (\mathrm{std}(\mathrm{SAR}) / \sqrt{N})$.
        With ``kolari_pynnonen_adjust=True``, scale $z$ by
        $1 / \sqrt{1 + (N_{\mathrm{eff}} - 1)\, \hat r}$.

        **Which K-P deflator.** Kolari-Pynnönen's published statistic
        multiplies by $\sqrt{(1 - \hat r) / (1 + (N_{\mathrm{eff}} - 1)\hat r)}$.
        The $(1 - \hat r)$ numerator belongs to their setting, where the
        SAR variance is estimated *within a single event period*: a
        one-date cross-sectional variance under equicorrelation estimates
        only the idiosyncratic share $(1 - \hat r)\sigma^2$, so the
        numerator restores the missing between-date component. factrix
        pools SAR across all event periods, so ``std_sar`` already contains
        that component and re-applying $(1 - \hat r)$ would deflate $z$
        twice — anti-powered rather than merely conservative. The variant
        implemented here is therefore the pure design-effect deflator
        $1/\sqrt{1 + (N_{\mathrm{eff}} - 1)\hat r}$, the standard
        clustered-sampling correction for a pooled variance. Callers who
        need the literal published constant can recover it from
        ``metadata["stat_uncorrected"]`` and
        ``metadata["kolari_pynnonen_r"]``.

        **Event validity.** An event enters the test only when its
        estimation-window vol is finite and above ``EPSILON`` *and* its
        signed AR is finite. The two rejection reasons are reported
        separately (``n_dropped_no_vol`` / ``n_dropped_non_finite_return``,
        summing to ``n_dropped``) because they mean different things: the
        first is a short history, the second is a hole in ``return_col``.
        Both must be excluded — a NaN SAR propagates through mean/std and
        turns $z$ into a silent 0 with $p = 1$.

        factrix simplifies the original BMP by omitting the prediction-
        error term from the standardiser (using mean-adjusted residuals
        rather than market-model residuals) — adequate for the default
        Brown-Warner / MacKinlay event-study path; pair with the K-P
        adjustment when ``clustering_hhi`` flags same-period shock sharing.
        Pass ``include_prediction_error_variance=True`` for the strict
        BMP denominator $\sigma_i \cdot \sqrt{1 + 1/T_{\mathrm{est}}}$.

    References:
        - [Boehmer, Musumeci & Poulsen (1991)][boehmer-musumeci-poulsen-1991].
          "Event-study Methodology Under Conditions of Event-induced
          Variance." Journal of Financial Economics, 30(2), 253–272.
          The BMP standardised AR test factrix simplifies (mean-adjusted
          residuals, no prediction-error correction by default).
        - [Kolari & Pynnönen (2010)][kolari-pynnonen-2010]. "Event Study
          Testing with Cross-sectional Correlation of Abnormal Returns."
          Review of Financial Studies, 23(11), 3996–4025. Clustering-
          adjusted BMP variant; enabled via
          ``kolari_pynnonen_adjust=True`` on this function.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.caar import bmp_z
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = bmp_z(panel, forward_periods=5)
        >>> result.name == ""
        True
    """
    sorted_df = data.sort(["asset_id", "date"])

    uses_price = "price" in sorted_df.columns
    if uses_price:
        sorted_df = sorted_df.with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1).alias(
                "_period_ret"
            )
        )
        # WHY: forward_return = (price[t+1+forward_periods]/price[t+1] - 1)
        # / forward_periods has std ≈ σ_1 / sqrt(forward_periods), where σ_1 is
        # the one-period (one-row) return std. Scale estimation vol to match.
        # "One period" is whatever the panel's own frequency is — factrix never
        # reads the calendar, so this holds for daily, weekly or monthly rows.
        vol_scale = 1.0 / np.sqrt(forward_periods)
        # Price one-period returns at [t-forward_periods+1, t] precede the event
        # window (t, t+h], so no extra lag is needed.
        vol_lag = 0
    else:
        sorted_df = sorted_df.with_columns(pl.col(return_col).alias("_period_ret"))
        vol_scale = 1.0
        # WHY: forward_return[t] realises over (t+1, t+1+h], so a rolling std
        # ending at row t would standardise the event AR with a window that
        # already contains that event's own (and adjacent) forward returns —
        # the numerator leaks into its own denominator. Lag the fallback std by
        # forward_periods so the estimation window ends before the event window.
        vol_lag = forward_periods

    # Strict BMP (1991) denominator for mean-adjusted residuals: a
    # forecast SE is √(1 + 1/T) larger than the in-sample residual std.
    # Off by default — flipping it shifts every z by a known factor and
    # downstream callers may calibrate against the simpler denominator.
    if include_prediction_error_variance:
        vol_scale *= float(np.sqrt(1.0 + 1.0 / estimation_window))

    # With price the window [t-N+1, t] already precedes the event window; the
    # fallback adds a forward_periods lag (see above) so it too ends pre-event.
    est_vol_expr = pl.col("_period_ret").rolling_std(
        window_size=estimation_window, min_samples=20
    )
    if vol_lag:
        est_vol_expr = est_vol_expr.shift(vol_lag)
    sorted_df = sorted_df.with_columns(
        (est_vol_expr.over("asset_id") * vol_scale).alias("_est_vol")
    )

    events = sorted_df.filter(pl.col(factor_col) != 0)
    if len(events) == 0:
        return _short_circuit_output(
            "bmp_z",
            "no_events",
            n_obs=0,
            n_obs_axis="events",
            min_required=1,
        )

    events = events.with_columns(
        (pl.col(return_col) * pl.col(factor_col).sign()).alias("_signed_ar")
    )

    # A usable event needs BOTH an estimation-window vol and a finite signed
    # AR. Filtering on the vol alone let a null/NaN return_col through: the
    # SAR became NaN, mean/std of SAR became NaN, and _calc_t_stat silently
    # returned z=0, p=1 while n_obs still advertised the full event count.
    # is_not_nan() is required alongside is_not_null() because polars treats
    # float NaN as a value, not a null.
    vol_ok = (
        pl.col("_est_vol").is_not_null()
        & pl.col("_est_vol").is_not_nan()
        & (pl.col("_est_vol") > EPSILON)
    )
    ar_ok = pl.col("_signed_ar").is_not_null() & pl.col("_signed_ar").is_not_nan()

    n_dropped_no_vol = events.filter(~vol_ok).height
    # Counted on the vol-ok subset so the two reasons partition the drops
    # and sum to n_dropped.
    n_dropped_non_finite_return = events.filter(vol_ok & ~ar_ok).height

    valid = events.filter(vol_ok & ar_ok)

    n_valid_raw = len(valid)
    # Overlap discipline on the TIME axis. The Kolari-Pynnonen adjustment below
    # handles same-period cross-asset correlation; it has nothing to work with
    # on a single asset, where the only clustering axis is time. Two events on
    # one asset less than forward_periods apart share future bars, so pooling
    # them into the cross-sectional z counts one draw twice.
    valid = _sample_events_non_overlapping(
        valid, forward_periods, calendar_dates=sorted_df["date"]
    )
    n_valid = len(valid)

    # The floor is enforced on the SAMPLED count — the sample the z actually
    # runs on. Pre-flight reads the raw non-zero factor count as the documented
    # loose upper bound.
    sc = _enforce_min_floor(
        bmp_z,
        "bmp_z",
        n_valid,
        "insufficient_estimation_window",
        axis="events",
        forward_periods=forward_periods,
        n_events_raw=n_valid_raw,
    )
    if sc is not None:
        return sc

    valid = valid.with_columns(
        (pl.col("_signed_ar") / pl.col("_est_vol")).alias("_sar")
    )
    sar = valid["_sar"].to_numpy()
    mean_sar = float(np.mean(sar))
    std_sar = float(np.std(sar, ddof=DDOF))

    z_bmp = _calc_t_stat(mean_sar, std_sar, n_valid)
    n_event_periods = int(valid["date"].n_unique())

    warning_codes: list[str] = []
    metadata: dict = {
        "n_events": n_valid,
        "n_event_periods": n_event_periods,
        "n_dropped": len(events) - n_valid_raw,
        "n_dropped_no_vol": n_dropped_no_vol,
        "n_dropped_non_finite_return": n_dropped_non_finite_return,
        "std_sar": std_sar,
        "estimation_window": estimation_window,
        "stat_type": "z",
        "h0": "mu_SAR=0",
        "method": "BMP standardized cross-sectional test",
        "include_prediction_error_variance": include_prediction_error_variance,
        "vol_source": "price" if uses_price else "forward_return",
        "vol_estimation_lag": vol_lag,
    }
    _warn_event_window_overlap(
        "bmp_z",
        n_valid_raw,
        n_valid,
        forward_periods,
        metadata,
        warning_codes,
        expected_warnings=expected_warnings,
    )
    if not uses_price:
        warning_codes.append(WarningCode.BMP_RETURN_VOL_FALLBACK.value)
        warnings.warn(
            f"bmp_z: no 'price' column; estimation-window volatility falls back "
            f"to the per-asset rolling std of '{return_col}', lagged by "
            f"forward_periods={forward_periods} so the window ends before each "
            f"event's forward return. This is a coarser, horizon-overlapping vol "
            f"proxy than a price-derived one-period std — supply 'price' for the "
            f"clean BMP "
            f"standardiser.",
            UserWarning,
            stacklevel=2,
        )

    if kolari_pynnonen_adjust:
        r_hat, n_eff, kp_source = _estimate_within_date_icc(
            valid.select("date", "_sar"), "_sar"
        )
        metadata["kolari_pynnonen_r"] = r_hat
        metadata["kolari_pynnonen_n_eff"] = n_eff
        metadata["kolari_pynnonen_r_source"] = kp_source
        if r_hat is None or n_eff <= 1.0:
            metadata["kolari_pynnonen_applied"] = False
            z = z_bmp
        else:
            scale = _kp_cluster_scale(r_hat, n_eff)
            z = z_bmp * scale
            metadata["kolari_pynnonen_scaling"] = scale
            metadata["kolari_pynnonen_applied"] = True
            metadata["stat_uncorrected"] = z_bmp
            metadata["method"] = (
                "BMP + Kolari-Pynnönen (2010) cross-sectional-correlation adjustment"
            )
    else:
        z = z_bmp

    # Two ways this z runs on too little independent sample, one code.
    # (1) The raw event count is below the stride-scaled floor caar and
    #     corrado_rank apply: the spacing pass above keeps about one event in
    #     h per asset, so the raw count must carry h times MIN_EVENTS_WARN to
    #     land on MIN_EVENTS_WARN independent ones.
    raw_min_warn = _scaled_min_periods(MIN_EVENTS_WARN, forward_periods)
    warn_code = _warn_below_scaled_floor(
        n_valid_raw,
        MIN_EVENTS_WARN,
        forward_periods,
        f"bmp_z: n_events={n_valid_raw} below the floor of {raw_min_warn} "
        f"(= MIN_EVENTS_WARN {MIN_EVENTS_WARN} x forward_periods "
        f"{forward_periods}); {n_valid} events and {n_event_periods} event "
        f"periods survive non-overlap sampling at stride h={forward_periods}, "
        f"which keeps about one event in h per asset. z is returned but the "
        f"cross-sectional test is power-thin on a sample this short.",
        WarningCode.FEW_EVENTS,
        expected_warnings=expected_warnings,
    )
    # (2) Events share periods, so the effective sample is closer to the
    #     distinct event periods than to the event count (KP deflates z for the
    #     shared shock but cannot manufacture independent periods). Measured
    #     regime, kept as its own trigger — the count can clear (1) and still
    #     rest on a handful of periods.
    if (
        warn_code is None
        and n_event_periods < n_valid
        and (n_event_periods < MIN_EVENTS_WARN)
    ):
        warn_code = WarningCode.FEW_EVENTS.value
        if warn_code not in expected_warnings:
            warnings.warn(
                f"bmp_z: n_event_periods={n_event_periods} below "
                f"MIN_EVENTS_WARN={MIN_EVENTS_WARN} with n_events={n_valid}. "
                f"Same-period events share a common shock, so the effective "
                f"sample is closer to the number of distinct event periods than "
                f"to the event count; the Kolari-Pynnönen adjustment removes the "
                f"shared-shock inflation but not the small-sample residual "
                f"(measured ~11% at 8 periods, ~15% at 4, clearing by ~15; "
                f"nominal 5%). z is returned but read borderline p-values "
                f"cautiously.",
                UserWarning,
                stacklevel=2,
            )
    if warn_code is not None:
        warning_codes.append(warn_code)

    p = _p_value_from_z(z)
    # An identical standardized AR across every event zeroes the BMP
    # cross-sectional denominator: ``mean_sar`` is real, the z is not.
    stat, p_out, alternative = _degenerate_test_fields(
        z, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=mean_sar,
        n_obs=n_valid,
        n_obs_axis="events",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
