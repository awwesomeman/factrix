r"""CAAR (Cumulative Average Abnormal Return) significance tests.

Tests $H_0$: event abnormal return = 0, using two complementary methods:
    compute_caar — per-event-date weighted abnormal return series
    caar         — CAAR t-test (parametric, non-overlapping sampling)
    bmp_z     — BMP standardized AR test (robust to event-induced variance)

Notes:
    `caar` and `bmp_z` are complementary inferential tests on the
    per-event-date abnormal-return series. `caar` is the parametric
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
    _enforce_min_floor,
    _estimate_within_date_icc,
    _kp_cluster_scale,
    _sample_event_spaced,
    _scaled_min_periods,
    _short_circuit_output,
)
from factrix.metrics._metric_capabilities import per_date_series_rename
from factrix.metrics._primitives import compute_caar

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "compute_caar",
    "caar",
    "bmp_z",
]

# structure=None (event-axis): caar/bmp_z aggregate over the event cross-section
# (event dates / events), not the asset cross-section, so they run on single-asset
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
    """Dynamic event floor for ``caar``: the raw event-date count scales with
    ``forward_periods`` because the t-test runs on a non-overlap subsample
    (stride ``forward_periods``). Delegates to the same ``_scaled_min_periods``
    the in-body short-circuit reads, so pre-flight and run-time floors agree.
    """
    return SampleThreshold(
        min_events=_scaled_min_periods(MIN_EVENTS_HARD, self.forward_periods),
        warn_events=_scaled_min_periods(MIN_EVENTS_WARN, self.forward_periods),
    )


@metric(
    cell=_CAAR_CELL,
    aggregation=Aggregation.EVENT_TIME,
    input_shape=InputShape.SERIES,
    requires={"caar_df": compute_caar},
    sample_threshold=_caar_sample_threshold,
)
def caar(
    caar_df: pl.DataFrame,
    *,
    forward_periods: int = 5,
) -> MetricResult:
    r"""CAAR significance: is mean CAAR significantly different from zero?

    The event floor is dynamic — the minimum event-date count scales with the
    forward_periods parameter (non-overlapping stride) — so it is declared as a
    resolver (a callable sample_threshold) rather than a constant. Pre-flight
    counts non-zero factor rows as a loose upper bound; this in-body short-circuit
    on event dates stays authoritative.

    Args:
        caar_df: Output of ``compute_caar()`` with columns ``date, caar,
            n_events, date_ordinal``.
        forward_periods: Sampling interval for non-overlapping dates.
            Maps to ``config.forward_periods`` — the return horizon used
            in ``compute_forward_return``. Distinct from
            ``EventConfig.event_window_post`` which controls MFE/MAE.

    Returns:
        MetricResult with value=mean CAAR, stat=t from non-overlapping sampling.

    Notes:
        $t = \mathrm{mean}(\mathrm{CAAR}) / (\mathrm{std}(\mathrm{CAAR}) / \sqrt{n})$
        on a non-overlap subsample of the per-event-date $\mathrm{CAAR}$
        series; $H_0: \mathbb{E}[\mathrm{CAAR}] = 0$.

        The subsample is drawn **calendar-aware**: the CAAR series is
        event-date-indexed (``compute_caar`` keeps only ``factor != 0``
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
        inference unit is the event *date*. Same-date events are collapsed
        to one cross-asset mean (which absorbs same-date cross-sectional
        correlation by construction), and the t-test runs across those
        dates — so ``n`` counts event *dates* (the number of periods with
        an event), not events. It uses non-overlap resampling rather than
        Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent
        (HAC), the same convention as ``ic``.

        The across-events siblings are complementary, not redundant:
        ``bmp_z`` is the across-events standardized-AR z-test with an
        optional Kolari-Pynnönen clustering correction — use it when events
        are heavily clustered or across-events power is wanted; and
        ``corrado_rank`` is the non-parametric rank test robust to
        heavy-tailed event returns. The per-date portfolio breadth behind
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
    vals = caar_df["caar"].drop_nulls()
    n = len(vals)
    # Total underlying events behind the event-date portfolio. compute_caar
    # supplies the per-date n_events; a hand-built caar_df without it falls
    # back to one-event-per-date (n).
    total_events = (
        int(caar_df["n_events"].sum()) if "n_events" in caar_df.columns else n
    )
    raw_min_hard = _scaled_min_periods(MIN_EVENTS_HARD, forward_periods)
    raw_min_warn = _scaled_min_periods(MIN_EVENTS_WARN, forward_periods)
    if n < raw_min_hard:
        return _short_circuit_output(
            "caar",
            "insufficient_event_periods",
            n_obs=n,
            n_obs_axis="events",
            min_required=raw_min_hard,
            forward_periods=forward_periods,
        )

    warning_codes: list[str] = []
    if n < raw_min_warn:
        warning_codes.append(WarningCode.FEW_EVENTS.value)
        warnings.warn(
            f"caar: n_event_periods={n} below the MIN_EVENTS_WARN-scaled floor="
            f"{raw_min_warn}. caar is an equal-weight calendar-time portfolio "
            f"across event *dates*, so this counts the number of periods with "
            f"an event, not events; a sub-30 series is "
            f"power-thin for the asymptotic t-distribution. t-stat returned "
            f"but read p-values cautiously. For an across-events test under "
            f"heavy clustering, use bmp_z.",
            UserWarning,
            stacklevel=2,
        )

    mean_caar = float(vals.mean())  # type: ignore[arg-type]
    # Normal input arrives from compute_caar carrying date_ordinal (the
    # full-calendar position). A hand-built caar_df that bypasses
    # compute_caar lacks it; fall back to the dense rank of the event dates
    # themselves — event-index spacing is less calendar-aware, but never raises.
    if "date_ordinal" not in caar_df.columns:
        caar_df = caar_df.with_columns(
            (pl.col("date").rank(method="dense") - 1).alias("date_ordinal")
        )
    sampled = _sample_event_spaced(caar_df, forward_periods)["caar"].drop_nulls()
    n_sampled = len(sampled)

    t = (
        _calc_t_stat(float(sampled.mean()), float(sampled.std()), n_sampled)  # type: ignore[arg-type]
        if n_sampled >= 2
        else 0.0
    )
    p = _p_value_from_t(t, n_sampled)

    metadata: dict = {
        "n_event_periods": n,
        "total_events": total_events,
        "n_event_periods_sampled": n_sampled,
        "stat_type": "t",
        "h0": "mu=0",
        "method": "non-overlapping t-test",
    }

    return MetricResult(
        p_value=p,
        value=mean_caar,
        n_obs=n_sampled,
        n_obs_axis="events",
        stat=t,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


@metric(
    cell=_CAAR_CELL,
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def bmp_z(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    estimation_window: int = 60,
    forward_periods: int = 5,
    kolari_pynnonen_adjust: bool = False,
    include_prediction_error_variance: bool = False,
) -> MetricResult:
    r"""Boehmer-Musumeci-Poulsen Standardized Abnormal Return test.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the standardized-AR z-test on the count of events with a usable estimation-window volatility.

    Standardizes each event's abnormal return by the asset's pre-event
    residual volatility, making the test robust to event-induced variance
    inflation that biases the ordinary CAAR $t$-test.

    Uses ``price`` column for estimation-window volatility if available;
    falls back to per-asset historical ``forward_return`` std otherwise.
    The fallback std is lagged by ``forward_periods`` so the estimation
    window ends before each event's own forward return (which spans
    ``(t, t+h]``) rather than leaking the event AR into its own
    standardiser; it remains a coarser, horizon-overlapping vol proxy
    than a daily-price std and raises ``WarningCode.BMP_RETURN_VOL_FALLBACK``
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
        forward_periods: Return horizon for vol scaling (default 5).
            When using price-derived daily vol, scales by
            ``1/sqrt(forward_periods)`` to match per-period forward_return.
        kolari_pynnonen_adjust: When True, apply the
            [Kolari-Pynnönen (2010)][kolari-pynnonen-2010] adjustment for
            cross-sectional correlation of SAR:
            $z_{\mathrm{KP}} = z_{\mathrm{BMP}} \cdot \sqrt{(1 - \hat r) / (1 + (N_{\mathrm{eff}} - 1) \cdot \hat r)}$
            where $\hat r$ is the ICC-style within-date correlation of
            SAR and
            ``N_eff`` is the average events per event date. Vanilla BMP
            overstates significance when events cluster on the same
            date (earnings season, macro release), inflating z by
            factors of 1.5-2×. Enable this when the event-study
            ``clustering_hhi`` diagnostic is high (≥ 0.3) or when you
            otherwise expect same-date shock sharing.
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
        $1/\sqrt{h}$ (with $h$ = ``forward_periods``) when daily prices are available;
        $\mathrm{SAR}_i = \mathrm{AR}^{\mathrm{signed}}_i / \sigma_i$; aggregate to
        $z = \mathrm{mean}(\mathrm{SAR}) / (\mathrm{std}(\mathrm{SAR}) / \sqrt{N})$.
        With ``kolari_pynnonen_adjust=True``, scale $z$ by
        $\sqrt{(1 - \hat r) / (1 + (N_{\mathrm{eff}} - 1)\, \hat r)}$.

        factrix simplifies the original BMP by omitting the prediction-
        error term from the standardiser (using mean-adjusted residuals
        rather than market-model residuals) — adequate for the default
        Brown-Warner / MacKinlay event-study path; pair with the K-P
        adjustment when ``clustering_hhi`` flags same-date shock sharing.
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
                "_daily_ret"
            )
        )
        # WHY: forward_return = (price[t+1+N]/price[t+1] - 1) / N has
        # std ≈ σ_daily / sqrt(N). Scale estimation vol to match.
        vol_scale = 1.0 / np.sqrt(forward_periods)
        # Price daily returns at [t-N+1, t] precede the event window (t, t+h],
        # so no extra lag is needed.
        vol_lag = 0
    else:
        sorted_df = sorted_df.with_columns(pl.col(return_col).alias("_daily_ret"))
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
    est_vol_expr = pl.col("_daily_ret").rolling_std(
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

    valid = events.filter(
        pl.col("_est_vol").is_not_null() & (pl.col("_est_vol") > EPSILON)
    )

    n_valid = len(valid)
    sc = _enforce_min_floor(
        bmp_z, "bmp_z", n_valid, "insufficient_estimation_window", axis="events"
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

    warning_codes: list[str] = []
    metadata: dict = {
        "n_events": n_valid,
        "n_dropped": len(events) - n_valid,
        "std_sar": std_sar,
        "estimation_window": estimation_window,
        "stat_type": "z",
        "h0": "mu_SAR=0",
        "method": "BMP standardized cross-sectional test",
        "include_prediction_error_variance": include_prediction_error_variance,
        "vol_source": "price" if uses_price else "forward_return",
        "vol_estimation_lag": vol_lag,
    }
    if not uses_price:
        warning_codes.append(WarningCode.BMP_RETURN_VOL_FALLBACK.value)
        warnings.warn(
            f"bmp_z: no 'price' column; estimation-window volatility falls back "
            f"to the per-asset rolling std of '{return_col}', lagged by "
            f"forward_periods={forward_periods} so the window ends before each "
            f"event's forward return. This is a coarser, horizon-overlapping vol "
            f"proxy than a daily-price std — supply 'price' for the clean BMP "
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

    p = _p_value_from_z(z)

    return MetricResult(
        p_value=p,
        value=mean_sar,
        n_obs=n_valid,
        n_obs_axis="events",
        stat=z,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
