"""CAAR (Cumulative Average Abnormal Return) metrics for event signals.

Input: DataFrame with ``date, asset_id, factor, forward_return``.
``factor`` is discrete {-1, 0, +1} — events only on non-zero rows.

Core metrics:
    compute_caar — per-event-date signed abnormal return series
    caar         — CAAR t-test (H₀: mean = 0)
    bmp_test     — Boehmer-Musumeci-Poulsen standardized AR test
    event_hit_rate — fraction of events where signed_car > 0

References:
    MacKinlay (1997), "Event Studies in Economics and Finance"
    Boehmer, Musumeci & Poulsen (1991), "Event-study methodology
        under conditions of event-induced variance"
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import DDOF, EPSILON, MIN_EVENTS, MetricOutput
from factorlib._stats import (
    _calc_t_stat,
    _p_value_from_t,
    _p_value_from_z,
    _significance_marker,
)
from factorlib.metrics._helpers import _sample_non_overlapping, _signed_car


def compute_caar(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """Per-event-date signed abnormal return series.

    ``signed_car = return × sign(factor)``
    ``caar = per-date mean of signed_car across events``

    Only rows where ``factor ≠ 0`` are included (event rows).

    Args:
        df: Panel with ``date``, ``asset_id``, ``factor_col``, ``return_col``.
        factor_col: Column with discrete signal {-1, 0, +1}.
        return_col: Column with forward/abnormal return.

    Returns:
        DataFrame with columns ``date, caar`` sorted by date.
    """
    return (
        df.filter(pl.col(factor_col) != 0)
        .with_columns(
            (pl.col(return_col) * pl.col(factor_col).sign())
            .alias("_signed_car")
        )
        .group_by("date")
        .agg(pl.col("_signed_car").mean().alias("caar"))
        .sort("date")
    )


def caar(
    caar_df: pl.DataFrame,
    *,
    forward_periods: int = 5,
) -> MetricOutput:
    """CAAR significance: is mean CAAR significantly different from zero?

    Uses non-overlapping sampling (every ``forward_periods``-th date) to
    eliminate autocorrelation from overlapping forward returns.

    Statistical method: t = mean / (std / sqrt(n)) on non-overlapping samples.
    H₀: mean(CAAR) = 0.

    Args:
        caar_df: Output of ``compute_caar()`` with columns ``date, caar``.
        forward_periods: Sampling interval for non-overlapping dates.
            Maps to ``config.forward_periods`` — the return horizon used
            in ``compute_forward_return``. Distinct from
            ``EventConfig.event_window_post`` which controls MFE/MAE.

    Returns:
        MetricOutput with value=mean CAAR, stat=t from non-overlapping sampling.
    """
    vals = caar_df["caar"].drop_nulls()
    n = len(vals)
    if n < MIN_EVENTS:
        return MetricOutput(name="caar", value=0.0, stat=0.0, significance="")

    mean_caar = float(vals.mean())
    sampled = _sample_non_overlapping(caar_df, forward_periods)["caar"].drop_nulls()
    n_sampled = len(sampled)

    t = (
        _calc_t_stat(float(sampled.mean()), float(sampled.std()), n_sampled)
        if n_sampled >= 2
        else 0.0
    )
    p = _p_value_from_t(t, n_sampled)

    return MetricOutput(
        name="caar",
        value=mean_caar,
        stat=t,
        significance=_significance_marker(p),
        metadata={
            "n_event_dates": n,
            "n_sampled": n_sampled,
            "p_value": p,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "non-overlapping t-test",
        },
    )


def bmp_test(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    estimation_window: int = 60,
    forward_periods: int = 5,
) -> MetricOutput:
    """Boehmer-Musumeci-Poulsen Standardized Abnormal Return test.

    Standardizes each event's abnormal return by the asset's pre-event
    residual volatility, making the test robust to event-induced variance
    inflation that biases the ordinary CAAR t-test.

    Steps:
        1. For each event (factor ≠ 0), look back ``estimation_window``
           periods of the same asset's returns to estimate σ_i.
        2. Scale σ_i to match the forward_return horizon.
        3. SAR_i = signed_AR_i / σ_scaled_i
        4. z = mean(SAR) / (std(SAR) / √N)

    Uses ``price`` column for estimation-window volatility if available;
    falls back to per-asset historical ``forward_return`` std otherwise.

    Args:
        df: Full panel (including non-event rows) with ``date, asset_id,
            factor, forward_return``. Must include enough history for
            estimation window.
        estimation_window: Number of periods before each event for
            volatility estimation (default 60).
        forward_periods: Return horizon for vol scaling (default 5).
            When using price-derived daily vol, scales by
            ``1/sqrt(forward_periods)`` to match per-period forward_return.

    Returns:
        MetricOutput(name="bmp_sar", value=mean_SAR, stat=z_bmp, ...).
    """
    sorted_df = df.sort(["asset_id", "date"])

    if "price" in sorted_df.columns:
        sorted_df = sorted_df.with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("asset_id") - 1)
            .alias("_daily_ret")
        )
        # WHY: forward_return = (price[t+1+N]/price[t+1] - 1) / N has
        # std ≈ σ_daily / sqrt(N). Scale estimation vol to match.
        vol_scale = 1.0 / np.sqrt(forward_periods)
    else:
        sorted_df = sorted_df.with_columns(
            pl.col(return_col).alias("_daily_ret")
        )
        vol_scale = 1.0

    # WHY: no .shift(1) needed — forward_return already starts at t+1
    # (compute_forward_return uses t+1 entry), so the estimation window
    # at row t naturally covers [t-N+1, t] without event contamination.
    sorted_df = sorted_df.with_columns(
        (
            pl.col("_daily_ret")
            .rolling_std(window_size=estimation_window, min_samples=20)
            .over("asset_id")
            * vol_scale
        ).alias("_est_vol")
    )

    events = sorted_df.filter(pl.col(factor_col) != 0)
    if len(events) == 0:
        return MetricOutput(
            name="bmp_sar", value=0.0, stat=0.0, significance="",
        )

    events = events.with_columns(
        (pl.col(return_col) * pl.col(factor_col).sign()).alias("_signed_ar")
    )

    valid = events.filter(
        pl.col("_est_vol").is_not_null() & (pl.col("_est_vol") > EPSILON)
    )

    n_valid = len(valid)
    if n_valid < MIN_EVENTS:
        return MetricOutput(
            name="bmp_sar", value=0.0, stat=0.0, significance="",
            metadata={"n_events": n_valid, "reason": "insufficient estimation data"},
        )

    sar = (valid["_signed_ar"] / valid["_est_vol"]).to_numpy()
    mean_sar = float(np.mean(sar))
    std_sar = float(np.std(sar, ddof=DDOF))

    z_bmp = _calc_t_stat(mean_sar, std_sar, n_valid)
    p = _p_value_from_z(z_bmp)

    return MetricOutput(
        name="bmp_sar",
        value=mean_sar,
        stat=z_bmp,
        significance=_significance_marker(p),
        metadata={
            "n_events": n_valid,
            "n_dropped": len(events) - n_valid,
            "std_sar": std_sar,
            "estimation_window": estimation_window,
            "p_value": p,
            "stat_type": "z",
            "h0": "mu_SAR=0",
            "method": "BMP standardized cross-sectional test",
        },
    )


def event_hit_rate(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Fraction of events where signed abnormal return > 0.

    Uses binomial score test: H₀: p = 0.5 (random direction).
    z = (hits - n*p0) / sqrt(n*p0*(1-p0))

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=hit_rate, stat=z from binomial test.
    """
    events = df.filter(pl.col(factor_col) != 0)

    n = len(events)
    if n < MIN_EVENTS:
        return MetricOutput(
            name="event_hit_rate", value=0.0, stat=0.0, significance="",
        )

    signed = _signed_car(events, factor_col, return_col)
    hits = int(np.sum(signed > 0))
    rate = hits / n

    z = (hits - n * 0.5) / (np.sqrt(n) * 0.5)
    p = _p_value_from_z(z)

    return MetricOutput(
        name="event_hit_rate",
        value=rate,
        stat=z,
        significance=_significance_marker(p),
        metadata={
            "n_events": n,
            "n_hits": hits,
            "p_value": p,
            "stat_type": "z",
            "h0": "p=0.5",
            "method": "binomial score test",
        },
    )
