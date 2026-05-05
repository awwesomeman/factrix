r"""CAAR (Cumulative Average Abnormal Return) significance tests.

Aggregation: per-event-date weighted abnormal return (per-event-date
step) then non-overlapping cross-event sample; $t$-test on CAAR, or BMP
standardized AR $z$-test for event-induced variance.

Tests $H_0$: event abnormal return = 0, using two complementary methods:
    compute_caar — per-event-date weighted abnormal return series
    caar         — CAAR t-test (parametric, non-overlapping sampling)
    bmp_test     — BMP standardized AR test (robust to event-induced variance)

References:
    MacKinlay (1997), "Event Studies in Economics and Finance"
    Boehmer, Musumeci & Poulsen (1991), "Event-study methodology
        under conditions of event-induced variance"

Matrix-row: compute_caar, caar, bmp_test | (*, SPARSE, *, PANEL) | per-event | non-overlapping t / z | _calc_t_stat, _p_value_from_t, _p_value_from_z, _significance_marker, _sample_non_overlapping, _short_circuit_output
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import DDOF, EPSILON, KPSource, MIN_EVENTS, MetricOutput
from factrix._stats import (
    _calc_t_stat,
    _p_value_from_t,
    _p_value_from_z,
    _significance_marker,
)
from factrix.metrics._helpers import (
    _sample_non_overlapping,
    _scaled_min_periods,
    _short_circuit_output,
)


def compute_caar(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    r"""Per-event-date weighted abnormal return series.

    Aggregation:
        CS-first. For each event date, take the cross-sectional mean of
        ``signed_car`` $= r \times f$ across event rows where $f \neq 0$;
        the resulting ``n_event_dates``-length CAAR series feeds a
        downstream NW HAC $t$-test on the mean.

    Magnitude is preserved — no ``.sign()`` coercion. factrix accepts
    two input contracts; everything else (including signed
    $\{-1, 0, +1\}$) is just a special case of the second:

    | Input ``factor`` | ``signed_car`` reduces to | Statistic tested |
    |---|---|---|
    | $\{0, 1\}$ | $\text{return}$ on event rows | Average event-day return |
    | $\{0, R\}$, $R \in \mathbb{R}$ | $\text{return} \times R$ | Magnitude-weighted CAAR |

    Caveat on $\{-1, 0, +1\}$: it lands in the second row as a
    weight-$\pm 1$ case, which gives a magnitude-weighted CAAR, **not**
    the textbook MacKinlay (1997) signed CAAR (the latter averages
    direction-flipped abnormal returns and is a different estimator at
    finite samples when the negative-leg vol differs from the positive-
    leg vol). If literature-standard signed CAAR is what you want,
    pre-compute it externally; factrix's primitive treats $\pm 1$ as
    weights, not as direction labels.

    Scale:
        CAAR magnitude tracks the units of ``factor`` (bps, z-score,
        unit-less ±1). Hypothesis tests via the downstream ``caar``
        t-statistic are scale-invariant (numerator and denominator both
        scale linearly), but cross-factor *effect-size* comparisons
        require commensurate units.

    Args:
        df: Panel with ``date``, ``asset_id``, ``factor_col``, ``return_col``.
        factor_col: Numeric column. Magnitude is preserved as a weight
            in the per-row product; only zero rows are filtered.
        return_col: Column with forward/abnormal return.

    Returns:
        DataFrame with columns ``date, caar`` sorted by date.

    Notes:
        Per event date $d$,
        $\mathrm{CAAR}_d = \mathrm{mean}_{i \in \mathrm{events}(d)} (\mathrm{return}_i \times \mathrm{factor}_i)$.
        Magnitude of ``factor`` is preserved as a weight; only rows with
        ``factor == 0`` are dropped.

        factrix follows the MacKinlay (1997) event-window vocabulary
        (factor as the event indicator / sign on the announcement date)
        but generalises ``signed_car`` to numeric factor magnitude. With
        a continuous ``factor`` column, the resulting CAAR is the
        per-event regression-slope statistic in the
        Sefcik-Thompson (1986) lineage rather than the equal-weighted
        MacKinlay CAAR.

    References:
        [MacKinlay 1997][mackinlay-1997]: standardised event-window /
        estimation-window vocabulary inherited by ``EventConfig``.
        [Sefcik-Thompson 1986][sefcik-thompson-1986]: per-event
        regression-slope ancestor of the magnitude-weighted CAAR
        produced when ``factor`` is continuous.
        [Brown-Warner 1985][brown-warner-1985]: daily event-study
        methodology backing the parametric-test path.
    """
    return (
        df.filter(pl.col(factor_col) != 0)
        .with_columns(
            (pl.col(return_col) * pl.col(factor_col)).alias("_signed_car")
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
    r"""CAAR significance: is mean CAAR significantly different from zero?

    Args:
        caar_df: Output of ``compute_caar()`` with columns ``date, caar``.
        forward_periods: Sampling interval for non-overlapping dates.
            Maps to ``config.forward_periods`` — the return horizon used
            in ``compute_forward_return``. Distinct from
            ``EventConfig.event_window_post`` which controls MFE/MAE.

    Returns:
        MetricOutput with value=mean CAAR, stat=t from non-overlapping sampling.

    Notes:
        $t = \mathrm{mean}(\mathrm{CAAR}) / (\mathrm{std}(\mathrm{CAAR}) / \sqrt{n})$
        on a non-overlap subsample (stride ``forward_periods``) of the
        per-event-date $\mathrm{CAAR}$ series;
        $H_0: \mathbb{E}[\mathrm{CAAR}] = 0$.

        factrix uses non-overlap resampling rather than NW HAC for the
        default CAAR test — the same convention as ``ic`` — and exposes
        ``bmp_test`` as the variance-robust sibling for event-induced
        variance regimes.

    References:
        [Brown-Warner 1985][brown-warner-1985]: daily event-study
        t-test specification at standard sample sizes.
        [MacKinlay 1997][mackinlay-1997]: event-window vocabulary.
    """
    vals = caar_df["caar"].drop_nulls()
    n = len(vals)
    raw_min = _scaled_min_periods(MIN_EVENTS, forward_periods)
    if n < raw_min:
        return _short_circuit_output(
            "caar", "insufficient_event_dates",
            n_observed=n, min_required=raw_min,
            forward_periods=forward_periods,
        )

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
    kolari_pynnonen_adjust: bool = False,
) -> MetricOutput:
    r"""Boehmer-Musumeci-Poulsen Standardized Abnormal Return test.

    Standardizes each event's abnormal return by the asset's pre-event
    residual volatility, making the test robust to event-induced variance
    inflation that biases the ordinary CAAR $t$-test.

    Steps:
        1. For each event ($\text{factor} \neq 0$), look back
           ``estimation_window`` periods of the same asset's returns to
           estimate $\sigma_i$.
        2. Scale $\sigma_i$ to match the forward_return horizon.
        3. $\mathrm{SAR}_i = \mathrm{AR}^{\mathrm{signed}}_i / \sigma^{\text{scaled}}_i$.
        4. $z = \mathrm{mean}(\mathrm{SAR}) / (\mathrm{std}(\mathrm{SAR}) / \sqrt{N})$.

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
        kolari_pynnonen_adjust: When True, apply the Kolari-Pynnönen
            (2010) adjustment for cross-sectional correlation of SAR:
            $z_{\mathrm{KP}} = z_{\mathrm{BMP}} \cdot \sqrt{(1 - \hat r) / (1 + (N_{\mathrm{eff}} - 1) \cdot \hat r)}$
            where $\hat r$ is the ICC-style within-date correlation of
            SAR and
            ``N_eff`` is the average events per event date. Vanilla BMP
            overstates significance when events cluster on the same
            date (earnings season, macro release), inflating z by
            factors of 1.5-2×. Enable this when the event-study
            ``clustering_hhi`` diagnostic is high (≥ 0.3) or when you
            otherwise expect same-date shock sharing.

    Returns:
        MetricOutput(name="bmp_test", value=mean_SAR, stat=z_bmp, ...).

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

    References:
        [Boehmer-Musumeci-Poulsen 1991][boehmer-musumeci-poulsen-1991]:
        the BMP standardised AR test.
        [Kolari-Pynnönen 2010][kolari-pynnonen-2010]: clustering-
        adjusted BMP variant referenced by
        ``EventConfig.adjust_clustering`` (not yet implemented).
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
        return _short_circuit_output(
            "bmp_test", "no_events",
            n_observed=0, min_required=1,
        )

    events = events.with_columns(
        (pl.col(return_col) * pl.col(factor_col).sign()).alias("_signed_ar")
    )

    valid = events.filter(
        pl.col("_est_vol").is_not_null() & (pl.col("_est_vol") > EPSILON)
    )

    n_valid = len(valid)
    if n_valid < MIN_EVENTS:
        return _short_circuit_output(
            "bmp_test", "insufficient_estimation_window",
            n_observed=n_valid, min_required=MIN_EVENTS,
        )

    valid = valid.with_columns(
        (pl.col("_signed_ar") / pl.col("_est_vol")).alias("_sar")
    )
    sar = valid["_sar"].to_numpy()
    mean_sar = float(np.mean(sar))
    std_sar = float(np.std(sar, ddof=DDOF))

    z_bmp = _calc_t_stat(mean_sar, std_sar, n_valid)

    metadata: dict = {
        "n_events": n_valid,
        "n_dropped": len(events) - n_valid,
        "std_sar": std_sar,
        "estimation_window": estimation_window,
        "stat_type": "z",
        "h0": "mu_SAR=0",
        "method": "BMP standardized cross-sectional test",
    }

    if kolari_pynnonen_adjust:
        r_hat, n_eff, kp_source = _estimate_sar_icc(valid.select("date", "_sar"))
        metadata["kolari_pynnonen_r"] = r_hat
        metadata["kolari_pynnonen_n_eff"] = n_eff
        metadata["kolari_pynnonen_r_source"] = kp_source
        if r_hat is None or n_eff <= 1.0:
            metadata["kolari_pynnonen_applied"] = False
            z = z_bmp
        else:
            scale = float(np.sqrt((1.0 - r_hat) / (1.0 + (n_eff - 1.0) * r_hat)))
            z = z_bmp * scale
            metadata["kolari_pynnonen_scaling"] = scale
            metadata["kolari_pynnonen_applied"] = True
            metadata["stat_uncorrected"] = z_bmp
            metadata["method"] = (
                "BMP + Kolari-Pynnönen (2010) cross-sectional-correlation "
                "adjustment"
            )
    else:
        z = z_bmp

    p = _p_value_from_z(z)
    metadata["p_value"] = p

    return MetricOutput(
        name="bmp_test",
        value=mean_sar,
        stat=z,
        significance=_significance_marker(p),
        metadata=metadata,
    )


def _estimate_sar_icc(
    sar_by_date: pl.DataFrame,
) -> tuple[float | None, float, KPSource]:
    r"""ICC-style within-date correlation $\hat r$ of SAR and average cluster size.

    Args:
        sar_by_date: Event-level DataFrame with ``date`` and ``_sar``
            columns (one row per event, SAR already standardized).

    Returns ``(r_hat, n_eff, source)`` where ``source`` is one of:
        - ``"icc"``: standard between/within decomposition across event
          dates with $n_k \geq 2$ events each.
        - ``"no_multi_event_dates"``: not enough date-clusters to
          estimate within-variance; $\hat r$ is ``None``.

    Uses $\sigma^2_{\mathrm{between}} = \mathrm{var}(\overline{\mathrm{SAR}}_d)$ (date-mean SAR)
    and $\sigma^2_{\mathrm{within}}$ = the pooled within-date variance
    (weighted by $n_k - 1$).
    $\hat r = \sigma^2_{\mathrm{between}} / (\sigma^2_{\mathrm{between}} + \sigma^2_{\mathrm{within}})$
    clipped to $[0, 1]$.
    """
    per_date = sar_by_date.group_by("date").agg(
        pl.col("_sar").mean().alias("m"),
        pl.col("_sar").var(ddof=DDOF).alias("v"),
        pl.len().alias("n"),
    )
    if per_date.height == 0:
        return None, 0.0, "no_multi_event_dates"

    multi = per_date.filter(pl.col("n") >= 2)
    # n_eff must align with the subsample r̂ is estimated on: computing
    # n_eff across singleton-heavy dates and scaling by r̂ from the
    # multi-event subset conflates two different clustering regimes and
    # biases the correction downward when singletons dominate. Using the
    # multi-event mean keeps the two moments commensurate (conservative
    # on singleton-heavy datasets, as recommended by the K-P literature).
    if multi.height < 2:
        fallback_n_eff = (
            float(per_date["n"].sum() / per_date.height)
        )
        return None, fallback_n_eff, "no_multi_event_dates"
    n_eff = float(multi["n"].mean())

    w_num = (multi["v"] * (multi["n"] - 1)).sum()
    w_den = (multi["n"] - 1).sum()
    sigma2_within = float(w_num / w_den) if w_den > 0 else 0.0

    date_means = multi["m"].to_numpy()
    sigma2_between = float(np.var(date_means, ddof=DDOF))

    total = sigma2_between + sigma2_within
    if total < EPSILON:
        r_hat = 0.0
    else:
        r_hat = max(0.0, min(1.0, sigma2_between / total))
    return r_hat, n_eff, "icc"
