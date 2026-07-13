"""Shared helpers used across multiple tool modules.

These are internal utilities — not part of the public API.

Multi-factor function-name suffix taxonomy (project-wide convention):

- **(no suffix)** — batch-native unified API. Function takes
  ``factor_cols: list[str]`` and returns ``dict[str, ResultT]``; no
  single-factor sibling. Examples: ``compute_ic``, ``quantile_spread``,
  ``monotonicity``.
- **``_batch``** — batch variant that coexists with a single-factor
  sibling. Use when keeping the single-factor signature stable matters
  (callable API, third-party callers). Examples:
  ``bootstrap_mean_ci`` (1-D);
  ``_assign_quantile_groups`` (single) + ``_assign_quantile_groups_batch``
  (batch).

``_multi*`` is reserved for structural multivariate concepts (e.g.
multivariate test statistics), not for "this function handles many
factors".
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from factrix._codes import WarningCode, cross_section_tier
from factrix._errors import IncompatibleInferenceError

if TYPE_CHECKING:
    from factrix.inference import NeweyWest, NonOverlapping
    from factrix.metrics._base import MetricBase
from factrix._metric_index import SampleThreshold
from factrix._results import MetricResult, PValueAlternative
from factrix._stats import _calc_t_stat, _p_value_from_t
from factrix._stats.constants import MIN_ASSETS_WARN
from factrix._types import DDOF, EPSILON, KPSource, SampleAxis

# Median-across-dates tie_ratio above this triggers a UserWarning when
# tie_policy="ordinal". 0.3 is the empirical cutoff for "crowded" factors
# (bucketed signals, industry/size dummies routinely sit at ~0.5 — below
# 0.3 the sorting-artifact noise from ordinal tie-breaking is negligible).
TIE_RATIO_WARN_THRESHOLD = 0.3

# Aggregate share of dates a PANEL→SERIES primitive may silently drop at its
# cross-sectional filter before a metric flags it. Named after the existing
# ``TIE_RATIO_WARN_THRESHOLD`` rate-threshold convention. 0.05 is a soft floor:
# routine end-of-sample thinning sits well below it, while the 20-year-panel /
# 90%-dropped pathology the warning targets is far above.
DROP_RATE_WARN_THRESHOLD = 0.05

# Internal diagnostic column a PANEL→SERIES primitive attaches to each per-factor
# frame, carrying the canonical drop-stat struct from its ``.filter(...)`` step.
_DROP_STATS_COL = "_drop_stats"

# Fixed bootstrap seed for the small-cross-section significance path.
# A spread metric's headline p must be reproducible run-to-run, so the
# block-bootstrap branch draws from a fixed seed rather than system
# entropy; the resolved seed is echoed into metadata for the record.
_SPREAD_BOOTSTRAP_SEED = 0


def _spread_significance(
    spread: np.ndarray,
    n_assets: int,
    *,
    rng_seed: int = _SPREAD_BOOTSTRAP_SEED,
    expect_few_assets: bool = False,
) -> tuple[float, float, str, dict[str, object], tuple[str, ...]]:
    """Headline significance for a per-date long-short spread series.

    Returns ``(stat, p_value, method, extra_metadata, warning_codes)``.

    Small cross-sections (``n_assets < MIN_ASSETS_WARN``) switch the
    headline test from the non-overlapping ``t`` to a block-bootstrap
    CI: with few names per leg the per-date spread is heavy-tailed, so
    the ``t``-test's normality assumption is unreliable while the
    block bootstrap (Politis-Romano stationary scheme) is
    distribution-free and preserves any residual serial dependence.
    Larger cross-sections keep the cheaper ``t``-test. ``stat`` is the
    ``t`` statistic in both branches as a descriptive standardised
    effect size; in the bootstrap branch ``p_value`` comes from the
    resampling (``extra_metadata["p_value_t"]`` keeps the parametric ``p``
    for reference). The chosen path is named in the returned ``method``.

    The switch fires exactly when :func:`cross_section_tier` flags the
    cross-section (``n_assets < MIN_ASSETS_WARN``), so the returned
    ``warning_codes`` carry the single ``FEW_ASSETS`` code. This
    surfaces the method change as a :class:`Warning` on the result rather
    than leaving it buried in metadata; severity is read from ``n_assets``.

    ``expect_few_assets=True`` declares the thin regime as the study's
    design: the switch itself is unchanged and stays readable
    (``method`` / bootstrap metadata, plus a ``few_assets_expected``
    marker), but the ``FEW_ASSETS`` code is not emitted — the caller
    declared the expectation, so the warning would be noise, not signal.
    """
    n = len(spread)
    mean = float(np.mean(spread))
    std = float(np.std(spread, ddof=DDOF))
    t = _calc_t_stat(mean, std, n)

    if n_assets >= MIN_ASSETS_WARN:
        return t, _p_value_from_t(t, n), "non-overlapping t-test", {}, ()

    from factrix._stats.bootstrap import _block_bootstrap_diff_p

    p_boot, boot_meta = _block_bootstrap_diff_p(spread, rng_seed=rng_seed)
    extra: dict[str, object] = {
        "p_value_t": _p_value_from_t(t, n),
        "bootstrap_block_length": boot_meta["block_length"],
        "bootstrap_n_resamples": boot_meta["n_resamples"],
        "bootstrap_seed": boot_meta["rng_seed"],
    }
    if expect_few_assets:
        extra["few_assets_expected"] = True
        return t, float(p_boot), "block-bootstrap CI", extra, ()
    tier: WarningCode | None = cross_section_tier(n_assets)
    codes = (tier.value,) if tier is not None else ()
    return t, float(p_boot), "block-bootstrap CI", extra, codes


def _check_applicable_inference(
    inference: object,
    applicable: frozenset[NonOverlapping | NeweyWest],
    *,
    func_name: str,
) -> None:
    """Reject an ``inference=`` outside the metric's allowlist.

    Single chokepoint every ``inference=``-bearing metric calls before it
    dispatches: membership is by value (the members are frozen
    dataclasses), so it catches both a non-vetted ``Inference``
    (``HansenHodrick``) and a non-``Inference`` object (a stray string)
    without the metric body reaching an unintended ``compute`` or a silent
    non-overlap fallback. Raises :class:`IncompatibleInferenceError`
    listing the allowed methods.
    """
    if inference not in applicable:
        raise IncompatibleInferenceError(
            func_name=func_name,
            value=inference,
            applicable=sorted(type(member).__name__ for member in applicable),
        )


def _spread_significance_with_inference(
    inference: NonOverlapping | NeweyWest,
    *,
    strided_spread: np.ndarray,
    full_spread: pl.DataFrame | None,
    forward_periods: int,
    n_assets: int,
    rng_seed: int = _SPREAD_BOOTSTRAP_SEED,
    expect_few_assets: bool = False,
) -> tuple[float, float, float, str, dict[str, object], tuple[str, ...]]:
    """Single headline-significance chokepoint shared by every spread metric.

    Returns ``(value, stat, p_value, method, extra_metadata, warning_codes)``
    where ``value`` is the mean-spread point estimate under the chosen
    inference: the full-sample mean for the HAC path, the non-overlap mean
    otherwise.

    Two deliberate layers — both ``quantile_spread`` and ``k_spread`` route
    here so the policy lives in one place:

    1. **Small-cross-section fallback (automatic, data-driven).** When the
       cross-section is thin (``n_assets < MIN_ASSETS_WARN``) the per-date
       spread is heavy-tailed, so the distribution-free block-bootstrap CI
       overrides whatever ``inference`` was requested — HAC / OLS-t correct
       *autocorrelation*, not tail thickness. The switch emits the
       ``FEW_ASSETS`` tier code (never silent); a requested-but-overridden
       HAC is additionally flagged ``inference_overridden``.
    2. **Inference family (user-selected).** With an adequate cross-section
       the requested member runs: ``NonOverlapping`` reproduces the
       non-overlap t **bit-for-bit** (``_t_stat_from_array`` is the same
       ``_calc_t_stat`` formula on the same strided values, so the default
       stays byte-identical), while ``NeweyWest`` keeps the *full*
       overlapping ``full_spread`` series and HAC-corrects the MA(h-1) SE.

    The two series inputs are a perf split, not duplicated logic: the cheap
    panel-stride feeds ``strided_spread`` for the common path; the full
    series is built (h× more bucketing) only when the HAC path needs it.
    ``full_spread`` is ``None`` off the HAC path; a missing one degrades to
    the non-overlap path defensively. The block bootstrap is intentionally
    *not* a family member — it is an automatic small-cross-section fallback,
    not a blind-selectable method.
    """
    from factrix.inference import NeweyWest

    strided_mean = float(np.mean(strided_spread))
    use_hac = (
        isinstance(inference, NeweyWest)
        and n_assets >= MIN_ASSETS_WARN
        and full_spread is not None
    )
    if not use_hac:
        t, p, method, extra, codes = _spread_significance(
            strided_spread,
            n_assets,
            rng_seed=rng_seed,
            expect_few_assets=expect_few_assets,
        )
        if isinstance(inference, NeweyWest):
            # Requested HAC but the small-cross-section bootstrap (or a
            # missing full series) took precedence — surface the override.
            extra = {
                **extra,
                "inference_requested": inference.summary,
                "inference_overridden": True,
            }
        return strided_mean, t, p, method, extra, codes

    assert full_spread is not None  # narrowed by use_hac
    res = inference.compute(
        full_spread, value_col="spread", forward_periods=forward_periods
    )
    full_vals = full_spread["spread"].drop_nulls()
    full_mean = float(full_vals.mean())  # type: ignore[arg-type]
    extra = {**res.metadata, "n_periods_full": len(full_vals)}
    codes = tuple(code.value for code in res.warnings)
    return full_mean, res.stat, res.p_value, inference.summary, extra, codes


def _aggregate_to_per_date(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    factor_alias: str = "_f",
    return_alias: str = "_r",
) -> pl.DataFrame:
    """Collapse a panel to one row per ``date`` (mean factor + mean return).

    For COMMON-scope factors (broadcast within date) the mean is the
    identity. For single-asset TIMESERIES it is also the identity.
    For INDIVIDUAL panels the cross-section is silently averaged —
    callers using this on time-series-only metrics document that
    aggregation in their own docstrings.
    """
    return (
        data.lazy()
        .group_by("date")
        .agg(
            pl.col(factor_col).mean().alias(factor_alias),
            pl.col(return_col).mean().alias(return_alias),
        )
        .filter(pl.col(factor_alias).is_not_null() & pl.col(return_alias).is_not_null())
        .sort("date")
        .collect()
    )


def _short_circuit_output(
    name: str,
    reason: str,
    *,
    n_obs: int | None = None,
    n_obs_axis: SampleAxis | None = None,
    descriptive: bool = False,
    alternative: PValueAlternative = "two-sided",
    **extra_metadata: object,
) -> MetricResult:
    """Canonical short-circuit ``MetricResult`` for "cannot compute".

    Reason vocabulary (matches ``_insufficient_metrics`` prefixes):
        - ``insufficient_<thing>`` — data shortage (dropped from BHY)
        - ``no_<thing>`` — missing input / missing config / missing data

    ``value=NaN`` (not 0.0) because 0.0 is a legal factor-metric outcome
    (IC exactly 0, β exactly 0, spread exactly 0) indistinguishable from
    a silent short-circuit. NaN propagates through downstream aggregations
    and plots, making data shortages impossible to misread as valid zeros.

    ``p_value=1.0`` is the conservative scalar default for callers that read the
    field directly; ``multi_factor.bhy`` drops ``insufficient_*`` placeholders
    before forming the test family, so data-shortage rows do not inflate the
    multiple-testing denominator.
    Pass ``descriptive=True`` for metrics that emit no hypothesis test
    (`oos_decay`, `clustering_hhi`, ...) so callers cannot mis-route the
    short-circuit into BHY / gate logic expecting a probability.

    Use this instead of hand-rolling ``MetricResult(value=float("nan"),
    p_value=1.0, stat=None, metadata={"reason": ..., "p_value": 1.0, ...})``.
    """
    import logging

    logger = logging.getLogger(f"factrix.metric.{name}")
    logger.info(
        "Metric %s short-circuited: %s (n_obs=%s, extra=%s)",
        name,
        reason,
        n_obs,
        extra_metadata,
    )

    metadata: dict[str, object] = {"reason": reason, **extra_metadata}
    p: float | None = None if descriptive else 1.0
    return MetricResult(
        value=float("nan"),
        p_value=p,
        alternative=None if descriptive else alternative,
        n_obs=n_obs,
        n_obs_axis=n_obs_axis,
        stat=None,
        metadata=metadata,
    )


def _all_dates_degenerate(panel: pl.DataFrame, factor_col: str) -> bool:
    """True when no date has cross-sectional variation in ``factor_col``.

    A zero-variance (constant) factor carries no ranking signal: under
    ordinal tie-breaking it manufactures a spurious spread from row order,
    and under average tie-breaking every name shares a bucket so the
    top/bottom legs are empty. Spread metrics test this per date — all
    dates degenerate — and short-circuit to an explicit no-signal result
    (:func:`_no_signal_zero_variance`) instead of ranking. Nulls are
    excluded so an all-null date counts as degenerate, not as variation.
    """
    return bool(
        panel.group_by("date")
        .agg(
            pl.col(factor_col)
            .filter(pl.col(factor_col).is_not_null())
            .n_unique()
            .alias("_n_unique")
        )
        .select((pl.col("_n_unique") <= 1).all())
        .item()
    )


def _no_signal_zero_variance(n_periods: int, **extra: object) -> MetricResult:
    """Explicit no-signal result for a zero cross-sectional variance factor.

    A constant factor produces an identically zero long-short spread, so the
    honest answer is ``value=0`` with ``t=0``, ``p=1`` — a real (if null)
    finding, not a data shortage. Returned as a normal applicable
    ``MetricResult`` (no short-circuit ``reason``) so callers do not mis-route
    it as a shortage. ``extra`` carries metric-specific descriptive metadata.
    """
    return MetricResult(
        value=0.0,
        p_value=1.0,
        alternative="two-sided",
        n_obs=n_periods,
        n_obs_axis="periods",
        stat=0.0,
        metadata={
            "n_periods": n_periods,
            "stat_type": "t",
            "h0": "mu=0",
            "method": "no-signal zero-variance factor",
            "signal_status": "no_signal_zero_variance_factor",
            **extra,
        },
    )


def _enforce_min_floor(
    metric: Any,
    name: str,
    n: int,
    reason: str,
    *,
    axis: SampleAxis = "periods",
    descriptive: bool = False,
    alternative: PValueAlternative = "two-sided",
    **extra: object,
) -> MetricResult | None:
    """Short-circuit when ``n`` falls below the metric's declared ``min_<axis>``.

    Single owner for the "read declared floor → compare → short-circuit"
    step that was hand-copied across the metric bodies. Each metric still
    computes its own ``n`` (post-sampling / post-drop-nulls / post-aggregation
    counts are metric-specific) and passes it in; this helper holds only the
    comparison and the canonical :func:`_short_circuit_output` call.

    ``metric`` is typed ``Any`` so the ``@metric``-decorator-attached
    ``sample_threshold`` is reachable without a per-call
    ``# type: ignore[attr-defined]`` (the decorator types each metric as its
    wrapped function, which has no such attribute).

    Returns the short-circuit ``MetricResult`` to propagate, or ``None`` when
    the sample clears the floor (axis ungated → always ``None``). ``descriptive``
    and any extra keyword metadata are forwarded to
    :func:`_short_circuit_output`.

    Reads ``metric.sample_threshold`` — the default-config floor baked at class
    creation. This gate holds the metric class, not a configured instance, so it
    cannot know the run-time params a scaled floor depends on. A metric whose
    floor scales with run-time params (e.g. ``forward_periods``) enforces it in
    its own body, re-deriving the floor from the same source its resolver uses,
    rather than through here.
    """
    floor = getattr(metric.sample_threshold, f"min_{axis}")
    if floor is not None and n < floor:
        return _short_circuit_output(
            name,
            reason,
            n_obs=n,
            n_obs_axis=axis,
            min_required=floor,
            descriptive=descriptive,
            alternative=alternative,
            **extra,
        )
    return None


def _enforce_scaled_floor(
    name: str,
    n_raw: int,
    base: int,
    forward_periods: int,
    reason: str,
    alternative: PValueAlternative = "two-sided",
    **extra: object,
) -> MetricResult | None:
    """Short-circuit when the *raw* (pre-sampling) date count is below the
    stride-scaled periods floor — the run-time twin of a dynamic
    ``sample_threshold`` resolver.

    A metric that sub-samples dates at stride ``forward_periods`` declares its
    floor as
    ``SampleThreshold(min_periods=_scaled_min_periods(base, forward_periods))``
    so ``inspect_data`` pre-flights ``raw_n >= base * h`` against the full panel.
    This gate re-derives that *same* floor from the *same* ``base`` and
    :func:`_scaled_min_periods` against the body's actual ``forward_periods``, so
    the pre-flight floor and the run-time floor are numerically identical (cf.
    :func:`_enforce_min_floor`, which reads the default-config floor and so cannot
    track a run-time-scaled one). Periods-axis only — the sample stride is a date
    stride; ``n_raw`` is the full date count *before* sampling.
    """
    floor = _scaled_min_periods(base, forward_periods)
    if n_raw < floor:
        return _short_circuit_output(
            name,
            reason,
            n_obs=n_raw,
            n_obs_axis="periods",
            min_required=floor,
            descriptive=False,  # every stride-sampling metric runs a hypothesis test
            alternative=alternative,
            **extra,
        )
    return None


def _scaled_periods_threshold(
    base: int, *, warn: int | None = None
) -> Callable[[MetricBase], SampleThreshold]:
    """Build a dynamic ``periods`` floor resolver scaled to the sample stride.

    The returned ``Callable[[MetricBase], SampleThreshold]`` scales ``base`` (and
    optional ``warn``) by the instance's ``forward_periods`` through
    :func:`_scaled_min_periods` — the same source the in-body
    :func:`_enforce_scaled_floor` gate reads — so a metric that sub-samples at
    that stride pre-flights and gates against one numerically identical floor.
    Pass the result to ``@metric(sample_threshold=...)``.
    """

    def _resolver(self: MetricBase) -> SampleThreshold:
        fp = self.forward_periods
        return SampleThreshold(
            min_periods=_scaled_min_periods(base, fp),
            warn_periods=None if warn is None else _scaled_min_periods(warn, fp),
        )

    return _resolver


def _warn_below_floor(
    metric: Any,
    n: int,
    message: str,
    code: WarningCode,
    *,
    axis: str = "periods",
) -> str | None:
    """Flag the degraded tier when ``n`` falls below the declared ``warn_<axis>``.

    Warn-tier companion to :func:`_enforce_min_floor`: the sample clears the
    ``min`` floor (a result is still returned) but sits below ``warn``, so the
    metric runs with a documented bias. Reads ``warn_<axis>`` via an
    ``Any``-typed ``metric`` (no per-call ``# type: ignore[attr-defined]``);
    when the floor is breached it emits ``message`` as a ``UserWarning`` and
    returns ``code.value`` for the caller to fold into the result's
    ``warning_codes``. Returns ``None`` when the warn floor is clear or ungated.

    Like :func:`_enforce_min_floor`, reads the default-config ``sample_threshold``
    only; a run-time-scaled floor is not re-derived here, so dynamic-floor metrics
    warn in-body.
    """
    warn = getattr(metric.sample_threshold, f"warn_{axis}")
    if warn is not None and n < warn:
        warnings.warn(message, UserWarning, stacklevel=3)
        return code.value
    return None


def _warn_below_scaled_floor(
    n_raw: int,
    base_warn: int,
    forward_periods: int,
    message: str,
    code: WarningCode,
) -> str | None:
    """Warn-tier twin of :func:`_enforce_scaled_floor`.

    Flags the degraded tier when the raw (pre-sampling) count falls below the
    stride-scaled warn floor, re-derived from ``base_warn`` and
    :func:`_scaled_min_periods` against the body's actual ``forward_periods`` —
    the same source the dynamic resolver's ``warn_periods`` uses — so the
    pre-flight DEGRADED tier and the run-time warning fire on one identical
    floor. Periods-axis only (the sample stride is a date stride).
    """
    warn = _scaled_min_periods(base_warn, forward_periods)
    if n_raw < warn:
        warnings.warn(message, UserWarning, stacklevel=3)
        return code.value
    return None


def _estimate_within_date_icc(
    data: pl.DataFrame, value_col: str
) -> tuple[float | None, float, KPSource]:
    r"""ICC-style within-date correlation $\hat r$ of ``value_col`` and mean cluster size.

    Shared cross-sectional-correlation estimator for same-date pooled
    observations (``bmp_z`` SAR, ``directional_hit_rate`` sign-hit indicator).
    Decomposes the variance of ``value_col`` into
    $\sigma^2_{\mathrm{between}} = \mathrm{var}(\overline{v}_d)$ (date means)
    and the $(n_k - 1)$-weighted pooled within-date variance
    $\sigma^2_{\mathrm{within}}$, returning
    $\hat r = \sigma^2_{\mathrm{between}} /
    (\sigma^2_{\mathrm{between}} + \sigma^2_{\mathrm{within}})$ clipped to
    $[0, 1]$ and the mean events-per-date.

    Args:
        data: One row per pooled observation with a ``date`` column and
            ``value_col``.
        value_col: The clustered value (already standardised / 0-1 coded).

    Returns:
        ``(r_hat, n_eff, source)``:

        - ``"icc"``: between/within decomposition across dates with
          $n_k \geq 2$ observations each.
        - ``"no_multi_event_dates"``: too few multi-observation dates to
          estimate the within-variance; $\hat r$ is ``None`` (a
          single-asset series lands here, so the caller leaves the
          statistic uncorrected).
    """
    per_date = data.group_by("date").agg(
        pl.col(value_col).mean().alias("m"),
        pl.col(value_col).var(ddof=DDOF).alias("v"),
        pl.len().alias("n"),
    )
    if per_date.height == 0:
        return None, 0.0, "no_multi_event_dates"

    multi = per_date.filter(pl.col("n") >= 2)
    # n_eff must align with the subsample r̂ is estimated on: computing
    # n_eff across singleton-heavy dates and scaling by r̂ from the
    # multi-observation subset conflates two clustering regimes and biases
    # the correction downward when singletons dominate. Using the
    # multi-observation mean keeps the two moments commensurate
    # (conservative on singleton-heavy datasets, per the K-P literature).
    if multi.height < 2:
        fallback_n_eff = float(per_date["n"].sum() / per_date.height)
        return None, fallback_n_eff, "no_multi_event_dates"
    n_eff = float(multi["n"].mean())  # type: ignore[arg-type]

    w_num = (multi["v"] * (multi["n"] - 1)).sum()
    w_den = (multi["n"] - 1).sum()
    sigma2_within = float(w_num / w_den) if w_den > 0 else 0.0  # type: ignore[operator]

    date_means = multi["m"].to_numpy()
    sigma2_between = float(np.var(date_means, ddof=DDOF))

    total = sigma2_between + sigma2_within
    r_hat = 0.0 if total < EPSILON else max(0.0, min(1.0, sigma2_between / total))
    return r_hat, n_eff, "icc"


def _kp_cluster_scale(r_hat: float, n_eff: float) -> float:
    r"""Kolari-Pynnönen (2010) cross-sectional-correlation shrinkage factor.

    $\sqrt{(1 - \hat r) / (1 + (N_{\mathrm{eff}} - 1)\,\hat r)} \le 1$: the
    multiplier that deflates a pooled test statistic for within-date
    correlation, given the ICC $\hat r$ and mean cluster size
    $N_{\mathrm{eff}}$ from :func:`_estimate_within_date_icc`. At $\hat r = 0$
    (no clustering) it is 1 — the statistic is unchanged.
    """
    return float(np.sqrt((1.0 - r_hat) / (1.0 + (n_eff - 1.0) * r_hat)))


def _pick_event_return_col(data: pl.DataFrame) -> str:
    """Return the preferred return column for event analysis.

    ``abnormal_return`` (cross-sectionally de-meaned return) is preferred
    when present; ``forward_return`` is the fallback for single-asset
    panels where de-meaning is undefined. Centralized here so event metrics
    and single-asset sparse diagnostics agree on the same choice — diverging
    would silently route the same factor through different series.
    """
    return "abnormal_return" if "abnormal_return" in data.columns else "forward_return"


def _sample_non_overlapping(
    data: pl.DataFrame,
    forward_periods: int,
) -> pl.DataFrame:
    """Keep every N-th date to produce a non-overlapping series.

    Algorithm:
        1. ``unique_dates = sort(data[date].unique())``
        2. ``sampled = unique_dates[::forward_periods]``  (every N-th)
        3. Return ``data.filter(date ∈ sampled)``

    Why: with h-period forward returns, consecutive dates' forward
    returns share h−1 bars of future data — the series has an MA(h−1)
    structure ([Hansen-Hodrick (1980)][hansen-hodrick-1980]). Sub-sampling at
    interval h breaks this dependence at the cost of throwing away h−1 of
    every h observations. This is the most conservative overlap-aware
    path on the long-horizon limit theory documented by
    [Richardson-Stock (1989)][richardson-stock-1989];
    ``_newey_west_t_test`` is the less-lossy
    alternative (keeps all obs but corrects SE).

    Logs a WARNING at ``factrix.metrics`` when the sampled series
    has < 1.5 × MIN_SERIES_PERIODS_HARD rows — downstream t-tests may be frail
    even if they don't short-circuit.

    Args:
        data: DataFrame with a ``date`` column.
        forward_periods: Sampling interval (typically equals the
            ``forward_periods`` of the forward-return column).

    Returns:
        Filtered DataFrame containing only the sampled dates; all
        other columns untouched.
    """
    from factrix._logging import get_metrics_logger
    from factrix._types import MIN_SERIES_PERIODS_HARD

    sampled_dates = data["date"].unique().sort().gather_every(forward_periods)
    result = data.filter(pl.col("date").is_in(sampled_dates.implode()))
    n_after = len(sampled_dates)
    logger = get_metrics_logger()
    logger.debug(
        "non_overlap_sample: forward_periods=%d n_dates_before=%d n_after=%d",
        forward_periods,
        data["date"].n_unique(),
        n_after,
    )
    # WARNING: post-sampling series shorter than 1.5x the usual minimum is
    # a red flag — downstream t-tests either short-circuit or operate on
    # a frail sample that silently caller-doesn't-notice.
    min_safe = int(MIN_SERIES_PERIODS_HARD * 1.5)
    if 0 < n_after < min_safe:
        logger.warning(
            "non_overlap_sample shrunk to n=%d (< %d = MIN_SERIES_PERIODS_HARD*1.5); "
            "downstream significance tests may be unreliable. "
            "forward_periods=%d",
            n_after,
            min_safe,
            forward_periods,
        )
    return result


def _sample_event_spaced(
    data: pl.DataFrame,
    forward_periods: int,
    *,
    ordinal_col: str = "date_ordinal",
) -> pl.DataFrame:
    """Greedily keep event rows ``>= forward_periods`` calendar steps apart.

    The event-date counterpart of :func:`_sample_non_overlapping`. That
    helper keeps every N-th *unique date* (index distance), which is correct
    on a calendar-dense series but mis-samples an event-only series whose
    dates are irregular: sparse events get further thinned (power loss) and
    clustered events inside one forward-return window are admitted as
    independent (iid assumption violated, ``t`` inflated).

    This pass instead walks the event dates in order and keeps an event only
    when its calendar gap — the difference in ``ordinal_col``, the position
    on the full underlying calendar — to the previously kept event is
    ``>= forward_periods``. The first event is always kept. The result is a
    maximal subset whose consecutive kept dates are at least one full
    forward-return horizon apart, so the surviving observations no longer
    share overlapping forward-return windows ([Brown-Warner (1985)][brown-warner-1985]
    non-overlap sampling, made calendar-aware for the event-date axis).

    ``forward_periods <= 1`` is a no-op (consecutive events already
    independent); an empty frame returns unchanged. ``data`` must be sorted by
    date and carry ``ordinal_col`` (``compute_caar`` emits ``date_ordinal``).

    Args:
        data: Event-date series, sorted by date, with an ``ordinal_col``
            integer column giving each date's position on the full calendar.
        forward_periods: Minimum calendar gap (in those ordinal steps)
            required between consecutive kept events.
        ordinal_col: Name of the full-calendar position column.

    Returns:
        Filtered DataFrame containing only the kept event rows; all
        columns untouched.
    """
    if forward_periods <= 1 or data.height == 0:
        return data
    ordinals = data[ordinal_col].to_numpy()
    keep = np.zeros(data.height, dtype=bool)
    last_kept: int | None = None
    for i, ordinal in enumerate(ordinals):
        if last_kept is None or ordinal - last_kept >= forward_periods:
            keep[i] = True
            last_kept = int(ordinal)
    return data.filter(pl.Series(keep))


def _scaled_min_periods(base: int, forward_periods: int) -> int:
    """Raw-sample minimum for a metric that will sub-sample at stride h.

    ``MIN_*_PERIODS`` constants are calibrated for the *effective*
    sample size the downstream t-test operates on. When the metric
    first runs ``_sample_non_overlapping(data, h)`` the effective n
    shrinks to ``raw_n / h``, so the pre-sampling guard needs
    ``raw_n ≥ base · h`` to land with ≥ ``base`` independent
    observations after sampling. Clamps ``h ≥ 1`` so ``h = 1`` is a
    no-op.
    """
    return base * max(forward_periods, 1)


def _lag_within_asset(
    data: pl.DataFrame,
    col: str,
    *,
    periods: int = 1,
    by: str = "asset_id",
) -> pl.DataFrame:
    """Replace ``col`` with its per-asset lag; drop rows where the lag is null.

    Common post-sampling pattern: after ``_sample_non_overlapping`` sorts
    the panel to the rebalance schedule, we want each row's ``col`` to
    carry the value observed one sampled period earlier on the same
    asset (weight[t-1], rank[t-1], ...). Single helper so the whole
    codebase lags the same way — sort by (asset, date), shift within
    asset, drop the first row per asset.
    """
    return (
        data.sort([by, "date"])
        .with_columns(pl.col(col).shift(periods).over(by).alias(col))
        .drop_nulls([col])
    )


def _assign_quantile_groups(
    data: pl.DataFrame,
    factor_col: str = "factor",
    n_groups: int = 5,
    tie_policy: str = "ordinal",
) -> pl.DataFrame:
    """Assign quantile group labels (0 = bottom, n_groups-1 = top) per date.

    ``tie_policy="ordinal"`` (default): break ties deterministically by
    row order → balanced group sizes, but tied assets end up in different
    buckets (arbitrary but consistent).

    ``tie_policy="average"``: tied assets share an average rank → same
    bucket → honest density resolution, group sizes may be unbalanced.
    Prefer this for low-cardinality factors (binary, bucketed, or
    categorical signals) where ordinal tie-breaking would inject
    sorting-artifact noise indistinguishable from alpha.

    Returns:
        DataFrame with ``_group`` column appended.
    """
    rank_expr = pl.col(factor_col).rank(method=tie_policy).over("date").alias("_rank")  # type: ignore[arg-type]
    return (
        data.with_columns(
            rank_expr,
            # Denominator is the per-date *non-null* factor count, not the row
            # count: a null factor gets a null rank (it never lands in a bucket),
            # so counting it would shrink every quantile width and leave the top
            # bucket unreachable (max rank / n_assets < 1).
            pl.col(factor_col).count().over("date").alias("_n"),
        )
        .with_columns(
            ((pl.col("_rank") - 1) * n_groups / pl.col("_n"))
            .cast(pl.Int32)
            .clip(0, n_groups - 1)
            .alias("_group")
        )
        .drop("_rank", "_n")
    )


def _assign_quantile_groups_batch(
    data: pl.DataFrame,
    factor_cols: list[str],
    n_groups: int,
    tie_policy: str = "ordinal",
) -> pl.DataFrame:
    """Assign per-date quantile groups for N factors in one polars pass.

    Batch counterpart of :func:`_assign_quantile_groups`. Emits
    one ``_group__<factor_col>`` column per factor; the shared
    ``pl.len().over("date")`` is computed once and reused, and every
    rank expression lands in a single ``with_columns`` so the polars
    query optimiser can fuse them. Used by the batch paths of
    ``compute_spread_series`` and ``monotonicity``; both consume the
    ``_group__<f>`` columns directly.
    """
    rank_exprs = [
        pl.col(f).rank(method=tie_policy).over("date").alias(f"_rank__{f}")  # type: ignore[arg-type]
        for f in factor_cols
    ]
    # Per-date *non-null* count is per factor: each factor may null out a
    # different set of assets, and a null-inclusive denominator would shrink the
    # quantile widths and leave the top bucket unreachable (see
    # :func:`_assign_quantile_groups`).
    n_exprs = [pl.col(f).count().over("date").alias(f"_n__{f}") for f in factor_cols]
    with_ranks = data.with_columns(*rank_exprs, *n_exprs)
    group_exprs = [
        ((pl.col(f"_rank__{f}") - 1) * n_groups / pl.col(f"_n__{f}"))
        .cast(pl.Int32)
        .clip(0, n_groups - 1)
        .alias(f"_group__{f}")
        for f in factor_cols
    ]
    return with_ranks.with_columns(*group_exprs)


def _compute_tie_ratio(
    data: pl.DataFrame,
    factor_col: str = "factor",
) -> float:
    """Median-across-dates tie ratio ``1 - n_unique / n`` for ``factor_col``.

    A float in [0, 1]: 0 means every per-date cross-section has unique
    factor values (no ties); 1 means every cross-section is fully
    degenerate. Returns ``nan`` when the panel is empty (no dates).

    Used as a diagnostic on quantile-bucketing metrics — callers log a
    warning when the return exceeds ``TIE_RATIO_WARN_THRESHOLD`` and
    stash the value in ``MetricResult.metadata["tie_ratio"]`` for
    downstream inspection.
    """
    if data.is_empty():
        return float("nan")
    per_date = (
        data.group_by("date")
        .agg(
            pl.col(factor_col).n_unique().alias("_u"),
            pl.len().alias("_n"),
        )
        .with_columns(
            (1.0 - pl.col("_u") / pl.col("_n")).alias("_tr"),
        )
    )
    med = per_date["_tr"].median()
    return float("nan") if med is None else float(med)  # type: ignore[arg-type]


def _warn_high_tie_ratio(
    ratio: float,
    metric_name: str,
    tie_policy: str,
) -> None:
    """Emit a ``UserWarning`` when median tie_ratio exceeds the threshold.

    No-op for ``tie_policy="average"`` (the policy already handles ties
    honestly — warning would be noise) or NaN ratios. Uses ``warnings.warn``
    not ``logger`` so the advisory surfaces in notebooks where root logger
    defaults to WARNING. Python's default ``"default"`` filter dedupes
    by (module, lineno, message) so sweep loops naturally emit once.
    """
    if math.isnan(ratio) or ratio <= TIE_RATIO_WARN_THRESHOLD:
        return
    if tie_policy != "ordinal":
        return
    warnings.warn(
        f"{metric_name}: median tie_ratio={ratio:.3f} exceeds "
        f"{TIE_RATIO_WARN_THRESHOLD:.2f}. Ordinal tie-breaking on a "
        f"low-cardinality factor injects sorting-artifact noise. "
        f"Consider tie_policy='average' on the Config, or a coarser "
        f"n_groups.",
        UserWarning,
        stacklevel=3,
    )


# Per-axis WarningCode for the silent-drop flag. A drop along the time axis
# raises EXCESSIVE_PERIOD_DROPS, one along the cross-section EXCESSIVE_ASSET_DROPS;
# the code is dimension-specific so a reader resolves the dropped axis from the
# code alone (the naming grammar shared with SampleThreshold / n_<axis>).
_DROP_CODE_BY_AXIS: dict[str, WarningCode] = {
    "periods": WarningCode.EXCESSIVE_PERIOD_DROPS,
    "assets": WarningCode.EXCESSIVE_ASSET_DROPS,
}


def _drop_stat_keys(axis: str = "periods") -> tuple[str, ...]:
    """Canonical drop-stat schema keys for one sample ``axis``.

    Both the drop site and the consumer that surfaces it share these exact five
    names — no per-metric ad-hoc keys. The three count keys carry the ``axis``
    token (``n_periods_*`` / ``n_assets_*`` / ``dropped_<axis>``) to align with
    the sample-dimension naming grammar (``n_<axis>`` / ``min_<axis>``); the rate
    and reason are dimension-neutral.
    """
    return (
        f"n_{axis}_in",
        f"n_{axis}_out",
        f"dropped_{axis}",
        "drop_rate",
        "drop_reason",
    )


# Periods-axis schema keys — the common case, derived from the axis-generic
# source of truth above so the two never drift.
DROP_STAT_KEYS: tuple[str, ...] = _drop_stat_keys("periods")


def _make_drop_stats(
    *,
    axis: str = "periods",
    n_in: int,
    n_out: int,
    drop_reason: str,
) -> dict[str, Any]:
    """Build the canonical five-key drop-stat dict from in/out counts on ``axis``.

    Single source of truth for the schema, shared by the carrier
    (:func:`_attach_drop_stats`) and the SERIES→SCALAR consumer null-drop
    (:func:`_surface_null_drop`). The three count keys carry the ``axis`` token
    (see :func:`_drop_stat_keys`). ``drop_rate`` is 0.0 when nothing entered.
    """
    dropped = n_in - n_out
    drop_rate = dropped / n_in if n_in > 0 else 0.0
    return {
        f"n_{axis}_in": n_in,
        f"n_{axis}_out": n_out,
        f"dropped_{axis}": dropped,
        "drop_rate": drop_rate,
        # ``drop_reason`` names the criterion that *fired*; with nothing
        # dropped there is no reason, so report null rather than the static
        # predicate label (which otherwise reads as a contradiction at
        # ``drop_rate == 0``).
        "drop_reason": drop_reason if dropped > 0 else None,
    }


def _attach_drop_stats(
    frame: pl.DataFrame,
    *,
    axis: str = "periods",
    n_in: int,
    drop_reason: str,
) -> pl.DataFrame:
    """Attach the canonical drop-stat struct to a post-filter per-factor frame.

    The producing primitive holds the pre-filter count on ``axis`` (``n_in``)
    and the predicate (``drop_reason``); ``n_<axis>_out`` is the surviving row
    count (``frame.height``). The five stats are broadcast as a single
    ``_drop_stats`` struct column so the diagnostic rides the existing
    ``dict[str, pl.DataFrame]`` contract (cf. the per-date ``tie_ratio`` column).
    A consumer reads row 0 via :func:`_read_drop_stats`; a fully-dropped
    (0-row) frame carries an empty column and is never read because the
    consumer short-circuits first.
    """
    stats = _make_drop_stats(
        axis=axis,
        n_in=n_in,
        n_out=frame.height,
        drop_reason=drop_reason,
    )
    return frame.with_columns(
        pl.struct(
            pl.lit(stats[f"n_{axis}_in"], dtype=pl.Int64).alias(f"n_{axis}_in"),
            pl.lit(stats[f"n_{axis}_out"], dtype=pl.Int64).alias(f"n_{axis}_out"),
            pl.lit(stats[f"dropped_{axis}"], dtype=pl.Int64).alias(f"dropped_{axis}"),
            pl.lit(stats["drop_rate"], dtype=pl.Float64).alias("drop_rate"),
            pl.lit(stats["drop_reason"], dtype=pl.String).alias("drop_reason"),
        ).alias(_DROP_STATS_COL)
    )


def _read_drop_stats(frame: pl.DataFrame) -> dict[str, Any] | None:
    """Return the five drop-stat values from row 0, or ``None`` if unavailable.

    ``None`` when the primitive attached no ``_drop_stats`` column (e.g. a
    hand-built series) or the frame has no surviving rows. Consumers merge the
    returned dict straight into ``MetricResult.metadata``.
    """
    if _DROP_STATS_COL not in frame.columns or frame.is_empty():
        return None
    return frame[_DROP_STATS_COL][0]


def _warn_if_high_drop_rate(
    stats: dict[str, Any], metric_name: str, *, axis: str = "periods"
) -> str | None:
    """Emit one aggregate ``UserWarning`` when the drop rate clears the floor.

    Returns the axis-specific drop ``WarningCode`` (as a string — see
    :data:`_DROP_CODE_BY_AXIS`) for the caller to append to ``warning_codes`` so
    the DAG's result-assembly boundary also records a structured ``Warning`` —
    the dual-channel pattern shared with ``_warn_below_floor``. Reads the three
    count keys via the ``axis`` token (``n_<axis>_in`` etc.); the message names
    the axis. Returns ``None`` (no warning) when ``drop_rate`` is at or below
    :data:`DROP_RATE_WARN_THRESHOLD`. Uses ``warnings.warn`` so the advisory
    surfaces in notebooks; the default filter dedupes sweep loops.
    """
    drop_rate = float(stats["drop_rate"])
    if drop_rate <= DROP_RATE_WARN_THRESHOLD:
        return None
    warnings.warn(
        f"{metric_name}: {drop_rate:.0%} of {axis} dropped "
        f"({stats[f'dropped_{axis}']}/{stats[f'n_{axis}_in']}) — "
        f"{stats['drop_reason']}. The metric was computed on the surviving "
        f"{stats[f'n_{axis}_out']} {axis}; read it against that shortened sample.",
        UserWarning,
        stacklevel=3,
    )
    return _DROP_CODE_BY_AXIS[axis].value


def _surface_drop_stats(
    frame: pl.DataFrame,
    metric_name: str,
    metadata: dict[str, Any],
    warning_codes: list[str],
    *,
    axis: str = "periods",
) -> None:
    """Copy an upstream primitive's drop-stat schema into a consumer's result.

    Single call-site shared by every carrier consumer: reads the five drop-stat
    keys off *frame*, merges them into *metadata*, and (when the drop rate clears
    :data:`DROP_RATE_WARN_THRESHOLD`) emits one aggregate ``UserWarning`` and
    appends the axis-specific drop ``WarningCode`` to *warning_codes*. The
    consumer passes the *axis* its upstream primitive dropped along (``"periods"``
    for ``compute_ic`` / ``compute_fm_betas``, ``"assets"`` for
    ``compute_common_betas``). No-op when *frame* carries no drop stats (hand-built
    series) or has no surviving rows. Call only on the success path — a consumer
    that short-circuits first defers to its own short-circuit reason, so the drop
    warning never double-fires.
    """
    stats = _read_drop_stats(frame)
    if stats is None:
        return
    metadata.update(stats)
    code = _warn_if_high_drop_rate(stats, metric_name, axis=axis)
    if code is not None:
        warning_codes.append(code)


def _surface_null_drop(
    *,
    n_periods_in: int,
    n_periods_out: int,
    drop_reason: str,
    metric_name: str,
    metadata: dict[str, Any],
    warning_codes: list[str],
) -> None:
    """Record a SERIES→SCALAR consumer's own null-drop with the shared schema.

    The Phase-2 counterpart to :func:`_surface_drop_stats`: where a PANEL→SERIES
    primitive records the drop on a carrier column, a time-indexed (period-axis)
    consumer that collapses its value series to a scalar via ``drop_nulls`` knows
    both counts locally — ``n_periods_in`` is the series length entering the
    drop, ``n_periods_out`` the count of finite observations that survive. Merges
    the five keys into *metadata* and, when ``drop_rate`` clears the threshold,
    emits one aggregate ``UserWarning`` and appends the code to *warning_codes*.
    Call only on the success path so a short-circuit defers to its own reason.

    Scoped to the period axis: every current SERIES→SCALAR null-drop site is
    time-indexed. The carrier path (:func:`_surface_drop_stats`) already carries
    an ``axis`` for the cross-section; a future EVENT-axis null-drop would
    generalise this signature then.
    """
    stats = _make_drop_stats(
        axis="periods",
        n_in=n_periods_in,
        n_out=n_periods_out,
        drop_reason=drop_reason,
    )
    metadata.update(stats)
    code = _warn_if_high_drop_rate(stats, metric_name, axis="periods")
    if code is not None:
        warning_codes.append(code)


def _resolve_series_value_col(
    series: pl.DataFrame,
    value_col: str,
    *,
    fallback_col: str = "ic",
) -> str:
    """Resolve the scalar column for direct and DAG-produced series inputs."""
    if value_col in series.columns:
        return value_col
    if value_col == "value" and fallback_col in series.columns:
        return fallback_col
    return value_col


def _is_sparse_magnitude_weighted(
    data: pl.DataFrame,
    factor_col: str = "factor",
) -> bool:
    """``True`` iff ``factor_col`` is mixed-sign and not a clean ±1 ternary.

    Sparse procedures and ``compute_caar`` accept ``{0, 1}`` event
    indicators or ``{0, R}, R ∈ ℝ`` magnitude-weighted columns. Mixed
    signs with non-unit magnitudes (e.g. ``{-2.5, 0, +1.3}``) yield the
    [Sefcik-Thompson (1986)][sefcik-thompson-1986] magnitude-weighted
    statistic rather than the [MacKinlay (1997)][mackinlay-1997] signed
    CAAR — a different estimator at finite
    samples when the negative- and positive-leg vols disagree.
    ``{-1, 0, +1}`` does not trigger (sign and weight semantics coincide
    numerically); all-non-negative columns do not trigger (no flip
    ambiguity).
    """
    nz = data.filter(pl.col(factor_col) != 0)[factor_col].unique().to_list()
    if not nz:
        return False
    has_neg = any(v < 0 for v in nz)
    has_pos = any(v > 0 for v in nz)
    if not (has_neg and has_pos):
        return False
    # Tolerance check on |v|=1: upstream casts (e.g. .sign() composed
    # with floating-point arithmetic) can produce values like
    # ``-1.0000001`` that should still register as the clean ±1
    # ternary regime. Reuses the project-wide numerical noise floor
    # ``EPSILON``.
    return not all(abs(abs(v) - 1.0) < EPSILON for v in nz)


def _event_signal_is_discrete(
    data: pl.DataFrame,
    factor_col: str = "factor",
) -> bool:
    """``True`` iff ``|factor|`` over event rows has no magnitude variance.

    Event rows are ``factor_col != 0``. A discrete ±k indicator (e.g. the
    canonical ternary ``{-1, 0, +1}`` from ``make_event_panel``) has a single
    ``|factor|`` value, so the magnitude→return rank correlation that
    ``event_ic`` measures is undefined — there is no magnitude variation to
    correlate. This is the single source of truth for that condition: both
    ``event_ic``'s run-time short-circuit and ``inspect_data``'s pre-flight
    verdict call it, so the two cannot diverge.

    Returns ``False`` for an empty event set — that is a sample shortage
    ("too few events"), handled by the event-count floor, not a discreteness
    blocker.
    """
    events = data.filter(pl.col(factor_col) != 0)
    if events.is_empty():
        return False
    abs_signal = np.abs(events[factor_col].to_numpy())
    return bool(np.ptp(abs_signal) < EPSILON)


# Below this many assets per quantile bucket, each bucket mean rests on a
# handful of names and the spread can be dominated by individual assets — the
# threshold for the thin-group advisory (warning + WarningCode.THIN_QUANTILE_GROUPS).
MIN_GROUP_ASSETS = 5


def _median_universe_size(data: pl.DataFrame) -> int:
    """Median number of unique assets per date."""
    return int(
        data.group_by("date")
        .agg(pl.col("asset_id").n_unique().alias("n"))["n"]
        .median()  # type: ignore[arg-type]
    )


def _is_thin_quantile_groups(sampled: pl.DataFrame, n_groups: int) -> bool:
    """True when the median cross-section split into ``n_groups`` buckets leaves
    fewer than :data:`MIN_GROUP_ASSETS` assets per bucket.

    Single source for the thin-group condition shared by the spread primitive's
    advisory ``warnings.warn`` and the consumer's structured
    ``WarningCode.THIN_QUANTILE_GROUPS`` (dual-channel, same threshold).
    """
    if n_groups <= 0:
        return False
    return _median_universe_size(sampled) // n_groups < MIN_GROUP_ASSETS


def _signed_car(
    data: pl.DataFrame,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> np.ndarray:
    """Compute signed CAR for event rows (factor ≠ 0).

    ``signed_car = return × sign(factor)``

    Args:
        data: Event-filtered DataFrame (factor ≠ 0 rows only).

    Returns:
        1-D numpy array of signed abnormal returns.
    """
    return data[return_col].to_numpy() * np.sign(data[factor_col].to_numpy())
