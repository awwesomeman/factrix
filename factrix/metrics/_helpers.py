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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import polars as pl

from factrix._codes import WarningCode, _emit_warning, cross_section_tier
from factrix._errors import IncompatibleInferenceError

if TYPE_CHECKING:
    from factrix.inference import (
        InferenceResult,
        NeweyWest,
        NonOverlapping,
        StationaryBootstrap,
    )
    from factrix.metrics._base import MetricBase
from factrix._metric_index import SampleThreshold
from factrix._results import MetricResult, PValueAlternative
from factrix._types import (
    DDOF,
    DEFAULT_FORWARD_PERIODS,
    DEFAULT_N_GROUPS,
    EPSILON,
    N_GROUPS_FLOOR,
    KPSource,
    SampleAxis,
    TiePolicy,
)

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


def _finite_expr(col: str) -> pl.Expr:
    """``col`` is present and finite: neither null, NaN nor ±inf.

    polars treats null (missing) and float NaN (a value) differently and
    ``drop_nulls`` / ``mean`` / ``var`` / ``cov`` / ``rank`` do not skip NaN;
    this is the one predicate every producer uses to define its sample.
    Non-float dtypes have no NaN, so the cast keeps the expression valid on
    integer factors.
    """
    c = pl.col(col).cast(pl.Float64, strict=False)
    return c.is_not_null() & c.is_finite()


def _finite_values(series: pl.Series) -> pl.Series:
    """Series-level twin of :func:`_finite_expr`: drop null, NaN and ±inf.

    polars ``drop_nulls`` keeps float NaN, and one NaN reaching ``np.mean`` /
    ``np.std`` makes ``_calc_t_stat`` return a NaN t — which withholds the
    test as ``degenerate_variance``, mislabelling missing data as a
    dispersion-free sample (or makes the block bootstrap raise). Every series
    column a metric collapses to a scalar goes through here first, and the
    resulting length is what the
    metric reports as ``n_obs``.
    """
    s = series.cast(pl.Float64, strict=False)
    return s.filter(s.is_finite())


def _check_applicable_inference(
    inference: object,
    applicable: frozenset[NonOverlapping | NeweyWest | StationaryBootstrap],
    *,
    func_name: str,
) -> None:
    """Reject an ``inference=`` outside the metric's allowlist.

    Single chokepoint every ``inference=``-bearing metric calls before it
    dispatches: it catches both a non-vetted ``Inference``
    (``HansenHodrick``) and a non-``Inference`` object (a stray string)
    without the metric body reaching an unintended ``compute`` or a silent
    non-overlap fallback. Raises :class:`IncompatibleInferenceError`
    listing the allowed methods.

    Membership is by **exact type**, not by value: ``StationaryBootstrap``
    carries resampling knobs (``n_resamples`` / ``rng``), so a configured
    instance is a different value from the allowlisted default one while
    being the same vetted method. Comparing by value would allowlist the
    method and then reject every configuration of it. Exact type (not
    ``isinstance``) so a subclass cannot ride in on a vetted base.
    """
    if type(inference) not in {type(member) for member in applicable}:
        raise IncompatibleInferenceError(
            func_name=func_name,
            value=inference,
            applicable=sorted(type(member).__name__ for member in applicable),
        )


def _surface_inference_run_metadata(
    result: InferenceResult, metadata: dict[str, object]
) -> None:
    """Copy the resampling knobs an ``Inference`` ran with into ``metadata``.

    ``n_resamples`` / ``seed`` / ``p_value_mc_se`` are defined only by a
    resampling member (``StationaryBootstrap``), and only that member emits
    them. Surfacing them keeps a reported empirical p reproducible from the
    result alone and puts its Monte-Carlo error next to it. Shared by ``ic``
    and the spread chokepoint so both report the same three keys under the
    same names. ``seed`` is ``None`` when the caller supplied a
    ``numpy.random.Generator`` — that stream is the caller's to reproduce.
    """
    for key in ("n_resamples", "seed", "p_value_mc_se"):
        if key in result.metadata:
            metadata[key] = result.metadata[key]


#: One-line echo body per :class:`WarningCode` an inference member can raise on
#: the series it tested. The member records the code on its
#: :class:`~factrix.inference.InferenceResult`; this names, in one line, what
#: the reader is being told. ``{n}`` is the sample the member actually tested.
#: Anything unmapped falls back to the code's own ``description``.
_INFERENCE_CODE_MESSAGE: dict[WarningCode, str] = {
    WarningCode.UNRELIABLE_SE_SHORT_PERIODS: (
        "the inference member tested {n} periods, below the WARN floor of "
        "{warn}; the HAC standard error on a series this short is biased. The "
        "statistic is returned but read p-values cautiously."
    ),
    WarningCode.SERIAL_CORRELATION_DETECTED: (
        "the tested per-period series is persistent beyond the overlap "
        "horizon (lag-1 autocorrelation above {autocorr} on the strided "
        "series, {n} periods). No HAC or bootstrap path is calibrated there — "
        "read the p-value against a raised hurdle or lengthen the sample."
    ),
    WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED: (
        "the resolved Bartlett bandwidth exceeds n_periods / 5 on the {n} "
        "tested periods, so the long-run variance rests on too few lag "
        "products to be stable. Read the p-value as indicative only."
    ),
    WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE: (
        "the rectangular-kernel HAC variance-of-mean came out negative on the "
        "{n} tested periods and was clamped to 0, so there is no SE to divide "
        "by and the test is withheld."
    ),
    WarningCode.DEGENERATE_VARIANCE: (
        "the {n} tested periods admit no test statistic (zero dispersion, or "
        "an SE that collapsed to zero). The value is returned; stat and "
        "p_value are withheld."
    ),
}


def _emit_inference_warnings(
    res: InferenceResult,
    *,
    label: str,
    warning_codes: list[str],
    expected_warnings: tuple[str, ...] = (),
    stacklevel: int = 4,
) -> None:
    """Record **and echo** the codes an inference member raised on its series.

    The member returns them as a ``frozenset[WarningCode]`` on its
    :class:`~factrix.inference.InferenceResult`; before this existed every
    consumer folded them into ``warning_codes`` and nothing ever reached
    stderr, so a short or degenerate sample was visible only to a reader who
    went looking at the result object. Routing them through
    :func:`~factrix._codes._emit_warning` puts them on the one channel every
    other advisory uses, under the same frame, and keeps the structured record
    unchanged.
    """
    from factrix._stats.constants import (
        MIN_PERIODS_WARN,
        PERSISTENT_SERIES_AUTOCORR,
    )

    n = res.n_obs if res.n_obs is not None else 0
    for code in sorted(res.warnings, key=lambda c: c.value):
        template = _INFERENCE_CODE_MESSAGE.get(code)
        message = (
            template.format(
                n=n, warn=MIN_PERIODS_WARN, autocorr=PERSISTENT_SERIES_AUTOCORR
            )
            if template is not None
            else code.description
        )
        _emit_warning(
            code,
            message,
            label=label,
            expected_warnings=expected_warnings,
            warning_codes=warning_codes,
            stacklevel=stacklevel,
        )


def _spread_significance_with_inference(
    inference: NonOverlapping | NeweyWest | StationaryBootstrap,
    *,
    strided_spread: pl.DataFrame,
    full_spread: pl.DataFrame | None,
    overlap_periods: int,
    n_assets: int,
    metric_name: str,
    expected_warnings: tuple[str, ...] = (),
) -> tuple[float, float, float, str, str, dict[str, object], tuple[str, ...]]:
    """Single headline-significance chokepoint shared by every spread metric.

    Returns ``(value, stat, p_value, method, stat_type, extra_metadata,
    warning_codes)`` where ``value`` is the mean-spread point estimate over
    the series the chosen member actually tested: the full overlapping series for a member
    that consumes one, the strided series otherwise.

    Both ``quantile_spread`` and ``k_spread`` route here so the policy lives
    in one place, and the routing is the member's own declaration rather than
    a type check here: ``Inference.consumes_full_series`` says whether the
    member needs every period (it corrects or resamples the dependence
    itself) or takes the pre-strided series (it sub-samples, and the metric's
    panel-stride already did that). The chosen series then goes through the
    ordinary ``compute(data, value_col=..., overlap_periods=...)`` call, the
    same polymorphic path ``ic`` uses.

    A pre-strided series is handed over with ``overlap_periods=1``: the
    metric's panel-stride already broke the MA(h-1) overlap, and re-striding
    it at ``h`` would sub-sample a second time. That also makes the member's
    own stride bookkeeping a no-op, so it is not surfaced — the metric's
    ``overlap_periods`` is the authority on the stride that was applied. This
    keeps ``NonOverlapping`` **bit-for-bit** identical to the hard-branched
    dispatch it replaced (same ``_calc_t_stat`` formula on the same values).

    A thin cross-section (``n_assets < MIN_ASSETS_WARN``) attaches
    ``FEW_ASSETS`` on either path and changes nothing else — the advisory
    says each leg's mean rests on a handful of names, so the spread is a
    noisier *estimate*, not a differently-distributed one.

    An earlier version switched thin cross-sections to a block-bootstrap
    CI on the grounds that few names per leg make the spread heavy-tailed
    and the ``t`` unreliable. Measured, that was backwards on both counts.
    The ``t``-test is size-robust to heavy tails — on t(3) input it rejects
    3–4% at a nominal 5% — while the bootstrap p carries the usual small-n
    distortion (iid input: 13.6% at ``n = 12``, 9.8% at 30, 7.4% at 60,
    5.2% only by 120) and the strided spread series is short exactly when
    ``overlap_periods`` is large. Through the public path the bootstrap
    branch rejected 8–20% against the ``t`` branch's 7–9%. And the switch
    keyed on the cross-section while the bootstrap's validity depends on
    the number of periods, so it was effectively random which estimator a
    panel got. That is why the bootstrap is offered as a member the caller
    asks for by name — measured on the spread series, see the size table in
    ``reference/statistical-methods`` — and never as an automatic switch.

    The two series inputs are a perf split, not duplicated logic: the cheap
    panel-stride feeds ``strided_spread`` for the common path; the full
    series is built (h× more bucketing) only when the requested member
    declares it consumes one. ``full_spread`` is ``None`` otherwise; a
    missing one where the member needed it degrades to the strided
    non-overlap path defensively and is flagged ``inference_overridden``.
    """
    from factrix._stats.constants import MIN_ASSETS_WARN
    from factrix.inference import NON_OVERLAPPING

    use_full = inference.consumes_full_series and full_spread is not None
    member: NonOverlapping | NeweyWest | StationaryBootstrap = (
        inference if use_full else NON_OVERLAPPING
    )
    data = full_spread if use_full else strided_spread
    assert data is not None  # narrowed by use_full
    res = member.compute(
        data, value_col="spread", overlap_periods=overlap_periods if use_full else 1
    )
    n_tested = res.n_obs if res.n_obs is not None else 0
    # ``value`` must describe the sample the test ran on, and the member that
    # ran is the only thing that knows what that sample was — a sub-sampling
    # member tested the survivors of its own stride. So the headline estimate
    # is always the member's, on every path.
    mean_spread = res.estimate if res.estimate is not None else float("nan")

    extra: dict[str, object]
    if use_full:
        extra = {**res.metadata, "n_periods_full": n_tested}
    else:
        extra = {}
        _surface_inference_run_metadata(res, extra)
        if inference.consumes_full_series:
            # Requested a member that needs the full series but none was
            # supplied — surface the degradation rather than report the
            # strided non-overlap t under the requested method's name.
            extra["inference_requested"] = inference.summary
            extra["inference_overridden"] = True
    # ``n_periods_tested`` is the sample the stat / p / value describe: the
    # full overlapping series on one path, the strided series on the other.
    extra["n_periods_tested"] = n_tested

    code_list: list[str] = []
    _emit_inference_warnings(
        res,
        label=metric_name,
        warning_codes=code_list,
        expected_warnings=expected_warnings,
    )
    tier: WarningCode | None = cross_section_tier(n_assets)
    if tier is not None:
        _emit_warning(
            tier,
            f"the median cross-section holds {n_assets} assets, below "
            f"MIN_ASSETS_WARN={MIN_ASSETS_WARN}; each bucket mean rests on a "
            f"handful of names, so the spread is a noisier estimate (not a "
            f"differently distributed one).",
            label=metric_name,
            expected_warnings=expected_warnings,
            warning_codes=code_list,
            stacklevel=3,
        )
    codes = tuple(code_list)
    # ``method`` / ``stat_type`` describe the member that actually ran, not
    # the one that was asked for: an ``inference_overridden`` degradation runs
    # the non-overlap t and must say so, and a member whose ``stat`` is not a
    # t-ratio (the bootstrap reports the observed mean under an empirical p)
    # must not be read against a t-distribution.
    return (
        mean_spread,
        res.stat,
        res.p_value,
        member.summary,
        member.test,
        extra,
        codes,
    )


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
    # ``mean`` propagates a float NaN (only nulls are skipped), so average
    # the finite values only and drop a date whose mean is still undefined.
    return (
        data.lazy()
        .group_by("date")
        .agg(
            pl.col(factor_col)
            .filter(_finite_expr(factor_col))
            .mean()
            .alias(factor_alias),
            pl.col(return_col)
            .filter(_finite_expr(return_col))
            .mean()
            .alias(return_alias),
        )
        .filter(_finite_expr(factor_alias) & _finite_expr(return_alias))
        .sort("date")
        .collect()
    )


# Axes whose sample count is a count of positions on the panel's period grid,
# and so the only ones a stale ``overlap_periods`` stamp can shrink. The
# remaining tokens ("assets", "pairs", "asset_pairs") count a cross-section,
# which no evaluation-grid stamp bears on.
_TIME_AXES: frozenset[SampleAxis] = frozenset({"periods", "events"})


def _short_circuit_output(
    name: str,
    reason: str,
    *,
    n_obs: int | None = None,
    n_obs_axis: SampleAxis | None = None,
    descriptive: bool = False,
    alternative: PValueAlternative = "two-sided",
    warning_codes: tuple[str, ...] = (),
    **extra_metadata: object,
) -> MetricResult:
    """Canonical short-circuit ``MetricResult`` for "cannot compute".

    Reason vocabulary (matches ``_insufficient_metrics`` prefixes):
        - ``insufficient_<thing>`` — data shortage (dropped from BHY)
        - ``no_<thing>`` — missing input / missing config / missing data

    A sample that is large enough but carries no dispersion is *not* a
    short-circuit: see :func:`_degenerate_test_fields`, which keeps the
    point estimate and withholds only the test.

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

    Every short-circuit carries :attr:`WarningCode.METRIC_UNAVAILABLE` on the
    result's ``warning_codes``. The bundle-level :class:`Warning` the executor
    attaches for the same event is keyed on the caller's *label*, so a caller
    holding only the ``MetricResult`` (a standalone call, a stacked
    ``to_frame`` row, a hand-rolled screen) would otherwise see a clean
    ``warning_codes=()`` on a metric that never ran. ``metadata['reason']``
    stays the specific cause. ``warning_codes=`` adds further codes naming the
    *condition* behind the reason (e.g. ``THIN_QUANTILE_GROUPS`` when the
    cross-section could not fill the requested buckets); they are appended
    after ``METRIC_UNAVAILABLE``, de-duplicated.

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
    # A data shortage on a floor that scales with the evaluation-grid overlap
    # carries one sentence on its most common false trigger: a panel
    # sub-sampled by hand after compute_forward_return, whose overlap stamp
    # still says "horizon". At overlap 1 the stamp cannot be stale in that
    # direction, and a metric that already explains itself keeps its own hint.
    # Only a shortfall on a time axis can be caused by a stale stamp: the
    # scaled floors that read the stamp gate the period and event axes, so a
    # cross-section shortfall (too few assets / pairs to rank) is unrelated to
    # it however the panel was sampled, and must not be sent down that path.
    overlap = metadata.get("overlap_periods")
    if (
        reason.startswith("insufficient_")
        and n_obs_axis in _TIME_AXES
        and "hint" not in metadata
        and isinstance(overlap, int)
        and overlap > 1
    ):
        metadata["hint"] = STALE_OVERLAP_HINT
    p: float | None = None if descriptive else 1.0
    return MetricResult(
        value=float("nan"),
        p_value=p,
        alternative=None if descriptive else alternative,
        n_obs=n_obs,
        n_obs_axis=n_obs_axis,
        stat=None,
        metadata=metadata,
        warning_codes=tuple(
            dict.fromkeys((WarningCode.METRIC_UNAVAILABLE.value, *warning_codes))
        ),
    )


def _all_dates_degenerate(panel: pl.DataFrame, factor_col: str) -> bool:
    """True when no date has cross-sectional variation in ``factor_col``.

    A zero-variance (constant) factor carries no ranking signal: under
    ordinal tie-breaking it manufactures a spurious spread from row order,
    and under average tie-breaking every name shares a bucket so the
    top/bottom legs are empty. Spread metrics test this per period — all
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


# ``metadata["signal_status"]`` value marking a result whose point estimate is
# real but whose hypothesis test does not exist. Sits alongside
# ``_no_signal_zero_variance``'s ``"no_signal_zero_variance_factor"``: both
# describe a zero-variance sample, but that one *is* a null finding (a constant
# factor makes the spread identically zero, so t=0 / p=1 is honest) while this
# one carries a non-zero point estimate no test can speak to.
DEGENERATE_SIGNAL_STATUS = "degenerate_zero_variance"


def _degenerate_test_fields(
    stat: float,
    p_value: float,
    alternative: PValueAlternative,
    metadata: dict[str, object],
    warning_codes: list[str],
) -> tuple[float | None, float | None, PValueAlternative | None]:
    """Drop the hypothesis-test fields when the sample admits no statistic.

    Pass what ``_calc_t_stat`` (or a HAC t-test) just returned together with
    the p derived from it. When ``stat`` is finite the triple comes back
    unchanged; when it is NaN the caller gets ``(None, None, None)`` and this
    stamps ``metadata["signal_status"]`` plus a ``DEGENERATE_VARIANCE``
    warning code (both containers are mutated in place, as the
    ``_surface_*`` helpers do).

    **Why the test dies but the result lives.** A zero-dispersion sample is
    not a null result: every observation identical and non-zero is degeneracy
    in the *maximum*-evidence direction (``t → ±∞``); identical and zero is an
    undefined ``0/0``. Reporting ``t = 0, p = 1`` for either — the behaviour
    this replaced — read "no predictive power" off a sample that carried
    neither. ``scipy.stats.ttest_1samp`` propagates instead (``t ≈ 1e16`` /
    ``nan``) and R's ``t.test`` refuses outright ("data are essentially
    constant"); neither lands on the null.

    So the *test* is withheld — ``p_value=None`` is the repo's existing
    "this result carries no hypothesis test" shape, shared with
    ``_short_circuit_output(descriptive=True)``. The *point estimate* is not:
    a factor that is perfectly monotonic every period has a real
    ``value = 1.0``, and a full short-circuit to ``value=NaN`` would discard a
    genuine measurement — the mirror image of the bug being fixed. Callers
    keep their normal ``value`` / ``n_obs`` / ``metadata`` and pass the
    returned triple straight into ``MetricResult``.

    ±inf is deliberately not used in place of NaN anywhere in this path: it
    would spread through serialization, aggregation and plotting as a
    legitimate extreme value.
    """
    if not math.isnan(stat):
        return stat, p_value, alternative
    metadata["signal_status"] = DEGENERATE_SIGNAL_STATUS
    code = WarningCode.DEGENERATE_VARIANCE.value
    if code not in warning_codes:
        warning_codes.append(code)
    return None, None, None


def _enforce_min_floor(
    metric: Any,
    name: str,
    n: int,
    reason: str,
    *,
    axis: SampleAxis = "periods",
    descriptive: bool = False,
    alternative: PValueAlternative = "two-sided",
    warning_codes: tuple[str, ...] = (),
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
    floor scales with run-time params (e.g. ``overlap_periods``) enforces it in
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
            warning_codes=warning_codes,
            **extra,
        )
    return None


def _enforce_scaled_floor(
    name: str,
    n_raw: int,
    base: int,
    overlap_periods: int,
    reason: str,
    alternative: PValueAlternative = "two-sided",
    warning_codes: tuple[str, ...] = (),
    *,
    axis: SampleAxis = "periods",
    descriptive: bool = False,
    **extra: object,
) -> MetricResult | None:
    """Short-circuit when the *raw* (pre-sampling) date count is below the
    stride-scaled periods floor — the run-time twin of a dynamic
    ``sample_threshold`` resolver.

    A metric that sub-samples dates at stride ``overlap_periods`` declares its
    floor as
    ``SampleThreshold(min_periods=_scaled_min_periods(base, overlap_periods))``
    so ``inspect_data`` pre-flights ``raw_n >= base * h`` against the full panel.
    This gate re-derives that *same* floor from the *same* ``base`` and
    :func:`_scaled_min_periods` against the body's actual ``overlap_periods``, so
    the pre-flight floor and the run-time floor are numerically identical (cf.
    :func:`_enforce_min_floor`, which reads the default-config floor and so cannot
    track a run-time-scaled one). ``n_raw`` is the count *before* sampling on
    ``axis`` — the date count on the ``"periods"`` axis, the event-row count on
    the ``"events"`` axis (the event battery strides its own event axis at the
    same horizon; see :func:`_sample_events_non_overlapping`).
    """
    floor = _scaled_min_periods(base, overlap_periods)
    if n_raw < floor:
        return _short_circuit_output(
            name,
            reason,
            n_obs=n_raw,
            n_obs_axis=axis,
            min_required=floor,
            # Stride-sampling metrics run a hypothesis test unless they are
            # descriptive by contract (``top_concentration``), which pass
            # ``descriptive=True`` so callers cannot mis-route the placeholder
            # p into BHY or gate logic.
            descriptive=descriptive,
            alternative=alternative,
            warning_codes=warning_codes,
            overlap_periods=overlap_periods,
            **extra,
        )
    return None


# One sentence every stride-scaled ``insufficient_*`` short-circuit carries
# (``metadata["hint"]``; echoed by ``evaluate``'s InsufficientSampleError and
# the bundle Warning). The floor scales with ``overlap_periods``, so the most
# common way to trip it on a healthy panel is a panel sub-sampled to a coarser
# evaluation grid *after* ``compute_forward_return`` — its overlap stamp still
# says "horizon" while the true overlap on that grid is smaller.
STALE_OVERLAP_HINT: str = (
    "If this panel was sub-sampled to a coarser evaluation grid after "
    "compute_forward_return, its overlap_periods stamp is stale (it still "
    "counts the horizon on the full grid); rebuild it with "
    "compute_forward_return(..., dates=<evaluation dates>) so the overlap is "
    "derived on that grid."
)


def _scaled_periods_threshold(
    base: int, *, warn: int | None = None
) -> Callable[[MetricBase], SampleThreshold]:
    """Build a dynamic ``periods`` floor resolver scaled to the sample stride.

    The returned ``Callable[[MetricBase], SampleThreshold]`` scales ``base`` (and
    optional ``warn``) by the instance's ``overlap_periods`` through
    :func:`_scaled_min_periods` — the same source the in-body
    :func:`_enforce_scaled_floor` gate reads — so a metric that sub-samples at
    that stride pre-flights and gates against one numerically identical floor.
    Pass the result to ``@metric(sample_threshold=...)``.
    """

    def _resolver(self: MetricBase) -> SampleThreshold:
        fp = self.overlap_periods
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
    label: str,
    axis: str = "periods",
    expected_warnings: tuple[str, ...] = (),
) -> str | None:
    """Flag the degraded tier when ``n`` falls below the declared ``warn_<axis>``.

    Warn-tier companion to :func:`_enforce_min_floor`: the sample clears the
    ``min`` floor (a result is still returned) but sits below ``warn``, so the
    metric runs with a documented bias. Reads ``warn_<axis>`` via an
    ``Any``-typed ``metric`` (no per-call ``# type: ignore[attr-defined]``);
    when the floor is breached it emits ``message`` through
    :func:`~factrix._codes._emit_warning` under ``label`` and returns ``code.value`` for the caller to fold into the result's
    ``warning_codes``. Returns ``None`` when the warn floor is clear or ungated.

    Like :func:`_enforce_min_floor`, reads the default-config ``sample_threshold``
    only; a run-time-scaled floor is not re-derived here, so dynamic-floor metrics
    warn in-body.

    ``expected_warnings`` is the caller's study-level declaration (injected by
    ``evaluate``): a declared code keeps its structured record — the return value
    is unchanged — and only the per-run ``UserWarning`` echo stops.
    """
    warn = getattr(metric.sample_threshold, f"warn_{axis}")
    if warn is not None and n < warn:
        return _emit_warning(
            code,
            message,
            label=label,
            expected_warnings=expected_warnings,
            stacklevel=3,
        )
    return None


def _warn_below_scaled_floor(
    n_raw: int,
    base_warn: int,
    overlap_periods: int,
    message: str,
    code: WarningCode,
    *,
    label: str,
    expected_warnings: tuple[str, ...] = (),
) -> str | None:
    """Warn-tier twin of :func:`_enforce_scaled_floor`.

    Flags the degraded tier when the raw (pre-sampling) count falls below the
    stride-scaled warn floor, re-derived from ``base_warn`` and
    :func:`_scaled_min_periods` against the body's actual ``overlap_periods`` —
    the same source the dynamic resolver's ``warn_periods`` uses — so the
    pre-flight DEGRADED tier and the run-time warning fire on one identical
    floor. Axis-agnostic: the caller supplies the raw count on whichever axis it
    strides (dates, or the event axis of the event battery).

    ``expected_warnings`` is the caller's study-level declaration (injected by
    ``evaluate``): a declared code keeps its structured record and only the
    per-run ``UserWarning`` echo stops.
    """
    warn = _scaled_min_periods(base_warn, overlap_periods)
    if n_raw < warn:
        return _emit_warning(
            code,
            message,
            label=label,
            expected_warnings=expected_warnings,
            stacklevel=3,
        )
    return None


def _estimate_within_date_icc(
    data: pl.DataFrame, value_col: str
) -> tuple[float | None, float, KPSource]:
    r"""One-way ANOVA intraclass correlation ICC(1) of ``value_col`` within dates.

    Shared cross-sectional-correlation estimator for same-period pooled
    observations (``bmp_z`` SAR, ``directional_hit_rate`` sign-hit indicator).
    With $K$ dates, $n_d$ observations on date $d$, $N = \sum n_d$:

    $$
    \mathrm{MSB} = \frac{\sum_d n_d (\bar v_d - \bar v)^2}{K - 1},\quad
    \mathrm{MSW} = \frac{\sum_d (n_d - 1) s_d^2}{N - K},\quad
    n_0 = \frac{N - \sum_d n_d^2 / N}{K - 1},
    $$

    $$
    \hat r = \frac{\mathrm{MSB} - \mathrm{MSW}}
                   {\mathrm{MSB} + (n_0 - 1)\,\mathrm{MSW}}
    $$

    clipped to $[0, 1]$ ([Shrout-Fleiss 1979][shrout-fleiss-1979] ICC(1);
    $n_0$ is the unbalanced-design cluster size of Donner-Koval). The
    returned ``n_eff`` is $n_0$.

    Why ANOVA rather than the naive ``var(date means) / total`` ratio: under
    independence the variance of a date mean is $\sigma^2_w / n_d$, not
    zero, so the naive ratio converges to $1/(n_d + 1)$ — e.g. $0.17$ at
    five names per period — and the downstream Kolari-Pynnönen deflator then
    fires at full strength on data with no clustering at all (an earlier
    factrix version did exactly this and was ~5× under-sized). The ANOVA
    estimator subtracts the within-period component and is unbiased at
    $\hat r = 0$ under independence.

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
    finite = pl.col(value_col).is_not_null() & pl.col(value_col).is_not_nan()
    per_period = (
        data.filter(finite)
        .group_by("date")
        .agg(
            pl.col(value_col).mean().alias("m"),
            pl.col(value_col).var(ddof=DDOF).alias("v"),
            pl.len().alias("n"),
        )
    )
    if per_period.height == 0:
        return None, 0.0, "no_multi_event_dates"

    n_d = per_period["n"].to_numpy().astype(float)
    m_d = per_period["m"].to_numpy().astype(float)
    k_dates = len(n_d)
    n_total = float(n_d.sum())
    n_multi = int((n_d >= 2).sum())
    # MSW needs at least one date with n_d >= 2 and MSB at least two dates;
    # a single-asset series (all singletons) lands here and the caller
    # leaves its statistic uncorrected (the canonical PT / BMP setting).
    if k_dates < 2 or n_multi < 1 or n_total - k_dates < 1:
        return None, float(n_total / k_dates), "no_multi_event_dates"

    grand = float((n_d * m_d).sum() / n_total)
    msb = float((n_d * (m_d - grand) ** 2).sum() / (k_dates - 1))
    v_d = per_period["v"].fill_null(0.0).to_numpy().astype(float)
    msw = float(((n_d - 1.0) * v_d).sum() / (n_total - k_dates))
    n0 = float((n_total - (n_d**2).sum() / n_total) / (k_dates - 1))

    denom = msb + (n0 - 1.0) * msw
    if denom < EPSILON:
        return 0.0, n0, "icc"
    r_hat = max(0.0, min(1.0, (msb - msw) / denom))
    return r_hat, n0, "icc"


# Deflation below which the clustering correction is treated as the identity:
# a scale of 0.95 moves a z of 2 to 1.9 and the two-sided p from 0.0455 to
# 0.0574 — the third decimal. Under independence the ICC(1) estimate is
# clipped at 0 but its sampling noise is not, so without a floor the deflator
# "applied" on every multi-asset run (100% of iid draws, mean scale 0.99),
# fired EVENT_CLUSTERING_ADJUSTED on nothing, and moved event_hit_rate off the
# exact binomial for a correction that did not exist.
KP_MATERIAL_SCALE: float = 0.95


def _kp_deflation_scale(r_hat: float | None, n_eff: float) -> float | None:
    """The clustering deflator when it is worth applying, else ``None``.

    ``None`` when the ICC could not be estimated, when no period holds two
    units (``n_eff <= 1``), when the estimate is non-positive (a design
    effect is a variance *inflation*; a negative sample correlation is not a
    licence to shrink the SE — the same rule ``common_beta`` applies), or when
    the deflation is immaterial (``scale >= KP_MATERIAL_SCALE``).
    """
    if r_hat is None or n_eff <= 1.0 or r_hat <= 0.0:
        return None
    scale = _kp_cluster_scale(r_hat, n_eff)
    return scale if scale < KP_MATERIAL_SCALE else None


def _kp_cluster_scale(r_hat: float, n_eff: float) -> float:
    r"""Design-effect deflator for a pooled statistic under within-period correlation.

    $1 / \sqrt{1 + (N_{\mathrm{eff}} - 1)\,\hat r} \le 1$: the multiplier
    that deflates a pooled ``mean / (sd / √N)`` statistic for within-period
    intraclass correlation $\hat r$ and cluster size $N_{\mathrm{eff}}$
    from :func:`_estimate_within_date_icc` (Kish design effect). At
    $\hat r = 0$ (no clustering) it is 1 — the statistic is unchanged.

    Why not the full Kolari-Pynnönen (2010) factor
    $\sqrt{(1 - \bar r)/(1 + (N - 1)\bar r)}$: K-P's $(1 - \bar r)$
    numerator corrects a cross-sectional variance estimated on a *single
    event period*, which
    under clustering estimates only $\sigma^2 (1 - \bar r)$. factrix pools
    SARs / hit indicators across many event periods, so the pooled variance
    already contains the between-date component and is an unbiased
    estimate of $\sigma^2$; applying the $(1 - \bar r)$ term on top
    double-counts and over-deflates. The design-effect form is the
    textbook clustered-mean variance ($\mathrm{Var}(\bar x) =
    \sigma^2 (1 + (n - 1) r) / N$) and is what the pooled statistics here
    need.
    """
    return float(1.0 / np.sqrt(1.0 + (n_eff - 1.0) * r_hat))


def _deflate_for_within_date_clustering(
    scores: pl.DataFrame,
    value_col: str,
    stat: float,
    metric_name: str,
    metadata: dict[str, Any],
    warning_codes: list[str],
    *,
    expected_warnings: tuple[str, ...] = (),
) -> float:
    r"""Deflate a pooled statistic for same-period clustering of its own score.

    The pooled event statistics — a hit count, a rank correlation — treat every
    event as an independent draw. Events sharing a period share that period's
    shock, so they are not, and the pooled statistic grows with $\sqrt{N}$ for
    no new information: an ``event_hit_rate`` on 20 assets all firing on the
    same 40 dates rejected 63.5% of true nulls at a nominal 5%.

    This is the same correction ``bmp_z`` and ``directional_hit_rate`` already
    apply, factored out: estimate the within-period intraclass correlation of
    the metric's own per-event score with :func:`_estimate_within_date_icc`,
    then scale the statistic by the Kish design effect
    :func:`_kp_cluster_scale`. With one event per period the estimate is
    undefined and the statistic is returned untouched — the correction is the
    identity exactly when there is nothing to correct.

    The estimate and the deflator are written to ``metadata`` whenever they
    exist; ``kolari_pynnonen_applied`` says whether the statistic was actually
    scaled. It is not when $\hat r \le 0$ or when the scale sits above
    ``KP_MATERIAL_SCALE`` (see :func:`_kp_deflation_scale`): an earlier
    version applied at $\hat r = 0$, scale 1.0, so the code fired on 100% of
    iid multi-asset runs and ``event_hit_rate`` left the exact binomial for
    nothing. A deflation that actually bit fires
    ``EVENT_CLUSTERING_ADJUSTED``, because the change is data-driven rather
    than configured.

    Args:
        scores: Frame with ``date`` and the per-event score column.
        value_col: The score column — the quantity whose mean drives ``stat``.
        stat: The pooled statistic to deflate.
        metric_name: For the warning text.
        metadata: Mutated with the disclosure keys.
        warning_codes: Mutated with the code when the deflator applies.
        expected_warnings: Study-level declaration; silences the echo only.

    Returns:
        The deflated statistic (or ``stat`` unchanged).
    """
    r_hat, n_eff, source = _estimate_within_date_icc(scores, value_col)
    metadata["kolari_pynnonen_r"] = r_hat
    metadata["kolari_pynnonen_n_eff"] = n_eff
    metadata["kolari_pynnonen_r_source"] = source
    scale = _kp_deflation_scale(r_hat, n_eff)
    if scale is None:
        metadata["kolari_pynnonen_applied"] = False
        if r_hat is not None and n_eff > 1.0:
            # Disclose the deflator that was judged immaterial.
            metadata["kolari_pynnonen_scaling"] = _kp_cluster_scale(r_hat, n_eff)
        return stat
    metadata["kolari_pynnonen_applied"] = True
    metadata["kolari_pynnonen_scaling"] = scale
    metadata["stat_uncorrected"] = stat
    _emit_warning(
        WarningCode.EVENT_CLUSTERING_ADJUSTED,
        f"events share periods (mean {n_eff:.1f} per event "
        f"period, within-period correlation of the per-event score "
        f"r={r_hat:.3f}), so they are not independent draws. The statistic "
        f"is deflated by the Kish design effect ({scale:.3f}) before the "
        f"p-value; the point estimate is unchanged. Uncorrected statistic "
        f"in metadata['stat_uncorrected'].",
        label=metric_name,
        expected_warnings=expected_warnings,
        warning_codes=warning_codes,
        stacklevel=3,
    )
    return stat * scale


def _pick_event_return_col(data: pl.DataFrame) -> str:
    """Return the preferred return column for event analysis.

    ``abnormal_return`` (cross-sectionally de-meaned return) is preferred
    when present; ``forward_return`` is the fallback for single-asset
    panels where de-meaning is undefined. Centralized here so event metrics
    and single-asset sparse diagnostics agree on the same choice — diverging
    would silently route the same factor through different series.
    """
    return "abnormal_return" if "abnormal_return" in data.columns else "forward_return"


def _densify_on_period_grid(
    frame: pl.DataFrame,
    *,
    grid_dates: pl.Series | None = None,
    asset_col: str = "asset_id",
) -> tuple[pl.DataFrame, bool]:
    """Reindex a panel so a row step equals one step on its own period grid.

    Every event-study window, lag and offset is a count of periods on the
    panel's distinct-date grid (CLAUDE.md, "Period grid, not calendar"), but a
    polars ``rolling_*`` window or ``shift`` counts *rows within the frame*.
    The two agree only on a dense panel. On a ragged one — an asset missing
    periods the other names have — a 30-period estimation window on an asset
    with a 20-period hole reaches 50 grid periods back, and a ``k``-offset
    return steps over the hole as though it were one period.

    This helper takes the cross product of the panel's distinct dates with its
    assets, left-joins the frame onto it and sorts by ``(asset, date)``, so an
    absent period becomes a null row rather than no row at all. Rolling and
    shift expressions evaluated on the result count grid periods for every
    asset, and a period missing inside a window counts as missing — polars
    skips nulls and honours ``min_samples`` — instead of pulling an older
    period into the window to fill it.

    The second element of the return says whether densifying added rows. An
    already-dense panel is returned merely sorted, so the dense path stays
    bit-identical to a plain per-asset rolling window; a caller that gets
    ``True`` holds a frame with filler rows and must join its derived columns
    back onto the original rows (and may fire
    :attr:`~factrix._codes.WarningCode.RAGGED_PERIOD_GRID`).

    Args:
        frame: Panel with ``date`` and, for a panel, ``asset_col``.
        grid_dates: Date column defining the grid; defaults to ``frame``'s own
            distinct dates.
        asset_col: Asset identifier; a frame without it is one series.

    Returns:
        ``(dense_frame, densified)``.
    """
    sort_keys = [k for k in (asset_col, "date") if k in frame.columns]
    ordered = frame.sort(sort_keys) if sort_keys else frame
    if "date" not in frame.columns or frame.height == 0:
        return ordered, False
    dates = (frame["date"] if grid_dates is None else grid_dates).unique().sort()
    grid = pl.DataFrame({"date": dates})
    if asset_col not in frame.columns:
        if ordered.height == grid.height:
            return ordered, False
        return grid.join(ordered, on="date", how="left").sort("date"), True
    assets = ordered[asset_col].unique().sort()
    if ordered.height == grid.height * assets.len():
        return ordered, False
    dense = (
        grid.join(pl.DataFrame({asset_col: assets}), how="cross")
        .join(ordered, on=[asset_col, "date"], how="left")
        .sort([asset_col, "date"])
    )
    return dense, True


def _ragged_event_grid_message(frame: pl.DataFrame) -> str | None:
    """The ``RAGGED_PERIOD_GRID`` echo text for a ragged panel, else ``None``.

    Single owner of the wording, split out from the recorder because the
    ``caar`` pipeline sees the panel in ``compute_caar`` and the warning codes
    one node later in :func:`~factrix.metrics.caar.caar`; the message rides
    between them as a broadcast column. Returns the message **body** only —
    the metric label and the code token are added by
    :func:`~factrix._codes._emit_warning` at the recording end.
    """
    if "asset_id" not in frame.columns or "date" not in frame.columns:
        return None
    n_periods = int(frame["date"].n_unique())
    per_asset = frame.group_by("asset_id").agg(pl.col("date").n_unique().alias("_n"))
    n_ragged = int((per_asset["_n"] < n_periods).sum())
    if not n_ragged:
        return None
    return (
        f"{n_ragged} of "
        f"{per_asset.height} assets are missing periods that others have "
        f"(panel grid: {n_periods} periods). Estimation windows, lags and "
        "event offsets are counted on the panel's period grid, so those "
        "windows still span the requested number of periods; the missing "
        "periods count as missing observations inside them, leaving those "
        "assets with a smaller estimation sample than the rest. Reindex the "
        "panel onto a common grid if the estimates must be comparable across "
        "names."
    )


def _record_ragged_event_grid(
    message: str | None,
    warning_codes: list[str],
    *,
    label: str,
    expected_warnings: tuple[str, ...] = (),
) -> None:
    """Record ``RAGGED_PERIOD_GRID`` on ``warning_codes`` and echo ``message``.

    Marked, never dropped: a code the caller declared via
    ``evaluate(..., expected_warnings=(...,))`` is still appended (the record
    is kept, and the result marks it expected) — the declaration suppresses
    the ``UserWarning`` echo only.
    """
    if message is None:
        return
    _emit_warning(
        WarningCode.RAGGED_PERIOD_GRID,
        message,
        label=label,
        expected_warnings=expected_warnings,
        warning_codes=warning_codes,
        stacklevel=3,
    )


def _warn_ragged_event_grid(
    metric_name: str,
    frame: pl.DataFrame,
    warning_codes: list[str],
    *,
    expected_warnings: tuple[str, ...] = (),
) -> None:
    """Record ``RAGGED_PERIOD_GRID`` for an event path handed a ragged panel.

    The windows are correct — :func:`_densify_on_period_grid` counts them on
    the panel grid — but an asset missing periods contributes fewer usable
    returns inside a window of the same width, so its estimate rests on a
    smaller sample than a caller comparing names would assume.
    """
    _record_ragged_event_grid(
        _ragged_event_grid_message(frame),
        warning_codes,
        label=metric_name,
        expected_warnings=expected_warnings,
    )


def _attach_abnormal_return(
    data: pl.DataFrame,
    *,
    return_col: str = "forward_return",
    estimation_window: int = 60,
    overlap_periods: int = DEFAULT_FORWARD_PERIODS,
    price_col: str = "price",
    factor_col: str = "factor",
    out_col: str = "_abnormal_return",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    r"""Attach the event family's abnormal return $AR_{it} = R_{it} - E[R_{it}]$.

    Every event statistic in factrix — the CAAR $t$, the BMP $z$, the Corrado
    rank $z$, the per-event quality summaries — is defined on an *abnormal*
    return, and until this helper existed each of them silently used the raw
    forward return instead. The difference is not cosmetic: with no expected
    return subtracted, any unconditional drift is read as event alpha. On a
    20-asset panel drifting 0.08% per period with 30 events per asset at
    uniformly random dates carrying **zero** information, ``bmp_z`` returned
    $z = 5.34$ ($p = 9\times10^{-8}$); the same panel with the estimation-window
    mean subtracted returns $z = 1.15$.

    Two models, in the order the standard event-study literature prefers what
    is available ([MacKinlay (1997)][mackinlay-1997] §3):

    - **Market-adjusted**, when the panel already carries an
      ``abnormal_return`` column: that column is used as-is. It is what
      :func:`~factrix.preprocess.compute_abnormal_return` produces — the
      return less the equal-weight cross-sectional mean of the same period,
      i.e. the market-model form with $\alpha = 0$, $\beta = 1$. This is the
      branch :func:`_pick_event_return_col` was written for.
    - **Mean-adjusted** otherwise ([Brown-Warner (1985)][brown-warner-1985]):
      $AR_{it} = R_{it} - \hat\mu_i$, with $\hat\mu_i$ the mean of the same
      asset's returns over an ``estimation_window`` that **ends before the
      event window opens**. No market model is fitted — factrix does not
      require a benchmark column and will not invent one. Which returns feed
      $\hat\mu_i$ depends on what the panel carries
      (``estimation_window_source``):

      - With a ``price_col``, the mean of the one-bar returns over the
        ``estimation_window`` bars ending at $t$ — the same window ``bmp_z``
        estimates its volatility on. ``return_col`` is a per-bar average
        (:func:`~factrix.preprocess.compute_forward_return` divides by the
        horizon), so the units match with no scaling, and no lag is needed:
        the bar $(t-1, t]$ precedes the event's forward return, which opens
        at $t + 1$. This is the textbook BMP estimation window — mean and
        volatility from the same bars. An earlier version estimated
        $\hat\mu_i$ on lagged forward-return *rows* here as well; aligning
        the two windows is neutral on every Gaussian null measured and does
        **not** remove ``bmp_z``'s skew sensitivity (see its Notes: the bias
        is $E[\hat\mu / \hat\sigma] \neq 0$ from the shared window, which the
        textbook form carries too).
      - Without one, the mean of the ``return_col`` rows over the
        ``estimation_window`` rows ending at $t - h$ (lagged by
        ``overlap_periods``, for exactly the reason ``bmp_z`` lags its
        fallback volatility): a window ending at $t$ would contain the
        event's own overlapping forward return and subtract part of the
        effect from itself. The last row in that window, $t - h$, spans bars
        $(t - h + 1, t + 1]$ — up to and including the bar before the
        event's forward return opens at $t + 2$, none of which the tested
        return contains.

    The window and its lag are counted in **panel periods**: the estimation
    window is evaluated on the panel's distinct-date grid
    (:func:`_densify_on_period_grid`), so ``estimation_window`` spans that many
    grid periods for every asset and periods an asset is missing count as
    missing observations inside the window rather than stretching it further
    back. A dense panel is unaffected; on a ragged one the caller records
    :attr:`~factrix._codes.WarningCode.RAGGED_PERIOD_GRID`
    (:func:`_warn_ragged_event_grid`).

    An event whose asset has too little history for the estimation window gets
    a null ``out_col`` and is dropped by the caller's finiteness filter, the
    same way ``bmp_z`` already drops events with no estimation-window
    volatility. ``min_samples`` is ``min(20, estimation_window)`` so a caller
    that deliberately shortens the window still gets an estimate — on a ragged
    panel that floor now bites where the row-counted window used to reach back
    far enough to fill itself.

    Args:
        data: Full panel — event *and* non-event rows. The non-event rows are
            the estimation window; passing only event rows estimates the mean
            from events alone, which is not the same quantity.
        return_col: Raw return column to adjust.
        estimation_window: Panel periods of history behind $\hat\mu_i$ —
            grid periods, not rows of the asset's own frame.
        overlap_periods: Overlap horizon; the lag, in panel periods, applied
            to $\hat\mu_i$ on the return-column path.
        price_col: Price column; when present the mean is taken on one-bar
            returns derived from it.
        factor_col: Event column (``!= 0`` marks an event), read only for
            the ``estimation_window_event_share`` diagnostic.
        out_col: Name of the attached abnormal-return column.

    Returns:
        ``(frame, diagnostics)`` — the frame with ``out_col`` attached (sorted
        by asset then date when both are present), and the diagnostics every
        consumer surfaces in ``metadata``: ``abnormal_return_model``,
        ``estimation_window``, ``estimation_window_source`` (``"price"`` or
        the ``return_col`` name), ``estimation_window_lag`` (``0`` on the
        price path) and ``estimation_window_event_share`` (below).

    Notes:
        **Identifying condition, and what happens when it fails.** The
        mean-adjusted model estimates $E[R_{it}]$ from the asset's own recent
        history, and that history contains the realised forward returns of the
        asset's *earlier* events. The model is identified when the share of
        estimation-window periods that lie inside another event's forward
        window is small; then $\hat\mu_i$ is the unconditional mean. When a
        trigger fires densely, or $h$ is long, most of the window is other
        events' realised returns: event $j$'s abnormal return is then
        negatively correlated with event $i$'s for $i < j$, the cross-event
        sample variance overstates the variance of the mean abnormal return,
        and **every** consumer of this helper reads conservative — the
        opposite failure from the drift leak it exists to remove. It is a
        property of the model, not a coding error: using one-bar returns for
        $\hat\mu$ leaves it unchanged (the contamination is in the bars),
        and masking every bar inside an event window out of the estimation
        sample over-corrects (7–15% size), because at these densities almost
        nothing survives and the surviving noise is shared across events.

        Measured sizes at a nominal 5% (300 draws, iid Gaussian null,
        price-derived forward returns, 20 assets with a 5% trigger unless
        stated), ``bmp_z`` / ``corrado_rank`` / ``event_hit_rate`` / ``caar``:
        $h = 1$: 4.3 / 4.7 / 6.0 / 4.7% (share 0.05); $h = 5$: 3.7 / 4.3 /
        1.7 / 2.3% (0.21); $h = 21$: 0.3 / 2.0 / 1.7 / 5.0% (0.60); one asset
        with an 8% trigger at $h = 5$: 1.3 / 1.3 / 3.3 / 1.7% (0.33) and at
        $h = 21$: 0.0 / 0.0 / 0.7 / 0.0% (0.81); 20 names all firing on the
        same 40 periods at $h = 5$ with no shared shock: 0.7 / 2.7 / 1.0 /
        2.3% (0.33). ``caar`` is the least affected because its
        calendar-time portfolio averages same-period events before the
        variance is taken. The share in brackets is published as
        ``estimation_window_event_share`` — the mean over the tested events
        of the fraction of their window periods inside another event's
        forward window on the same asset — and every consumer fires
        ``ESTIMATION_WINDOW_CONTAMINATED`` above
        ``ESTIMATION_WINDOW_EVENT_SHARE_WARN`` (0.25), which separates the
        cells above at or below ~2% from the rest.

        **The market-adjusted branch does not close it.** Routing the same
        panels through :func:`~factrix.preprocess.compute_abnormal_return`
        (cross-sectional de-meaning, no estimation window) measured 3.0 /
        3.7 / 1.7 / 4.0% at $h = 5$ and 0.3 / 1.0 / 1.3 / 5.7% at $h = 21$
        on the iid panel — the de-meaned returns of events on different
        names with overlapping windows are negatively correlated through
        the shared cross-sectional mean, the same mechanism on the other
        axis — and on the same-period panel it removes the effect itself
        (power at a 0.15σ effect: 54 / 64 / 53 / 47% mean-adjusted, 0 / 4 /
        0 / 7% market-adjusted), because the cross-sectional mean *is* the
        event return when every name fires together. factrix therefore does
        not switch models on the share; the code is advisory, and the honest
        reading of a flagged p-value is an upper bound.
    """
    diagnostics: dict[str, Any] = {}
    if _pick_event_return_col(data) == "abnormal_return":
        diagnostics["abnormal_return_model"] = "market_adjusted_supplied"
        diagnostics["estimation_window"] = None
        diagnostics["estimation_window_source"] = None
        diagnostics["estimation_window_lag"] = None
        diagnostics["estimation_window_event_share"] = None
        return data.with_columns(pl.col("abnormal_return").alias(out_col)), diagnostics

    diagnostics["abnormal_return_model"] = "mean_adjusted"
    diagnostics["estimation_window"] = estimation_window
    sort_keys = [k for k in ("asset_id", "date") if k in data.columns]
    frame = data.sort(sort_keys) if sort_keys else data
    has_asset = "asset_id" in frame.columns
    # Every window and lag below is a count of panel periods, so they are
    # evaluated on the grid-dense frame (see _densify_on_period_grid) and the
    # result joined back onto the real rows. On a dense panel this is a no-op.
    dense, densified = _densify_on_period_grid(frame)
    uses_price = price_col in frame.columns
    if uses_price:
        bar = pl.col(price_col) / pl.col(price_col).shift(1) - 1.0
        if has_asset:
            bar = bar.over("asset_id")
        dense = dense.with_columns(bar.alias("_bar_return"))
        source_col = "_bar_return"
        lag = 0
    else:
        source_col = return_col
        lag = overlap_periods
    diagnostics["estimation_window_source"] = "price" if uses_price else return_col
    diagnostics["estimation_window_lag"] = lag
    # Mask NaN to null before the rolling mean. polars skips nulls and honours
    # min_samples, but a float NaN is a *value* to it and propagates through
    # every window that contains it — one bad cell would otherwise blank the
    # abnormal return of the next `estimation_window` events (project-wide
    # convention: NaN is masked, never fed to a rolling aggregate).
    finite_source = pl.when(_finite_expr(source_col)).then(pl.col(source_col))
    mean_expr = finite_source.rolling_mean(
        window_size=estimation_window, min_samples=min(20, estimation_window)
    )
    if lag > 0:
        mean_expr = mean_expr.shift(lag)
    if has_asset:
        mean_expr = mean_expr.over("asset_id")
    dense = dense.with_columns((pl.col(return_col) - mean_expr).alias(out_col))
    # The share reads the same windows, so it too runs on the dense frame; the
    # filler rows carry a null factor and are excluded as non-events.
    diagnostics["estimation_window_event_share"] = _estimation_window_event_share(
        dense,
        factor_col,
        out_col,
        estimation_window=estimation_window,
        overlap_periods=overlap_periods,
        lag=lag,
        bars=uses_price,
        has_asset=has_asset,
    )
    if densified:
        out = frame.join(dense.select([*sort_keys, out_col]), on=sort_keys, how="left")
    else:
        out = dense
    if uses_price and "_bar_return" in out.columns:
        out = out.drop("_bar_return")
    return out, diagnostics


def _estimation_window_event_share(
    frame: pl.DataFrame,
    factor_col: str,
    ar_col: str,
    *,
    estimation_window: int,
    overlap_periods: int,
    lag: int,
    bars: bool,
    has_asset: bool,
) -> float | None:
    """Mean share of the tested events' estimation windows inside other events'
    forward windows — the diagnostic behind ``ESTIMATION_WINDOW_CONTAMINATED``.

    A period is "inside an event's forward window" when the return it
    contributes to the estimation window overlaps the forward return
    ``(t+1, t+1+h]`` of some event at ``t`` on the same asset. On the bar
    path (``bars=True``) the period is the bar ``(s-1, s]``, inside when an
    event sits at ``s-1-h .. s-2``; on the row path the period is the row
    ``s`` whose return spans ``(s+1, s+1+h]``, inside when an event sits within
    ``h-1`` rows on either side. The indicator is then averaged over the same
    window (and lag) as the mean itself, and the result averaged over the
    event rows with a finite abnormal return. ``None`` when there are none.
    """
    if factor_col not in frame.columns:
        return None
    is_event = (_finite_expr(factor_col) & (pl.col(factor_col) != 0)).cast(pl.Float64)
    if bars:
        inside = is_event.rolling_sum(window_size=overlap_periods, min_samples=1).shift(
            2
        )
    else:
        inside = is_event.rolling_sum(
            window_size=2 * overlap_periods - 1, min_samples=1, center=True
        )
    contaminated = (inside.fill_null(0.0) > 0).cast(pl.Float64)
    share = contaminated.rolling_mean(
        window_size=estimation_window, min_samples=min(20, estimation_window)
    )
    if lag > 0:
        share = share.shift(lag)
    if has_asset:
        share = share.over("asset_id")
    tested = frame.with_columns(share.alias("_share")).filter(
        _finite_expr(factor_col)
        & (pl.col(factor_col) != 0)
        & _finite_expr(ar_col)
        & pl.col("_share").is_not_null()
    )
    if tested.height == 0:
        return None
    return float(tested["_share"].mean())  # type: ignore[arg-type]


def _warn_estimation_window_contamination(
    metric_name: str,
    metadata: dict[str, Any],
    warning_codes: list[str],
    *,
    expected_warnings: tuple[str, ...] = (),
) -> None:
    """Fire ``ESTIMATION_WINDOW_CONTAMINATED`` when the mean-adjusted model's
    identifying condition fails (see :func:`_attach_abnormal_return` Notes).

    Reads ``metadata["estimation_window_event_share"]``; nothing fires on the
    market-adjusted branch (``None``) or below
    ``ESTIMATION_WINDOW_EVENT_SHARE_WARN``. The statistic is not changed —
    the code says its null is conservative.
    """
    from factrix._types import ESTIMATION_WINDOW_EVENT_SHARE_WARN

    share = metadata.get("estimation_window_event_share")
    if share is None or share <= ESTIMATION_WINDOW_EVENT_SHARE_WARN:
        return
    _emit_warning(
        WarningCode.ESTIMATION_WINDOW_CONTAMINATED,
        f"on average {share:.0%} of each tested event's "
        f"estimation-window periods lie inside another event's "
        f"forward-return window on the same asset (advisory floor "
        f"{ESTIMATION_WINDOW_EVENT_SHARE_WARN:.0%}). The mean-adjusted "
        f"abnormal return then subtracts neighbouring events' realised "
        f"returns rather than the unconditional mean, the per-event "
        f"abnormal returns are negatively correlated, and the test is "
        f"conservative. Supply an 'abnormal_return' column "
        f"(compute_abnormal_return) when the event universe is a small "
        f"share of the panel, or read the p-value as an upper bound.",
        label=metric_name,
        expected_warnings=expected_warnings,
        warning_codes=warning_codes,
        stacklevel=3,
    )


def _stride_dates(
    data: pl.DataFrame,
    overlap_periods: int,
) -> pl.DataFrame:
    """Rows on every ``overlap_periods``-th distinct date, first-of-each-block.

    The bare stride behind :func:`_sample_non_overlapping`, split out so a
    caller that only wants to *inspect* the strided series (the persistence
    screen in ``factrix.inference.series_mean``) does not emit that helper's
    short-sample WARNING, which speaks about a t-test being run on the
    subsample.

    Striding on the distinct dates — not on row position — is load-bearing:
    the panel's period grid may be unevenly spaced, and phase must be fixed
    by the grid rather than by which rows happen to be finite.
    """
    sampled_dates = data["date"].unique().sort().gather_every(overlap_periods)
    return data.filter(pl.col("date").is_in(sampled_dates.implode()))


def _sample_non_overlapping(
    data: pl.DataFrame,
    overlap_periods: int,
) -> pl.DataFrame:
    """Keep every N-th date to produce a non-overlapping series.

    Algorithm:
        1. ``unique_dates = sort(data[date].unique())``
        2. ``sampled = unique_dates[::overlap_periods]``  (every N-th)
        3. Return ``data.filter(date ∈ sampled)``

    Why: with h-period forward returns, consecutive dates' forward
    returns share h−1 bars of future data — the series has an MA(h−1)
    structure ([Hansen-Hodrick (1980)][hansen-hodrick-1980]). Sub-sampling at
    interval h breaks this dependence at the cost of throwing away h−1 of
    every h observations. This is the most conservative overlap-aware
    path on the long-horizon limit theory documented by
    [Richardson-Stock (1989)][richardson-stock-1989];
    ``_newey_west_t_test`` keeps all observations and corrects the SE
    instead — a different trade, not a strict improvement: striding is
    calibrated in every overlapping cell measured (4.5–5.4% at a nominal
    5%), the HAC t-test 3.9–7.3%. Striding is also not the weaker test:
    measured power is equal at ``h = 5`` (.636 vs .628 at ``T = 60``) and
    better at ``h = 1`` (.703 vs .575) and at ``h = 21`` once
    ``T ≥ 240`` (.665 vs .589). What striding cannot do is handle a
    per-period series that is autocorrelated in its own right — at
    ``h = 1`` it removes nothing, and the plain t-test it falls back to
    rejects 32% on AR(0.6) input. Prefer the HAC t-test in that regime or
    when ``h`` is long relative to ``T``; prefer striding otherwise.

    Logs a WARNING at ``factrix.metrics`` when the sampled series
    has < 1.5 × MIN_SERIES_PERIODS_HARD rows — downstream t-tests may be frail
    even if they don't short-circuit.

    Args:
        data: DataFrame with a ``date`` column.
        overlap_periods: Sampling interval (typically equals the
            ``overlap_periods`` of the forward-return column).

    Returns:
        Filtered DataFrame containing only the sampled dates; all
        other columns untouched.
    """
    from factrix._logging import get_metrics_logger
    from factrix._types import MIN_SERIES_PERIODS_HARD

    result = _stride_dates(data, overlap_periods)
    n_after = result["date"].n_unique()
    logger = get_metrics_logger()
    logger.debug(
        "non_overlap_sample: overlap_periods=%d n_dates_before=%d n_after=%d",
        overlap_periods,
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
            "overlap_periods=%d",
            n_after,
            min_safe,
            overlap_periods,
        )
    return result


def _sample_event_spaced(
    data: pl.DataFrame,
    overlap_periods: int,
    *,
    ordinal_col: str = "date_ordinal",
) -> pl.DataFrame:
    """Greedily keep event rows ``>= overlap_periods`` grid steps apart.

    The event-period counterpart of :func:`_sample_non_overlapping`. That
    helper keeps every N-th *unique date* (index distance), which is correct
    on a series dense on the panel's period grid but mis-samples an event-only
    series whose dates are irregular on it: sparse events get further thinned
    (power loss) and clustered events inside one forward-return window are
    admitted as independent (iid assumption violated, ``t`` inflated).

    This pass instead walks the event periods in order and keeps an event only
    when its gap — the difference in ``ordinal_col``, the position on the full
    underlying period grid — to the previously kept event is
    ``>= overlap_periods``. The first event is always kept. The result is a
    maximal subset whose consecutive kept dates are at least one full
    forward-return horizon apart, so the surviving observations no longer
    share overlapping forward-return windows ([Brown-Warner (1985)][brown-warner-1985]
    non-overlap sampling, measured on the period grid for the event-period
    axis).

    ``overlap_periods <= 1`` is a no-op (consecutive events already
    independent); an empty frame returns unchanged. ``data`` must be sorted by
    date and carry ``ordinal_col`` (``compute_caar`` emits ``date_ordinal``).

    Args:
        data: Event-date series, sorted by date, with an ``ordinal_col``
            integer column giving each date's position on the full period grid.
        overlap_periods: Minimum gap in periods (in those ordinal steps)
            required between consecutive kept events.
        ordinal_col: Name of the period-grid position column.

    Returns:
        Filtered DataFrame containing only the kept event rows; all
        columns untouched.
    """
    if overlap_periods <= 1 or data.height == 0:
        return data
    ordinals = data[ordinal_col].to_numpy()
    keep = np.zeros(data.height, dtype=bool)
    last_kept: int | None = None
    for i, ordinal in enumerate(ordinals):
        if last_kept is None or ordinal - last_kept >= overlap_periods:
            keep[i] = True
            last_kept = int(ordinal)
    return data.filter(pl.Series(keep))


def _sample_events_non_overlapping(
    events: pl.DataFrame,
    overlap_periods: int,
    *,
    grid_dates: pl.Series | None = None,
    asset_col: str = "asset_id",
) -> pl.DataFrame:
    """Stride an event *row* frame so no asset keeps two overlapping windows.

    The panel-level entry point to :func:`_sample_event_spaced`. That helper
    walks one ordered event series; this one supplies the period-grid ordinal
    it needs and applies it **per asset**, because a forward-return window
    ``(t, t+h]`` can only overlap another window on the *same* asset.

    Why every event significance test needs it: with ``h``-period forward
    returns, two events on one asset fewer than ``h`` periods apart share
    future periods, so their returns are not independent draws. Pooling them
    inflates the test's effective sample and, with it, every cross-event
    statistic (``t``, ``z``, the binomial hit count). On a single-asset panel
    the cross-asset clustering corrections (Kolari-Pynnönen, the event-period
    collapse) have nothing to work with — time is the only clustering axis —
    so this pass is the whole discipline.

    Ordinals come from ``grid_dates`` (the full panel's date column) when
    supplied, so the gap is measured on the underlying period grid rather than
    on the sparse event dates; without it the event dates themselves are the
    grid. ``overlap_periods <= 1`` and an empty frame are no-ops.

    Args:
        events: Event rows (already filtered to ``factor != 0`` and to finite
            values), with ``date`` and — for a panel — ``asset_col``. A frame
            without ``date`` carries no gap to measure and passes through.
        overlap_periods: Minimum gap in periods required between kept events.
        grid_dates: Full-panel ``date`` column defining the period grid.
        asset_col: Asset identifier; a frame without it is treated as one asset.

    Returns:
        The kept event rows, sorted by asset then date.
    """
    if overlap_periods <= 1 or events.height == 0 or "date" not in events.columns:
        return events
    dates = events["date"] if grid_dates is None else grid_dates
    grid = pl.DataFrame({"date": dates.unique().sort()}).with_columns(
        pl.int_range(pl.len()).alias("_grid_ordinal")
    )
    joined = events.join(grid, on="date", how="left")
    if asset_col not in joined.columns:
        parts = [joined.sort("date")]
    else:
        parts = [
            part.sort("date")
            for part in joined.sort([asset_col, "date"]).partition_by(asset_col)
        ]
    kept = pl.concat(
        [
            _sample_event_spaced(part, overlap_periods, ordinal_col="_grid_ordinal")
            for part in parts
        ]
    )
    return kept.drop("_grid_ordinal")


def _warn_event_window_overlap(
    metric_name: str,
    n_events: int,
    n_sampled: int,
    overlap_periods: int,
    metadata: dict[str, Any],
    warning_codes: list[str],
    *,
    expected_warnings: tuple[str, ...] = (),
) -> None:
    """Record the overlap the event-axis spacing pass had to remove.

    Fires once per metric when :func:`_sample_events_non_overlapping` (or
    :func:`_sample_event_spaced`) dropped at least one event — i.e. some
    consecutive pair on one asset sat closer than ``overlap_periods`` and their
    ``(t, t+h]`` windows overlapped. The dropped count is the measurement, so
    it is always written to ``metadata``; the code fires only when the count is
    non-zero. Declaring it via ``evaluate(..., expected_warnings=(...,))``
    keeps the record and stops the ``UserWarning`` echo.
    """
    dropped = n_events - n_sampled
    metadata["n_events_overlapping"] = dropped
    metadata["n_events_sampled"] = n_sampled
    if dropped <= 0:
        return
    _emit_warning(
        WarningCode.EVENT_WINDOW_OVERLAP,
        f"{dropped} of {n_events} events sat within "
        f"overlap_periods={overlap_periods} of the previously kept event on "
        f"the same asset, so their forward-return windows overlapped. The "
        f"test runs on the {n_sampled} non-overlapping events that survive "
        f"the spacing pass; overlapping events are not independent draws "
        f"and pooling them inflates the statistic.",
        label=metric_name,
        expected_warnings=expected_warnings,
        warning_codes=warning_codes,
        stacklevel=3,
    )


def _event_sample_threshold(self: MetricBase) -> SampleThreshold:
    """Event floor shared by the event significance tests that stride their
    event axis at ``overlap_periods``.

    ``min_events`` stays the static math floor ``MIN_EVENTS_HARD``, enforced
    in-body against the *sampled* (post-spacing) count — pre-flight counts raw
    non-zero factor rows, which is the documented loose upper bound. The
    **warn** floor is the one that has to scale: the spacing pass keeps about
    one event in ``h`` per asset, so a raw series needs ``h`` times
    ``MIN_EVENTS_WARN`` events to land on ``MIN_EVENTS_WARN`` independent
    ones. Delegates to the same :func:`_scaled_min_periods` the in-body
    :func:`_warn_below_scaled_floor` call reads, so the pre-flight DEGRADED
    tier and the run-time warning fire on one identical floor.
    """
    from factrix._types import MIN_EVENTS_HARD, MIN_EVENTS_WARN

    return SampleThreshold(
        min_events=MIN_EVENTS_HARD,
        warn_events=_scaled_min_periods(MIN_EVENTS_WARN, self.overlap_periods),
    )


def _scaled_min_periods(base: int, overlap_periods: int) -> int:
    """Raw-sample minimum for a metric that will sub-sample at stride h.

    ``MIN_*_PERIODS`` constants are calibrated for the *effective*
    sample size the downstream t-test operates on. When the metric
    first runs ``_sample_non_overlapping(data, h)`` the effective n
    shrinks to ``raw_n / h``, so the pre-sampling guard needs
    ``raw_n ≥ base · h`` to land with ≥ ``base`` independent
    observations after sampling. Clamps ``h ≥ 1`` so ``h = 1`` is a
    no-op.
    """
    return base * max(overlap_periods, 1)


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


def _validate_choice(
    value: object,
    choices: object,
    *,
    func_name: str,
    field: str,
    docs_path: str,
) -> None:
    """Reject a closed-set knob value outside its ``Literal`` alias.

    ``choices`` is the ``Literal`` alias that annotates the knob (e.g.
    :data:`~factrix._types.ConcentrationWeight`); the legal set is read off it
    with :func:`typing.get_args`, so the annotation and the runtime check
    cannot drift apart. Every closed-set knob routes through here, which is
    what makes a typo a :class:`~factrix._errors.UserInputError` naming the
    legal values instead of a silent fall-through to whichever branch has no
    ``else`` (or a polars error naming polars' own vocabulary).
    """
    from factrix._errors import UserInputError

    allowed = get_args(choices)
    if value not in allowed:
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            candidates=allowed,
            docs_path=docs_path,
        )


def _validate_open_unit_interval(
    value: float,
    *,
    func_name: str,
    field: str,
    detail: str,
    docs_path: str,
) -> None:
    """Reject a fraction knob outside the open interval ``(0, 1)``.

    ``detail`` says what each endpoint would degenerate into, in the style of
    ``oos_decay(is_ratio=...)``.
    """
    from factrix._errors import UserInputError

    # A bool is an int to Python and a string is not a number at all; both
    # would otherwise reach the comparison and either pass (``True`` is 1) or
    # raise a bare TypeError instead of the library's own diagnostic. NaN
    # fails every comparison, so it lands here too.
    numeric = not isinstance(value, bool) and isinstance(value, int | float)
    if not numeric or not 0.0 < float(value) < 1.0:
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected=f"a fraction strictly inside (0, 1). {detail}",
            docs_path=docs_path,
        )


def _validate_factor_cols(
    factor_cols: object, *, func_name: str, docs_path: str
) -> None:
    """Reject an empty ``factor_cols`` on a batch-native metric.

    A batch metric returns one result per requested factor, so an empty
    request has no answer to return — it would hand back an empty mapping
    that reads as "every factor failed". ``evaluate`` rejects the same empty
    list at its own boundary; this is the constructor-side twin for a metric
    configured or called directly.
    """
    from factrix._errors import UserInputError

    if not list(factor_cols):  # type: ignore[call-overload]
        raise UserInputError(
            func_name=func_name,
            field="factor_cols",
            value=factor_cols,
            expected="a non-empty sequence of factor column names",
            docs_path=docs_path,
        )


def _validate_positive_count(
    value: object, *, func_name: str, field: str, detail: str, docs_path: str
) -> None:
    """Reject a count knob below 1 (a period stride, a leg size, a lag).

    The counterpart of :func:`_validate_open_unit_interval` for the integer
    knobs: one shared shape for "this is a count of periods / names, and zero
    or negative has no meaning". A bool is an int to Python and would
    otherwise pass as ``True == 1``.
    """
    from factrix._errors import UserInputError

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected=f"an integer >= 1. {detail}",
            docs_path=docs_path,
        )


def _validate_adf_threshold(
    adf_threshold: object, *, func_name: str, docs_path: str
) -> None:
    """Reject an ``adf_threshold`` outside ``(0, 1)``; ``None`` skips the gate.

    Shared by the two metrics that gate on a stationarity test: the knob is the
    p-value the augmented Dickey-Fuller test is compared against, so it is a
    probability, and ``None`` means "do not run the gate" rather than a bound.
    """
    if adf_threshold is None:
        return
    _validate_open_unit_interval(
        adf_threshold,  # type: ignore[arg-type]
        func_name=func_name,
        field="adf_threshold",
        detail=(
            "It is the p-value the augmented Dickey-Fuller test is compared "
            "against; pass None to skip the stationarity gate."
        ),
        docs_path=docs_path,
    )


def _validate_n_groups(n_groups: int, *, func_name: str, docs_path: str) -> None:
    """Reject a quantile count below :data:`~factrix._types.N_GROUPS_FLOOR`.

    The single bound every bucketing path shares, applied at the top of every
    bucketing metric so a consumer cannot accept a split its siblings reject.
    ``n_groups=1`` used to sail through ``quantile_spread`` as a spread of
    exactly zero, while ``notional_turnover`` refused the two-group book the
    spread had just priced. Raised as a :class:`UserInputError` — it is a
    caller's knob, so it fails the way every other mistyped knob does whether
    the metric was reached through ``evaluate`` or called directly.
    """
    from factrix._errors import UserInputError

    if n_groups < N_GROUPS_FLOOR:
        raise UserInputError(
            func_name=func_name,
            field="n_groups",
            value=n_groups,
            expected=(
                f"n_groups must be >= {N_GROUPS_FLOOR} (a long-short split "
                f"needs distinct top and bottom buckets)"
            ),
            docs_path=docs_path,
        )


def _assign_quantile_groups(
    data: pl.DataFrame,
    factor_col: str = "factor",
    n_groups: int = DEFAULT_N_GROUPS,
    tie_policy: TiePolicy = "ordinal",
) -> pl.DataFrame:
    """Assign quantile group labels (0 = bottom, n_groups-1 = top) per period.

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

    ``n_groups`` is validated by the calling metric (``_validate_n_groups``)
    before any data work, so the bound is not re-checked per kernel call.
    """
    # A float NaN is *not* null to polars: ``rank`` places it above every
    # finite value (top bucket) and ``count`` includes it. Treat NaN like a
    # missing factor (pandas ``qcut`` / alphalens drop it) so it never lands
    # in a leg and never shrinks the quantile width.
    finite = _finite_expr(factor_col)
    rank_expr = (
        pl.when(finite)
        .then(pl.col(factor_col))
        .rank(method=tie_policy)
        .over("date")
        .alias("_rank")
    )
    return (
        data.with_columns(
            rank_expr,
            # Denominator is the per-period *finite* factor count, not the row
            # count: a null / NaN factor gets a null rank (it never lands in a
            # bucket), so counting it would shrink every quantile width and
            # leave the top bucket unreachable (max rank / n_assets < 1).
            finite.sum().over("date").alias("_n"),
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
    tie_policy: TiePolicy = "ordinal",
) -> pl.DataFrame:
    """Assign per-period quantile groups for N factors in one polars pass.

    Batch counterpart of :func:`_assign_quantile_groups`. Emits
    one ``_group__<factor_col>`` column per factor; the shared
    ``pl.len().over("date")`` is computed once and reused, and every
    rank expression lands in a single ``with_columns`` so the polars
    query optimiser can fuse them. Used by the batch paths of
    ``compute_spread_series`` and ``monotonicity``; both consume the
    ``_group__<f>`` columns directly.

    ``n_groups`` is validated by the calling metric (``_validate_n_groups``)
    before any data work, so the bound is not re-checked per kernel call.
    """
    rank_exprs = [
        pl.when(_finite_expr(f))
        .then(pl.col(f))
        .rank(method=tie_policy)
        .over("date")
        .alias(f"_rank__{f}")
        for f in factor_cols
    ]
    # Per-period *finite* count is per factor: each factor may null / NaN out a
    # different set of assets, and a null-inclusive denominator would shrink the
    # quantile widths and leave the top bucket unreachable (see
    # :func:`_assign_quantile_groups`).
    n_exprs = [
        _finite_expr(f).sum().over("date").alias(f"_n__{f}") for f in factor_cols
    ]
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

    A float in [0, 1]: 0 means every per-period cross-section has unique
    factor values (no ties); 1 means every cross-section is fully
    degenerate. Returns ``nan`` when the panel is empty (no dates).

    Used as a diagnostic on quantile-bucketing metrics — callers log a
    warning when the return exceeds ``TIE_RATIO_WARN_THRESHOLD`` and
    stash the value in ``MetricResult.metadata["tie_ratio"]`` for
    downstream inspection.
    """
    if data.is_empty():
        return float("nan")
    # Nulls are not a "value": they must count neither as a distinct level
    # nor in the denominator, or a sparse factor reads as tied.
    finite = _finite_expr(factor_col)
    per_period = (
        data.group_by("date")
        .agg(
            pl.col(factor_col).filter(finite).n_unique().alias("_u"),
            finite.sum().alias("_n"),
        )
        .filter(pl.col("_n") > 0)
        .with_columns(
            (1.0 - pl.col("_u") / pl.col("_n")).alias("_tr"),
        )
    )
    med = per_period["_tr"].median()
    return float("nan") if med is None else float(med)  # type: ignore[arg-type]


def _warn_high_tie_ratio(
    ratio: float,
    metric_name: str,
    tie_policy: TiePolicy,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> bool:
    """Emit the high-tie advisory and report whether its condition holds.

    No-op for ``tie_policy="average"`` (the policy already handles ties
    honestly — warning would be noise) or NaN ratios. Uses ``warnings.warn``
    not ``logger`` so the advisory surfaces in notebooks where root logger
    defaults to WARNING. Python's default ``"default"`` filter dedupes
    by (module, lineno, message) so sweep loops naturally emit once. A declared
    ``HIGH_TIE_RATIO`` stops the echo only; the return value remains ``True``
    so the metric still attaches the structured record.
    """
    if math.isnan(ratio) or ratio <= TIE_RATIO_WARN_THRESHOLD:
        return False
    if tie_policy != "ordinal":
        return False
    _emit_warning(
        WarningCode.HIGH_TIE_RATIO,
        f"median tie_ratio={ratio:.3f} exceeds "
        f"{TIE_RATIO_WARN_THRESHOLD:.2f}. Ordinal tie-breaking on a "
        f"low-cardinality factor injects sorting-artifact noise. "
        f"Consider tie_policy='average' on the Config, or a coarser "
        f"n_groups.",
        label=metric_name,
        expected_warnings=expected_warnings,
        stacklevel=3,
    )
    return True


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
    ``dict[str, pl.DataFrame]`` contract (cf. the per-period ``tie_ratio`` column).
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
    stats: dict[str, Any],
    metric_name: str,
    *,
    axis: str = "periods",
    expected_warnings: tuple[str, ...] = (),
) -> str | None:
    """Emit one aggregate ``UserWarning`` when the drop rate clears the floor.

    Returns the axis-specific drop ``WarningCode`` (as a string — see
    :data:`_DROP_CODE_BY_AXIS`) for the caller to append to ``warning_codes`` so
    the DAG's result-assembly boundary also records a structured ``Warning`` —
    the dual-channel pattern shared with ``_warn_below_floor``. Reads the three
    count keys via the ``axis`` token (``n_<axis>_in`` etc.); the message names
    the axis. Returns ``None`` (no warning) when ``drop_rate`` is at or below
    :data:`DROP_RATE_WARN_THRESHOLD`. Uses ``warnings.warn`` so the advisory
    surfaces in notebooks; the default filter dedupes sweep loops. A code the
    caller declared via ``evaluate(..., expected_warnings=(...,))`` is still
    returned (the record is kept) but its echo is suppressed.
    """
    drop_rate = float(stats["drop_rate"])
    if drop_rate <= DROP_RATE_WARN_THRESHOLD:
        return None
    return _emit_warning(
        _DROP_CODE_BY_AXIS[axis],
        f"{drop_rate:.0%} of {axis} dropped "
        f"({stats[f'dropped_{axis}']}/{stats[f'n_{axis}_in']}) — "
        f"{stats['drop_reason']}. The metric was computed on the surviving "
        f"{stats[f'n_{axis}_out']} {axis}; read it against that shortened sample.",
        label=metric_name,
        expected_warnings=expected_warnings,
        stacklevel=3,
    )


def _surface_drop_stats(
    frame: pl.DataFrame,
    metric_name: str,
    metadata: dict[str, Any],
    warning_codes: list[str],
    *,
    axis: str = "periods",
    expected_warnings: tuple[str, ...] = (),
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
    code = _warn_if_high_drop_rate(
        stats, metric_name, axis=axis, expected_warnings=expected_warnings
    )
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
    expected_warnings: tuple[str, ...] = (),
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
    code = _warn_if_high_drop_rate(
        stats, metric_name, axis="periods", expected_warnings=expected_warnings
    )
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
    """Median number of unique assets per period."""
    return int(
        data.group_by("date")
        .agg(pl.col("asset_id").n_unique().alias("n"))["n"]
        .median()  # type: ignore[arg-type]
    )


def _warn_thin_quantile_groups(
    sampled: pl.DataFrame,
    n_groups: int,
    *,
    metric_name: str,
    stacklevel: int = 3,
    expected_warnings: tuple[str, ...] = (),
) -> bool:
    """Emit the thin-bucket advisory and report whether it fired.

    The ``UserWarning`` half of the dual-channel thin-group diagnostic whose
    structured twin is :data:`WarningCode.THIN_QUANTILE_GROUPS`. Lives here
    rather than inside ``compute_spread_series`` so every bucketing consumer
    raises the *same* message off the *same* threshold — the value-weighted
    spread built its buckets inline and so reported clean on a panel where
    each leg was a single name.

    ``expected_warnings`` is the caller's study-level declaration (injected by
    ``evaluate``): a declared code stops only the echo. The return value is
    unchanged, so the consumer still attaches the structured twin.
    """
    if not _is_thin_quantile_groups(sampled, n_groups):
        return False
    median_n = _median_universe_size(sampled)
    per_group = median_n // n_groups if n_groups > 0 else 0
    if n_groups > 2:
        # Coarsest split keeping ~5 assets per group (floored at the
        # long-short minimum of 2): n_groups <= median_n // 5.
        suggested = max(2, median_n // 5)
        guidance = (
            f"Reduce n_groups to ~{suggested} (≈5 assets per group), or "
            f"treat this as a fragile small-cross-section diagnostic."
        )
    else:
        guidance = (
            "This is already the coarsest long-short split; treat the "
            "spread as a fragile small-cross-section diagnostic."
        )
    _emit_warning(
        WarningCode.THIN_QUANTILE_GROUPS,
        f"Median {per_group} assets per group (n_assets={median_n}, "
        f"n_groups={n_groups}). Bucket statistics may be dominated by "
        f"individual assets. {guidance}",
        label=metric_name,
        expected_warnings=expected_warnings,
        stacklevel=stacklevel,
    )
    return True


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
