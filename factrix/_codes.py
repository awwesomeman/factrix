"""Enum codes for structured warnings.

``WarningCode`` follows the ``*Code`` suffix invariant (§7.5).
"""

from __future__ import annotations

from enum import StrEnum


class WarningCode(StrEnum):
    """Procedure-degradation flags.

    Each value carries a one-line ``description`` gloss used by API docs,
    ``MetricResult.warning_codes``, and ``EvaluationResult.warnings``.
    """

    UNRELIABLE_SE_SHORT_PERIODS = "unreliable_se_short_periods"
    EVENT_WINDOW_OVERLAP = "event_window_overlap"
    # Fired when ADF p > 0.1 on a DENSE factor (Stambaugh-style
    # persistent-regressor flag, §5.2 / §7.3). Not raised for SPARSE.
    PERSISTENT_REGRESSOR = "persistent_regressor"
    SERIAL_CORRELATION_DETECTED = "serial_correlation_detected"
    # Single cross-asset n_assets guard for PANEL common_continuous: the cross-asset
    # t-test on E[β] runs for any n_assets >= 2 (this axis never raises) but its t_crit
    # inflates as n_assets shrinks. One code flags the whole thin regime
    # (n_assets < MIN_ASSETS_WARN); severity is read from the ``n_assets``
    # metadata rather than split across separate tier members.
    FEW_ASSETS = "few_assets"
    # Fired by ``quantile_spread`` when the median cross-section split into
    # ``n_groups`` buckets leaves fewer than MIN_GROUP_ASSETS (5) assets per
    # bucket: each bucket mean rests on a handful of names, so the spread can be
    # dominated by individual assets. Advisory only — the spread is still
    # computed. Distinct from FEW_ASSETS (which keys off the absolute
    # cross-section size driving the inference switch): a wide panel cut into
    # many buckets can trip this without tripping FEW_ASSETS.
    THIN_QUANTILE_GROUPS = "thin_quantile_groups"
    # Fired when a sparse ``factor`` column carries mixed signs but is
    # not a clean ±1 ternary (e.g. ``{-2.5, 0, +1.3}``). The CAAR /
    # sparse-panel statistic is the magnitude-weighted Sefcik-Thompson
    # (1986) variant, which differs from the textbook MacKinlay (1997)
    # signed CAAR at finite samples when negative- and positive-leg
    # vols disagree. ``{-1, 0, +1}`` does not trigger — sign and weight
    # semantics coincide numerically. All-non-negative columns
    # (``{0, 1}`` / ``{0, R≥0}``) do not trigger — no flip ambiguity.
    SPARSE_MAGNITUDE_WEIGHTED = "sparse_magnitude_weighted"
    # Fired by ``caar`` (significance test) when the per-event-date series
    # length sits in ``[MIN_EVENTS_HARD, MIN_EVENTS_WARN)`` — the t-stat
    # is returned but the Brown-Warner (1985) convention treats sub-30
    # event-date counts as power-thin for the asymptotic t-distribution.
    # Below the HARD floor the primitive short-circuits to NaN instead.
    # Naming follows the ``<axis>_<condition>`` grammar; the Brown-Warner
    # method reference lives in this gloss rather than the member name.
    FEW_EVENTS = "few_events"
    # Fired by ``top_concentration`` when the per-date ratio series sits
    # in ``[MIN_PORTFOLIO_PERIODS_HARD, MIN_PORTFOLIO_PERIODS_WARN)`` —
    # the one-sided t-test on the diversification ratio is returned but
    # ``df = n - 1 < 19`` inflates t_crit relative to the asymptotic
    # cutoff. Below the HARD floor the primitive short-circuits to NaN.
    BORDERLINE_PORTFOLIO_PERIODS = "borderline_portfolio_periods"
    # Fired by ``directional_hit_rate`` when the pooled non-overlapping
    # (date, asset) directional observations sit in
    # ``[MIN_DIRECTIONAL_PAIRS_HARD, MIN_DIRECTIONAL_PAIRS_WARN)`` — the
    # Pesaran-Timmermann (1992) hit rate is returned but the normal
    # approximation to S_n is power-thin below ~30 pooled pairs. Below the
    # HARD floor the metric short-circuits to NaN instead. Named on the
    # ``pairs`` axis token — the count is pooled (date, asset) trials, not
    # periods.
    FEW_DIRECTIONAL_PAIRS = "few_directional_pairs"
    # Fired when a rectangular-kernel HAC primitive (Hansen-Hodrick 1980)
    # produces a negative variance-of-mean estimate on short / mildly
    # anti-correlated samples. Unlike the Bartlett kernel, the rectangular
    # kernel carries no PSD guarantee (Andrews 1991 §3); the primitive
    # clamps variance to 0.0 and the t-test returns t=0, p=1.0 (cannot
    # reject), the conservative direction.
    RECT_KERNEL_NEGATIVE_VARIANCE = "rect_kernel_negative_variance"

    # Fired by ``bmp_z`` when no ``price`` column is present and the
    # estimation-window volatility falls back to the per-asset rolling std of
    # ``forward_return``. Because forward_return[t] looks ahead to [t+1, t+1+h],
    # the fallback std is lagged by ``forward_periods`` so the estimation window
    # ends before the event's own forward window — but it is still a coarser,
    # horizon-overlapping volatility proxy than a daily-price std. The z-test is
    # returned; supply ``price`` for the clean estimator.
    BMP_RETURN_VOL_FALLBACK = "bmp_return_vol_fallback"

    # Fired by the DAG executor when an upstream producer short-circuited
    # (returned a NaN MetricResult with metadata["reason"]) and the
    # consumer is skipped. The downstream MetricResult carries
    # metadata["upstream"] / ["upstream_reason"] so the original cause
    # is recoverable without re-walking the dependency graph.
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"

    # Fired by the DAG executor when a metric short-circuited on its OWN
    # precondition (not an upstream producer): a missing input column /
    # config, or an insufficient sample at its own floor. The NaN
    # MetricResult carries metadata["reason"] (e.g. "no_weight_column",
    # "insufficient_periods") with the specific cause; the warning message
    # mirrors it. Distinct from UPSTREAM_UNAVAILABLE so a root failure is
    # not mislabelled as a dependency failure.
    METRIC_UNAVAILABLE = "metric_unavailable"

    # Fired by evaluate(strict=False) when a metric's declared factor cell
    # (scope / density / data structure) does not match the detected factor
    # cell. The metric is not executed and short-circuits to NaN.
    STRUCTURE_MISMATCH = "structure_mismatch"

    # Fired by inspect_data when a DENSE factor has very few distinct
    # non-null values (e.g. {-1, +1} or small regime scores). Low cardinality
    # alone is not an event contract: sparse routing still requires an
    # explicit zero non-event state and enough zero rows to clear the sparse
    # ratio threshold. Advisory only; the factor remains DENSE.
    LOW_CARDINALITY_DENSE_SIGNAL = "low_cardinality_dense_signal"

    # Fired when a sparse event metric is explicitly run on a factor with
    # zero-valued rows but sparse_ratio below the automatic SPARSE routing
    # threshold. The metric treats those zeros as non-events; callers should
    # confirm that the zero values encode the intended event contract.
    FREQUENT_EVENT_SIGNAL = "frequent_event_signal"

    # Fired by inspect_data when factor columns carry inconsistent axes.
    CROSS_FACTOR_DENSITY_MISMATCH = "cross_factor_density_mismatch"
    CROSS_FACTOR_SCOPE_MISMATCH = "cross_factor_scope_mismatch"

    # Fired by inspect_data on single-asset event-shaped data (TIMESERIES +
    # SPARSE, i.e. n_assets=1). Event-axis metrics run over the event
    # cross-section (n_events) and are usable on a single name; only a metric
    # that needs the asset cross-section (cell.structure=PANEL, e.g.
    # clustering_hhi, whose same-date event clustering is degenerate at
    # ≤1 event/date) stays unusable. The warning names those so their absence
    # from `usable` is explained, not silent. Deliberately does NOT advise
    # adding assets — pooling unrelated names mixes return-generating processes.
    SINGLE_ASSET_EVENT_DATA = "single_asset_event_data"

    # Per-axis silent-drop flags. A metric whose upstream primitive silently
    # dropped a large share of its sample at a filter raises the code for the
    # dropped axis: PERIOD_DROPS for the time axis (e.g. compute_ic dropping
    # dates with n_assets below the per-date floor), ASSET_DROPS for the
    # cross-section (e.g. compute_common_betas dropping assets with insufficient
    # history or zero factor variance). The code is dimension-specific by
    # design — a reader resolves the dropped axis from the code alone, not by
    # digging into metadata (the dimension-naming grammar shared with
    # SampleThreshold and the n_<axis> sample-size constants). A single
    # aggregate flag per metric replaces per-row noise; the exact drop-stat
    # schema (n_<axis>_in / n_<axis>_out / dropped_<axis> / drop_rate /
    # drop_reason) rides in MetricResult.metadata. Fires only when drop_rate
    # exceeds DROP_RATE_WARN_THRESHOLD.
    EXCESSIVE_PERIOD_DROPS = "excessive_period_drops"
    EXCESSIVE_ASSET_DROPS = "excessive_asset_drops"

    # Fired by by_slice when a panel is partitioned on a date-axis column
    # (one whose value varies within an asset over time, e.g. calendar year
    # or regime label) and the metric's aggregation looks across dates
    # (TS_ONLY / TS_THEN_CS / EVENT_TIME / RETURN_SPANNING). by_slice
    # evaluates each slice as an independent dataset, so a rolling window,
    # per-asset time-series regression, or event window sees truncated
    # history at the slice boundary — the per-slice value differs from the
    # full-sample value decomposed by period. Per-date metrics (CS_THEN_TS /
    # CS_SNAPSHOT) are unaffected and do not trigger. Cross-sectional
    # partitions (sector, size bucket — constant within an asset) keep each
    # asset's history intact and do not trigger.
    SLICE_BOUNDARY_TRUNCATION = "slice_boundary_truncation"

    @property
    def description(self) -> str:
        return _WARNING_DESCRIPTIONS[self]


_WARNING_DESCRIPTIONS: dict[WarningCode, str] = {}


_WARNING_DESCRIPTIONS.update(
    {
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS: "n_periods is below the WARN floor (~30); NW HAC SE may be biased. "
        "Reused across panel time-series guards (MIN_PERIODS_WARN) and "
        "primitive inference (MIN_FM_PERIODS_WARN); both default to 30.",
        WarningCode.EVENT_WINDOW_OVERLAP: "Adjacent events sit within forward_periods; AR windows overlap.",
        WarningCode.PERSISTENT_REGRESSOR: "ADF p > 0.10 on the continuous factor; β may carry Stambaugh bias.",
        WarningCode.SERIAL_CORRELATION_DETECTED: "Ljung-Box p < 0.05 on residuals; NW lag may be under-set.",
        WarningCode.FEW_ASSETS: "Cross-section asset count is below the "
        "relevant WARN floor (panel-wide MIN_ASSETS_WARN=30, per-date "
        "MIN_IC_ASSETS_WARN=10, or per-date MIN_FM_ASSETS_WARN=10). The "
        "statistic is returned, but small n_assets inflates critical values or "
        "leaves minimal residual degrees of freedom. Severity scales with "
        "n_assets; read the relevant n_assets metadata.",
        WarningCode.THIN_QUANTILE_GROUPS: "quantile_spread with the median "
        "cross-section split into n_groups buckets leaving < MIN_GROUP_ASSETS "
        "(5) assets per bucket; each bucket mean rests on a handful of names so "
        "the spread can be dominated by individual assets. Advisory only — "
        "reduce n_groups (the warning suggests a value) or treat the spread as a "
        "fragile small-cross-section diagnostic. Distinct from few_assets, which "
        "keys off the absolute cross-section size.",
        WarningCode.SPARSE_MAGNITUDE_WEIGHTED: "Sparse factor column is mixed-sign and not a "
        "clean ±1 ternary; statistic is magnitude-weighted (Sefcik-Thompson) "
        "rather than textbook MacKinlay signed CAAR — apply .sign() before "
        "calling for sign-flip semantics.",
        WarningCode.FEW_EVENTS: "CAAR significance test with MIN_EVENTS_HARD ≤ "
        "n_event_periods < MIN_EVENTS_WARN (4..29). caar is an equal-weight "
        "calendar-time portfolio across event periods, so this counts the "
        "number of periods with an event, not events; a sub-30 series is "
        "power-thin for the asymptotic t-distribution — read borderline "
        "p-values cautiously.",
        WarningCode.BORDERLINE_PORTFOLIO_PERIODS: "top_concentration with MIN_PORTFOLIO_PERIODS_HARD "
        "≤ n_periods < MIN_PORTFOLIO_PERIODS_WARN (3..19); one-sided t-test "
        "on the per-date diversification ratio is returned but df=n-1 inflates "
        "t_crit relative to the asymptotic cutoff.",
        WarningCode.FEW_DIRECTIONAL_PAIRS: "directional_hit_rate with MIN_DIRECTIONAL_PAIRS_HARD "
        "≤ n_pairs < MIN_DIRECTIONAL_PAIRS_WARN (10..29); the Pesaran-Timmermann "
        "hit rate is returned but n counts pooled non-overlapping (date, asset) "
        "directional trials, and the normal approximation to S_n is power-thin "
        "below ~30 pooled pairs — read borderline p-values cautiously. Below the "
        "HARD floor the metric short-circuits to NaN.",
        WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE: "Rectangular-kernel HAC variance-of-mean came out "
        "negative (no PSD guarantee, Andrews 1991); clamped to 0 → SE=0, t=0, p=1.0. "
        "Fires only on short / mildly anti-correlated samples.",
        WarningCode.BMP_RETURN_VOL_FALLBACK: "bmp_z ran without a price column: the "
        "estimation-window volatility falls back to the per-asset rolling std of "
        "forward_return, lagged by forward_periods so it ends before the event's "
        "forward window. This is a coarser, horizon-overlapping vol proxy than a "
        "daily-price std — supply price for the clean BMP standardiser.",
        WarningCode.UPSTREAM_UNAVAILABLE: "DAG-executor consumer skipped because an upstream "
        "producer short-circuited. The downstream MetricResult carries "
        "metadata['upstream'] / ['upstream_reason'] for the original cause.",
        WarningCode.METRIC_UNAVAILABLE: "Metric short-circuited on its own precondition (missing "
        "input column / config, or insufficient sample at its own floor); "
        "the NaN MetricResult's metadata['reason'] carries the specific cause. "
        "Distinct from UPSTREAM_UNAVAILABLE, which flags a dependency failure.",
        WarningCode.STRUCTURE_MISMATCH: "Metric's declared factor cell "
        "(scope / density / data structure) does not match the detected factor "
        "cell; under strict=False the metric short-circuits to NaN instead "
        "of executing.",
        WarningCode.LOW_CARDINALITY_DENSE_SIGNAL: "Dense factor has few distinct "
        "non-null values but no sparse event contract. Sparse event metrics "
        "require the {0, R} zero-value event contract and a "
        "sparse_ratio above the routing threshold; always-in-market states "
        "such as {-1, +1} stay dense and should use dense / directional metrics.",
        WarningCode.FREQUENT_EVENT_SIGNAL: "Sparse event metric explicitly ran "
        "on a factor with zero-valued rows but sparse_ratio below the "
        "automatic SPARSE routing threshold. The metric treats zeros as "
        "non-events; confirm that zero encodes the intended event contract. "
        "Events are frequent, so read event-study inference cautiously and "
        "inspect clustering / overlap diagnostics.",
        WarningCode.CROSS_FACTOR_DENSITY_MISMATCH: "Factor columns carry inconsistent FactorDensity (dense and sparse mixed).",
        WarningCode.CROSS_FACTOR_SCOPE_MISMATCH: "Factor columns carry inconsistent FactorScope (individual and common mixed).",
        WarningCode.SINGLE_ASSET_EVENT_DATA: "Single-asset event-shaped data (TIMESERIES + SPARSE, n_assets=1): "
        "event-axis metrics run over the event cross-section (n_events) and are "
        "usable on a single name. Metrics that need the asset cross-section — "
        "same-date event clustering (clustering_hhi) is degenerate at one "
        "event per date — need n_assets>=2 and are unavailable. Do not pool "
        "unrelated assets to clear this; that mixes return-generating processes.",
        WarningCode.EXCESSIVE_PERIOD_DROPS: "An upstream PANEL→SERIES primitive dropped more than "
        "DROP_RATE_WARN_THRESHOLD of dates at its cross-sectional filter; the "
        "metric was computed on a shortened sample. Exact counts are in "
        "MetricResult.metadata (n_periods_in / n_periods_out / dropped_periods / "
        "drop_rate / drop_reason).",
        WarningCode.EXCESSIVE_ASSET_DROPS: "An upstream primitive dropped more than "
        "DROP_RATE_WARN_THRESHOLD of assets at its per-asset filter (e.g. "
        "compute_common_betas dropping assets with insufficient history or zero "
        "factor variance); the cross-asset aggregate was computed on a shortened "
        "sample. Exact counts are in MetricResult.metadata (n_assets_in / "
        "n_assets_out / dropped_assets / drop_rate / drop_reason).",
        WarningCode.SLICE_BOUNDARY_TRUNCATION: "by_slice partitioned a panel on "
        "a date-axis column (one whose value varies within an asset over time, "
        "e.g. calendar year or regime label) while the metric's aggregation "
        "looks across dates (TS_ONLY / TS_THEN_CS / EVENT_TIME / "
        "RETURN_SPANNING). Each slice is evaluated on its own rows, so a "
        "rolling window / per-asset time-series regression / event window sees "
        "truncated history at the slice boundary — the per-slice value differs "
        "from the full-sample value decomposed by period. Per-date metrics "
        "(CS_THEN_TS / CS_SNAPSHOT) and cross-sectional partitions (constant "
        "within an asset, e.g. sector) are unaffected and do not trigger.",
    }
)


def cross_section_tier(n_assets: int) -> WarningCode | None:
    """Map an inference-stage cross-asset ``n_assets`` to the appropriate warning code.

    The argument is the **inference-stage** ``n_assets`` — the count of assets
    actually entering the cross-asset test, not the panel-union
    ``n_assets`` surface field. For ``(COMMON, *, None,
    PANEL)`` cells the two differ: ``compute_common_betas`` drops assets
    with fewer than ``MIN_COMMON_BETA_PERIODS_HARD`` non-null observations, so the union
    can be materially larger than the post-filter count that drives
    ``primary_p``'s ``dof = n_assets - 1``. Callers (``suggest_config``,
    ``_compute_common_panel``) therefore pre-filter before calling.

    A single :attr:`WarningCode.FEW_ASSETS` flags the whole thin regime
    (``2 ≤ n_assets < MIN_ASSETS_WARN``); how severe it is scales with
    ``n_assets``, which callers carry in metadata rather than encoding into
    separate tier members. Returns ``None`` at ``n_assets ≥ MIN_ASSETS_WARN``
    (clean) or ``n_assets < 2`` (PANEL impossible by upstream structure
    routing; defensive).
    """
    from factrix._stats.constants import MIN_ASSETS_WARN

    if 2 <= n_assets < MIN_ASSETS_WARN:
        return WarningCode.FEW_ASSETS
    return None
