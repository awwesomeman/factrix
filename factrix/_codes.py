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
    # Fired when ADF p exceeds the configured threshold on a DENSE factor
    # (Stambaugh-style persistent-regressor flag, section 5.2 / 7.3).
    # Not raised for SPARSE.
    PERSISTENT_REGRESSOR = "persistent_regressor"
    # Fired by the series-mean inference members and ``fm_beta`` when the
    # tested per-period series — the series the mean test runs on (IC series,
    # per-period betas, spread series), not the raw factor / return columns —
    # has lag-1 autocorrelation above
    # ``PERSISTENT_SERIES_AUTOCORR`` (0.3). In that regime no path in the
    # library is calibrated — Newey-West, the stationary bootstrap and the
    # plain t all over-reject, by 2–4x nominal at phi = 0.6 and worse above —
    # so the practitioner response is a raised hurdle (Harvey-Liu-Zhu 2016:
    # t > 3) or a longer sample, not a different inference member.
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
    # cross-section size): a wide panel cut into
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
    # Fired by ``caar`` (significance test) and ``corrado_rank``, both of
    # which test a per-event-period series, when its length sits in
    # ``[MIN_EVENTS_HARD, MIN_EVENTS_WARN)`` — the statistic is returned
    # but the Brown-Warner (1985) convention treats sub-30 event-period
    # counts as power-thin for the asymptotic reference distribution.
    # Below the HARD floor the primitive short-circuits to NaN instead.
    # ``bmp_z`` pools events but fires it on the same axis once events
    # share periods: the effective sample is then the distinct event
    # periods, not the event count.
    # Naming follows the ``<axis>_<condition>`` grammar; the Brown-Warner
    # method reference lives in this gloss rather than the member name.
    FEW_EVENTS = "few_events"
    # Fired by ``top_concentration`` when the per-period ratio series sits
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
    # Fired by ``directional_pair_accuracy`` when the pooled non-overlapping
    # within-period ordering pairs sit in
    # ``[MIN_PAIR_ACCURACY_PAIRS_HARD, MIN_PAIR_ACCURACY_PAIRS_WARN)``. The
    # metric is descriptive and returns no p-value, but a small comparable-pair
    # count makes the per-period ordering accuracy fragile. Below the HARD floor
    # it short-circuits to NaN instead. Named on the ``pairs`` axis token: the
    # count is comparable asset pairs, not periods or assets.
    FEW_ORDERING_PAIRS = "few_ordering_pairs"
    # Fired when a rectangular-kernel HAC primitive (Hansen-Hodrick 1980)
    # produces a negative variance-of-mean estimate on short / mildly
    # anti-correlated samples. Unlike the Bartlett kernel, the rectangular
    # kernel carries no PSD guarantee (Andrews 1991 §3); the primitive
    # clamps variance to 0.0, which leaves no SE to divide by, so the t-test
    # is not computable and DEGENERATE_VARIANCE is raised alongside this code.
    # (It formerly returned t=0, p=1.0 — "conservative", but it read an
    # estimator breakdown as a non-rejection.)
    RECT_KERNEL_NEGATIVE_VARIANCE = "rect_kernel_negative_variance"
    # Fired when the resolved Bartlett bandwidth L is large relative to the
    # period count T (``n_periods < 5 * L``), so each autocovariance in the
    # kernel sum is estimated from few products and the long-run variance is
    # dominated by estimation noise. Reached mainly when a long overlap
    # horizon meets a short sample (h=21 at T=60 resolves L=20). This was a
    # ``logger.warning`` only; per the project's method-switch-warning norm a
    # regime this consequential belongs on the MetricResult.
    HAC_BANDWIDTH_ILL_CONDITIONED = "hac_bandwidth_ill_conditioned"
    # Fired by a series-mean inference member when the sample admits no
    # t-statistic: every observation identical (zero dispersion), or a HAC
    # SE that collapses to zero. An identical-and-non-zero sample is
    # degenerate in the *maximum*-evidence direction, so the former
    # ``t=0, p=1`` inverted the reading. On an ``InferenceResult`` the stat
    # and p are NaN; a metric carrying this code keeps its ``value`` but
    # reports ``stat=None`` / ``p_value=None`` (see
    # ``factrix.metrics._helpers._degenerate_test_fields``).
    DEGENERATE_VARIANCE = "degenerate_variance"

    # Fired by ``bmp_z`` when no ``price`` column is present and the
    # estimation-window volatility falls back to the per-asset rolling std of
    # ``forward_return``. Because forward_return[t] looks ahead to [t+1, t+1+h],
    # the fallback std is lagged by ``overlap_periods`` so the estimation window
    # ends before the event's own forward window — but it is still a coarser,
    # horizon-overlapping volatility proxy than a price-derived one-period std.
    # The z-test is
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
    # clustering_hhi, whose same-period event clustering is degenerate at
    # ≤1 event/date) stays unusable. The warning names those so their absence
    # from `usable` is explained, not silent. Deliberately does NOT advise
    # adding assets — pooling unrelated names mixes return-generating processes.
    SINGLE_ASSET_EVENT_DATA = "single_asset_event_data"

    # Per-axis silent-drop flags. A metric whose upstream primitive silently
    # dropped a large share of its sample at a filter raises the code for the
    # dropped axis: PERIOD_DROPS for the time axis (e.g. compute_ic dropping
    # dates with n_assets below the per-period floor), ASSET_DROPS for the
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
    # or regime label) and the metric declares
    # MetricSpec.slice_boundary_sensitive (a capability of the estimator —
    # not inferred from its aggregation category). by_slice evaluates each
    # slice as an independent dataset, so a rolling window, per-asset
    # time-series regression, or event window sees truncated history at the
    # slice boundary — the per-slice value differs from the full-sample
    # value decomposed by period. Metrics that don't declare the flag
    # (including some EVENT_TIME metrics whose window is self-contained,
    # e.g. event_hit_rate) are unaffected and do not trigger. Cross-sectional
    # partitions (sector, size bucket — constant within an asset) keep each
    # asset's history intact and do not trigger.
    SLICE_BOUNDARY_TRUNCATION = "slice_boundary_truncation"
    # Fired by ``top_concentration`` under ``weight_by="abs_factor"`` when
    # the factor never changes sign across the panel. |factor| is a
    # density weight only if zero is the factor's neutral point: the HHI
    # of |f| is not location-invariant, so a raw (uncentred) factor's
    # concentration reading moves with an arbitrary shift — on one bucket,
    # eff_n 9.7 (z-scored) vs 7.8 (shifted by −1) vs 10.0 (shifted by −10).
    # Advisory: the metric still runs; centre the factor (e.g. z-score
    # cross-sectionally) or use ``alpha_contribution``.
    ONE_SIGNED_FACTOR = "one_signed_factor"
    # Fired when a pooled statistic detects that its units are not independent
    # draws and deflates itself for the clustering: same-period events sharing
    # a market shock (event_hit_rate, event_ic) or cross-correlated per-asset
    # betas (common_beta). The estimator is unchanged; only the standard error
    # / p-value moves, and the code is the record that it did, since the
    # deflation is data-driven rather than a fixed configuration.
    EVENT_CLUSTERING_ADJUSTED = "event_clustering_adjusted"
    # Fired by every mean-adjusted event test (caar / bmp_z / corrado_rank /
    # event_hit_rate / event_ic / event_skewness) when the tested events'
    # estimation windows are mostly other events' realised returns —
    # ``metadata["estimation_window_event_share"]`` above
    # ``ESTIMATION_WINDOW_EVENT_SHARE_WARN``. The statistic is returned
    # unchanged; the code says the null it is read against is conservative.
    ESTIMATION_WINDOW_CONTAMINATED = "estimation_window_contaminated"

    # Preprocess scale-regime flags. ``factrix.preprocess.normalize`` returns a
    # bare DataFrame — there is no MetricResult to hang ``warning_codes`` on —
    # so these travel in the text of a ``UserWarning``. They exist because a
    # sample-regime-driven switch of estimator must never be silent.
    #
    # The per-date MAD collapsed to zero (>50% ties at the median) and the
    # non-robust per-date sample standard deviation stood in as the scale.
    # Robust and non-robust dates then sit in one output column with nothing
    # downstream able to tell them apart.
    ZERO_MAD_STD_FALLBACK = "zero_mad_std_fallback"
    # ``mad_winsorize`` left a date unwinsorized because the factor there is a
    # sparse ``{0, R}`` trigger column (median 0, MAD 0). Its standard
    # deviation is produced by the triggers themselves, so it shrinks with the
    # trigger rate and the clip would destroy event magnitudes — 58% of a unit
    # event at one trigger in fifty names.
    SPARSE_WINSORIZE_SKIPPED = "sparse_winsorize_skipped"
    # A date carries fewer than MIN_SCALE_ASSETS_HARD finite factor values, so
    # no robust per-date scale exists; the date is left unscaled (z-score null,
    # clip skipped) rather than fabricating one.
    INSUFFICIENT_SCALE_ASSETS = "insufficient_scale_assets"
    # NaN / +-Inf inputs were blanked to null. A non-finite tick is a data
    # error, not an extreme value: clipping it into a winsorization band
    # manufactures a plausible finite number that survives every downstream
    # drop_nulls().drop_nans().
    NON_FINITE_INPUT_DROPPED = "non_finite_input_dropped"

    # Fired by ``orthogonalize_factor`` when a per-date cross-section clears the
    # computability floor but leaves fewer than MIN_ORTHOGONALIZE_RESIDUAL_ASSETS
    # residual degrees of freedom. Those dates are skipped rather than fitted:
    # raw R2 is mechanically ~K/(N-1) even at a true R2 of 0, so a 6-name
    # cross-section on 4 regressors reported R2 = 0.79 while removing 83% of
    # the factor's variance.
    INSUFFICIENT_REGRESSION_DF = "insufficient_regression_df"
    # Fired by ``orthogonalize_factor`` when the per-date design matrix is rank
    # deficient (the classic full-dummy-set-plus-intercept trap). ``lstsq`` does
    # not raise there — it returns the minimum-norm solution — so residuals stay
    # correct but the reported betas are an arbitrary point in the solution
    # space. ``mean_betas`` is suppressed rather than reported.
    RANK_DEFICIENT_DESIGN = "rank_deficient_design"

    # Fired by ``compute_forward_return`` when the per-asset date grids are
    # ragged (an asset is missing periods other assets have). The horizon is a
    # row shift within an asset, so on a ragged grid an h-period-ahead return
    # spans a different number of real periods for different assets.
    RAGGED_PERIOD_GRID = "ragged_period_grid"

    @property
    def description(self) -> str:
        return _WARNING_DESCRIPTIONS[self]


_WARNING_DESCRIPTIONS: dict[WarningCode, str] = {}


_WARNING_DESCRIPTIONS.update(
    {
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS: "n_periods is below the WARN floor (~30); NW HAC SE may be biased. "
        "Reused across panel time-series guards (MIN_PERIODS_WARN) and "
        "primitive inference (MIN_FM_PERIODS_WARN); both default to 30.",
        WarningCode.EVENT_WINDOW_OVERLAP: "Two events on one asset sat fewer "
        "than overlap_periods apart, so their forward-return windows "
        "(t, t+h] overlapped and they are not independent draws. Every event "
        "significance test (caar / bmp_z / corrado_rank / event_hit_rate / "
        "event_ic / event_skewness) strides its event axis per asset before "
        "testing and fires this once, with the counts in "
        "metadata['n_events_overlapping'] / ['n_events_sampled']. The "
        "statistic is the calibrated one — it runs on the surviving "
        "non-overlapping events — so read the code as the cost in sample of a "
        "trigger that fires in bursts, not as a defect. It cannot fire at "
        "overlap_periods = 1 (consecutive events are already independent).",
        WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED: "The resolved Bartlett "
        "bandwidth L exceeds n_periods / 5, so the HAC long-run variance is "
        "estimated from too few lag products to be stable. Usually a long "
        "overlap_periods against a short sample. The t-test still runs with "
        "the effective-df correction, but treat the p-value as indicative "
        "only and lengthen the sample or shorten the horizon. In this regime "
        "the effective-df cap at n_periods / overlap_periods binds hard, so if "
        "the series carries less dependence than overlap_periods implies the "
        "test is conservative rather than oversized (measured 0.2-3.5% against "
        "a nominal 5% on an iid series at h=21).",
        WarningCode.PERSISTENT_REGRESSOR: "The predictive regressor is in a regime the corrected test is less well sized in: ADF p exceeds the configured threshold (unit-root suspect), or the measured Stambaugh channel |rho_hat * phi_corrected| exceeds 0.7, or the bias-corrected AR(1) coefficient came out at or above one. The Stambaugh (1999) bias itself is CORRECTED via the Amihud-Hurvich (2004) augmented regression, so this is not 'beta may be biased' - at overlap_periods=1 it is 'the strongest Stambaugh cells leave the corrected test at 6-8% against a nominal 5%, where the calibrated cells sit at 4-6%'. Note this code is about the regressor, NOT about overlap: at overlap_periods>1 the test is 7.5-14.5% oversized for every phi INCLUDING rho=0, which is the overlapping-regression HAC problem and fires no code of its own. Read the p against a raised hurdle.",
        WarningCode.SERIAL_CORRELATION_DETECTED: "The tested per-period series has "
        "lag-1 autocorrelation above PERSISTENT_SERIES_AUTOCORR (0.3). No HAC or "
        "bootstrap path is calibrated here — measured 13–17% (NW), 12–19% "
        "(bootstrap) and 32–34% (plain t) at a nominal 5% for phi=0.6, worse "
        "above — so read the p-value against a raised hurdle (Harvey-Liu-Zhu "
        "2016: t > 3) or lengthen the sample; switching inference member does "
        "not fix it.",
        WarningCode.FEW_ASSETS: "Cross-section asset count is below the "
        "relevant WARN floor (panel-wide MIN_ASSETS_WARN=30, per-period "
        "MIN_IC_ASSETS_WARN=10, or per-period MIN_FM_ASSETS_WARN=10). The "
        "statistic is returned, but small n_assets inflates critical values or "
        "leaves minimal residual degrees of freedom. Severity scales with "
        "n_assets; read the relevant n_assets metadata. A by-design few-asset "
        "study declares the regime once via "
        "evaluate(..., expected_warnings=('few_assets',)): the record is kept "
        "and marked expected=True, and the per-run UserWarning echo stops.",
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
        WarningCode.FEW_EVENTS: "An event significance test (caar / "
        "corrado_rank / bmp_z / event_hit_rate / event_ic / event_skewness) "
        "with a raw event count below MIN_EVENTS_WARN (30) x overlap_periods. "
        "The floor is scaled because every one of these tests first strides "
        "its event axis at the forward-return horizon — keeping at most one "
        "event in h per asset — so a raw series must carry h x 30 events to land on 30 "
        "independent ones. The message states the scaled floor, the raw count "
        "and the count that survived sampling; caar and corrado_rank count "
        "event *periods* on that axis (caar an equal-weight calendar-time "
        "portfolio, corrado_rank the per-period mean signed rank), the others "
        "count events. bmp_z fires on a second trigger as well: once events "
        "share periods its effective sample is the distinct event periods, "
        "not the event count (the Kolari-Pynnönen adjustment cannot "
        "manufacture independent periods; measured ~10% size at 8 periods, "
        "~7% at 15, clearing by ~30, nominal 5%). A sub-30 effective sample is "
        "power-thin for the asymptotic distribution — read borderline "
        "p-values cautiously.",
        WarningCode.BORDERLINE_PORTFOLIO_PERIODS: "top_concentration with MIN_PORTFOLIO_PERIODS_HARD "
        "≤ n_periods < MIN_PORTFOLIO_PERIODS_WARN (3..19); the one-sided t-test "
        "on the per-period diversification ratio is returned but df=n-1 inflates "
        "t_crit, and at the bottom of the range it is extremely conservative: "
        "at exactly 3 periods it rejected 0 of 250 null draws at a nominal 5%, "
        "so a p-value there carries essentially no information. Treat value "
        "as descriptive until the series is well inside the range.",
        WarningCode.FEW_DIRECTIONAL_PAIRS: "directional_hit_rate with MIN_DIRECTIONAL_PAIRS_HARD "
        "≤ n_pairs < MIN_DIRECTIONAL_PAIRS_WARN (10..29); the Pesaran-Timmermann "
        "hit rate is returned but n counts pooled non-overlapping (date, asset) "
        "directional trials, and the normal approximation to S_n is power-thin "
        "below ~30 pooled pairs — read borderline p-values cautiously. Below the "
        "HARD floor the metric short-circuits to NaN.",
        WarningCode.FEW_ORDERING_PAIRS: "directional_pair_accuracy with "
        "MIN_PAIR_ACCURACY_PAIRS_HARD ≤ n_pairs < "
        "MIN_PAIR_ACCURACY_PAIRS_WARN (10..29); the descriptive ordering "
        "accuracy is returned but n counts pooled non-overlapping within-period "
        "asset pairs after factor/return ties are removed. Below the HARD "
        "floor the metric short-circuits to NaN.",
        WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE: "Rectangular-kernel HAC variance-of-mean came out "
        "negative (no PSD guarantee, Andrews 1991); clamped to 0 → SE=0, so the t-test "
        "is not computable and returns NaN (also flagged degenerate_variance). "
        "Fires only on short / mildly anti-correlated samples.",
        WarningCode.DEGENERATE_VARIANCE: "The sample admits no test statistic: every "
        "observation is identical (zero dispersion), the HAC SE collapsed to zero, or "
        "a Wald restriction's covariance is singular. "
        "The metric keeps its value and reports stat=None / p_value=None (NaN on a "
        "raw InferenceResult) — an identical, non-zero sample is degenerate in the "
        "maximum-evidence direction, not evidence of a null, so t=0 / p=1 would "
        "invert the reading.",
        WarningCode.BMP_RETURN_VOL_FALLBACK: "bmp_z ran without a price column: the "
        "estimation-window volatility falls back to the per-asset rolling std of "
        "forward_return, lagged by overlap_periods so it ends before the event's "
        "forward window. This is a coarser, horizon-overlapping vol proxy than a "
        "price-derived one-period std — supply price for the clean BMP "
        "standardiser.",
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
        "same-period event clustering (clustering_hhi) is degenerate at one "
        "event per period — need n_assets>=2 and are unavailable. Do not pool "
        "unrelated assets to clear this; that mixes return-generating processes.",
        WarningCode.EXCESSIVE_PERIOD_DROPS: "An upstream PANEL→SERIES primitive dropped more than "
        "DROP_RATE_WARN_THRESHOLD of periods at its cross-sectional filter; the "
        "metric was computed on a shortened sample. Exact counts are in "
        "MetricResult.metadata (n_periods_in / n_periods_out / dropped_periods / "
        "drop_rate / drop_reason).",
        WarningCode.EXCESSIVE_ASSET_DROPS: "An upstream primitive dropped more than "
        "DROP_RATE_WARN_THRESHOLD of assets at its per-asset filter (e.g. "
        "compute_common_betas dropping assets with insufficient history or zero "
        "factor variance); the cross-asset aggregate was computed on a shortened "
        "sample. Exact counts are in MetricResult.metadata (n_assets_in / "
        "n_assets_out / dropped_assets / drop_rate / drop_reason).",
        WarningCode.EVENT_CLUSTERING_ADJUSTED: "A pooled event statistic "
        "found its units correlated and deflated itself by the Kish design "
        "effect 1/sqrt(1 + (n_eff - 1) * r_hat) — the same Kolari-Pynnonen "
        "(2010) machinery bmp_z and directional_hit_rate use. event_hit_rate "
        "and event_ic key it on the within-period intraclass correlation of "
        "their own per-event score (events sharing a period share that "
        "period's shock, so they are not separate trials). It fires only when "
        "the deflation is material — r_hat > 0 and a scale below "
        "KP_MATERIAL_SCALE (0.95); above that the statistic is left alone and "
        "event_hit_rate keeps the exact binomial. The point estimate is "
        "untouched; the p-value widens. Measured on a true null: "
        "event_hit_rate 63.5% -> nominal at 20 assets sharing 40 event dates. "
        "metadata['kolari_pynnonen_r'] / ['kolari_pynnonen_scaling'] disclose "
        "the estimate and the deflator.",
        WarningCode.ESTIMATION_WINDOW_CONTAMINATED: "A mean-adjusted event "
        "test (caar / bmp_z / corrado_rank / event_hit_rate / event_ic / "
        "event_skewness) found that, averaged over the tested events, more "
        "than ESTIMATION_WINDOW_EVENT_SHARE_WARN (25%) of each event's "
        "estimation-window periods lie inside another event's forward-return "
        "window on the same asset (metadata['estimation_window_event_share']). "
        "The window then estimates the neighbours' realised event returns "
        "rather than the asset's unconditional mean, the per-event abnormal "
        "returns are negatively correlated through the shared periods, and "
        "the cross-event variance overstates the variance of their mean: the "
        "test is conservative, not liberal (measured on an iid null at "
        "h = 21: bmp_z 0.3% size at nominal 5%). A dense trigger or a long "
        "horizon is the cause. A supplied market-adjusted 'abnormal_return' "
        "does not remove it (measured 0.3% -> 0.3%) and removes the effect "
        "itself when every name fires on the same periods, so the statistic "
        "is returned unchanged: read the p-value as an upper bound, or "
        "shorten the horizon / thin the trigger.",
        WarningCode.ONE_SIGNED_FACTOR: "top_concentration ran with "
        "weight_by='abs_factor' on a factor that never changes sign across the "
        "panel. |factor| is a density weight only when zero is the factor's "
        "neutral point; the HHI of |f| is not location-invariant, so an "
        "uncentred factor's concentration reading moves with an arbitrary "
        "shift. Centre the factor (cross-sectional z-score) or use "
        "alpha_contribution.",
        WarningCode.SLICE_BOUNDARY_TRUNCATION: "by_slice partitioned a panel on "
        "a date-axis column (one whose value varies within an asset over time, "
        "e.g. calendar year or regime label) while the metric declares "
        "MetricSpec.slice_boundary_sensitive (a capability of the estimator, "
        "not inferred from its aggregation category). Each slice is evaluated "
        "on its own rows, so a rolling window / per-asset time-series "
        "regression / event window sees truncated history at the slice "
        "boundary — the per-slice value differs from the full-sample value "
        "decomposed by period. Metrics that don't declare the flag and "
        "cross-sectional partitions (constant within an asset, e.g. sector) "
        "are unaffected and do not trigger.",
        WarningCode.ZERO_MAD_STD_FALLBACK: "A preprocess scale estimator fell "
        "back from the robust MAD to the non-robust per-date sample standard "
        "deviation because >50% of the cross-section ties at the median "
        "(bucketed / binary factors are the common case). The output column "
        "then mixes robust and non-robust dates; the fallback keeps the factor "
        "finite and rank-preserving but the scale is no longer outlier-proof.",
        WarningCode.SPARSE_WINSORIZE_SKIPPED: "mad_winsorize left a date "
        "unwinsorized: the factor is a sparse {0, R} trigger column whose "
        "median and MAD are both 0, so the standard-deviation fallback is "
        "driven by the triggers themselves and collapses as the trigger rate "
        "falls (at 1 trigger in 50 names a 3-std band clipped a unit event to "
        "0.42). Skipping preserves the event magnitudes the downstream "
        "sparse-factor metrics measure.",
        WarningCode.INSUFFICIENT_SCALE_ASSETS: "A date carries fewer than "
        "MIN_SCALE_ASSETS_HARD (3) finite factor values, so no robust per-date "
        "scale exists. cross_sectional_zscore returns null there and "
        "mad_winsorize skips the clip, rather than fabricating a score (n=1 "
        "used to yield 0.0, n=2 a constant +-0.6745 regardless of the values).",
        WarningCode.NON_FINITE_INPUT_DROPPED: "NaN / +-Inf input values were "
        "blanked to null. They are excluded from every per-date statistic and "
        "from the output: a non-finite tick is a data error, not an extreme "
        "value, and winsorizing it into the band would manufacture a plausible "
        "finite number that survives every downstream "
        "drop_nulls().drop_nans().",
        WarningCode.INSUFFICIENT_REGRESSION_DF: "orthogonalize_factor skipped a "
        "date whose cross-section left fewer than "
        "MIN_ORTHOGONALIZE_RESIDUAL_ASSETS residual degrees of freedom "
        "(n_assets - n_base - 1). Raw R2 is mechanically ~K/(N-1) even at a "
        "true R2 of 0, so fitting there reports noise as explanatory power "
        "while removing most of the factor's variance. Skipped dates keep "
        "their original values and are counted in n_dates_skipped.",
        WarningCode.RANK_DEFICIENT_DESIGN: "orthogonalize_factor met a "
        "rank-deficient per-date design matrix — most often a full industry "
        "dummy set alongside the always-prepended intercept. np.linalg.lstsq "
        "does not raise there; it returns the minimum-norm solution, so the "
        "residual (a unique projection) stays correct but the betas are an "
        "arbitrary point in the solution space. mean_betas is suppressed. Drop "
        "one dummy category as the reference level.",
        WarningCode.RAGGED_PERIOD_GRID: "compute_forward_return saw per-asset "
        "date grids that do not agree: at least one asset is missing periods "
        "that others have. The horizon is a shift along each asset's own "
        "period index, so an h-period-ahead return then spans a different "
        "number of panel periods for different assets. Reindex the panel onto "
        "a common grid if the horizons must be comparable across names.",
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
