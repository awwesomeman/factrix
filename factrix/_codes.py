"""v0.5 enum codes for warnings, info notes, and cell stats.

``WarningCode`` / ``InfoCode`` / ``StatCode`` follow the ``*Code`` suffix
invariant (§7.5).
"""

from __future__ import annotations

from enum import StrEnum


class WarningCode(StrEnum):
    """Procedure-degradation flags (replaces v3 ``DegradedMode``).

    Each value carries a one-line ``description`` gloss for
    ``profile.diagnose()`` consumers (review fix UX-4) — pure metadata,
    StrEnum value identity is unchanged.
    """

    UNRELIABLE_SE_SHORT_PERIODS = "unreliable_se_short_periods"
    EVENT_WINDOW_OVERLAP = "event_window_overlap"
    # Fired when ADF p > 0.1 on a DENSE factor (Stambaugh-style
    # persistent-regressor flag, §5.2 / §7.3). Not raised for SPARSE.
    PERSISTENT_REGRESSOR = "persistent_regressor"
    SERIAL_CORRELATION_DETECTED = "serial_correlation_detected"
    # Two-tier cross-asset N guards for PANEL common_continuous. Mirrors
    # the n_periods two-tier (UNRELIABLE_SE_SHORT_PERIODS) but the axis
    # never raises — cross-asset t-test on E[β] is well-defined for N≥2.
    SMALL_CROSS_SECTION_N = "small_cross_section_n"
    BORDERLINE_CROSS_SECTION_N = "borderline_cross_section_n"
    # Fired by the (COMMON, SPARSE, PANEL) procedure when the broadcast
    # dummy carries MIN_BROADCAST_EVENTS_HARD ≤ n_events <
    # MIN_BROADCAST_EVENTS_WARN. Per-asset β is identifiable but
    # the cross-event averaging is too thin for asymptotic t to be
    # trusted. Below the HARD floor raises InsufficientSampleError instead.
    SPARSE_COMMON_FEW_EVENTS = "sparse_common_few_events"
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
    # Fired when a rectangular-kernel HAC primitive (Hansen-Hodrick 1980)
    # produces a negative variance-of-mean estimate on short / mildly
    # anti-correlated samples. Unlike the Bartlett kernel, the rectangular
    # kernel carries no PSD guarantee (Andrews 1991 §3); the primitive
    # clamps variance to 0.0 and the t-test returns t=0, p=1.0 (cannot
    # reject), the conservative direction.
    RECT_KERNEL_NEGATIVE_VARIANCE = "rect_kernel_negative_variance"

    # Long-run covariance Ŝ of a moment-condition system was numerically
    # singular, so the J-statistic was computed using a Moore-Penrose
    # pseudo-inverse rather than a true inverse. Fires on rank-deficient
    # or strongly collinear moment matrices.
    SINGULAR_WEIGHT_MATRIX = "singular_weight_matrix"

    # Fired by the DAG executor when an upstream producer short-circuited
    # (returned a NaN MetricResult with metadata["reason"]) and the
    # consumer is skipped. The downstream MetricResult carries
    # metadata["upstream"] / ["upstream_reason"] so the original cause
    # is recoverable without re-walking the dependency graph.
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"

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
        WarningCode.SMALL_CROSS_SECTION_N: "PANEL cross-asset t-test with n_assets < MIN_ASSETS (10); "
        "df=n_assets-1 too low — t_crit at n_assets=3 ≈ 4.30 "
        "(+119% vs asymptotic 1.96).",
        WarningCode.BORDERLINE_CROSS_SECTION_N: "PANEL cross-asset t-test with MIN_ASSETS ≤ n_assets < "
        "MIN_ASSETS_WARN (10..29); residual t_crit inflation "
        "5–15% — read borderline p-values cautiously.",
        WarningCode.SPARSE_COMMON_FEW_EVENTS: "(COMMON, SPARSE, PANEL) broadcast dummy has "
        "MIN_BROADCAST_EVENTS_HARD ≤ n_events < MIN_BROADCAST_EVENTS_WARN "
        "(5..19); per-asset β estimable but cross-event averaging too thin "
        "for asymptotic t.",
        WarningCode.SPARSE_MAGNITUDE_WEIGHTED: "Sparse factor column is mixed-sign and not a "
        "clean ±1 ternary; statistic is magnitude-weighted (Sefcik-Thompson) "
        "rather than textbook MacKinlay signed CAAR — apply .sign() before "
        "calling for sign-flip semantics.",
        WarningCode.FEW_EVENTS: "CAAR significance test with MIN_EVENTS_HARD ≤ "
        "n_event_dates < MIN_EVENTS_WARN (4..29); t-stat returned but "
        "Brown-Warner (1985) convention treats sub-30 events as power-thin "
        "for the asymptotic t-distribution — read borderline p-values cautiously.",
        WarningCode.BORDERLINE_PORTFOLIO_PERIODS: "top_concentration with MIN_PORTFOLIO_PERIODS_HARD "
        "≤ n_periods < MIN_PORTFOLIO_PERIODS_WARN (3..19); one-sided t-test "
        "on the per-date diversification ratio is returned but df=n-1 inflates "
        "t_crit relative to the asymptotic cutoff.",
        WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE: "Rectangular-kernel HAC variance-of-mean came out "
        "negative (no PSD guarantee, Andrews 1991); clamped to 0 → SE=0, t=0, p=1.0. "
        "Fires only on short / mildly anti-correlated samples.",
        WarningCode.SINGULAR_WEIGHT_MATRIX: "GMM long-run covariance Ŝ was numerically "
        "singular; J-statistic was computed via Moore-Penrose pseudo-inverse rather than a "
        "true inverse. Fires on rank-deficient or strongly collinear moment matrices.",
        WarningCode.UPSTREAM_UNAVAILABLE: "DAG-executor consumer skipped because an upstream "
        "producer short-circuited. The downstream MetricResult carries "
        "metadata['upstream'] / ['upstream_reason'] for the original cause.",
    }
)


def cross_section_tier(n_assets: int) -> WarningCode | None:
    """Map an inference-stage cross-asset N to the appropriate warning code.

    The argument is the **inference-stage** N — the count of assets
    actually entering the cross-asset test, not the panel-union
    ``FactorProfile.n_assets`` surface field. For ``(COMMON, *, None,
    PANEL)`` cells the two differ: ``compute_ts_betas`` drops assets
    with fewer than ``MIN_TS_OBS`` non-null observations, so the union
    can be materially larger than the post-filter count that drives
    ``primary_p``'s ``dof = N - 1``. Callers (``suggest_config``,
    ``_compute_common_panel``) therefore pre-filter before calling.

    Tiers are mutually exclusive — SMALL is strictly more severe than
    BORDERLINE — so callers can membership-check the more severe code
    without an else branch. Returns ``None`` at ``n_assets ≥
    MIN_ASSETS_WARN`` (clean) or ``n_assets < 2`` (PANEL impossible
    by upstream structure routing; defensive).
    """
    from factrix._stats.constants import MIN_ASSETS, MIN_ASSETS_WARN

    if 2 <= n_assets < MIN_ASSETS:
        return WarningCode.SMALL_CROSS_SECTION_N
    if MIN_ASSETS <= n_assets < MIN_ASSETS_WARN:
        return WarningCode.BORDERLINE_CROSS_SECTION_N
    return None


class InfoCode(StrEnum):
    """Neutral facts surfaced to the caller — not warnings, not errors.

    Empty after the procedure-based dispatcher retired in #448 — the
    only member, ``SCOPE_AXIS_COLLAPSED``, tracked a legacy routing
    collapse that no longer happens under the DAG executor. The enum
    stays exported as the home for future neutral notes.
    """

    @property
    def description(self) -> str:
        return _INFO_DESCRIPTIONS[self]


_INFO_DESCRIPTIONS: dict[InfoCode, str] = {}


class StatCode(StrEnum):
    """Cell-specific scalar stats keyed in ``FactorProfile.stats``.

    Naming grammar (#187): each code is ``<TARGET>_<KIND>``.

    - ``TARGET`` — what is being measured. **Primary** (cell main
      effect) carries no prefix because ``FactorProfile.config``
      (``scope`` / ``density`` / ``metric``) is the single source of
      truth for cell identity. **Diagnostic** carries an explicit
      prefix (``FACTOR_`` / ``RESID_`` / ``EVENT_``) because the
      target lives outside ``config``.
    - ``KIND`` — what kind of number. ``_MEAN`` / ``_VALUE`` for point
      estimates, statistic-named suffixes (``_T_NW`` / ``_TAU`` /
      ``_Q``) for test statistics, ``_P_<algo>`` for p-values
      (``_P_NW`` / ``_P_HH`` / ``_P_GMM``). Diagnostic p-values keep
      their ``<TARGET>_<TEST>_P`` shape (``FACTOR_ADF_P`` /
      ``RESID_LJUNG_BOX_P``) — the asymmetry is structural, see below.

    **Inference primary stats — algorithm-suffixed pair shape**

    Each inference algorithm emits a (test statistic, p-value) pair with
    KINDs that abbreviate the test statistic's reference distribution:
    ``T`` (Student-t / asymptotic normal), ``J`` (Hansen J / χ²),
    ``WALD`` (Wald χ²), ``F`` (Snedecor F), ``LR`` (likelihood ratio).
    Currently shipping: ``(T_NW, P_NW)`` for Newey-West,
    ``(T_HH, P_HH)`` for Hansen-Hodrick, ``(J_GMM, P_GMM)`` for
    [Hansen (1982)][hansen-1982] generalized method of moments (GMM) J-test
    (over-identification is χ², not t, so GMM emits J rather than T),
    ``(WALD_NWCL, P_WALD_NWCL)`` for Newey-West (NW)
    heteroskedasticity-and-autocorrelation-consistent (HAC) + one-way
    cluster on the slice grouping, and ``(WALD_TWOWAY,
    P_WALD_TWOWAY)`` for two-way cluster on (date, asset) (slice-test
    functions, #153 / #176). The Wald pairs follow the same
    ``<KIND>_<ALGO>`` shape — KIND = ``WALD`` (χ² statistic name,
    parallel to ``T``), ALGO names the cluster-SE family
    (parallel to ``NW`` / ``HH`` naming the kernel family). ``P_BOOT``
    ships alongside as the singleton emitted by ``BlockBootstrap``:
    empirical p-values have no parametric test statistic to publish,
    and ``BlockBootstrap`` is a single Estimator class (fixed vs
    stationary scheme is a ctor arg living in metadata).

    **Why primary p-value is ``P_<algo>`` while diagnostic p-value is
    ``<target>_<test>_P``**: primary p has a single conceptual target
    (the cell's primary estimate, identified by ``profile.config``) so
    the prefix slot carries the algorithm choice. Diagnostic p has
    multiple non-primary targets (factor input / residual / event
    distribution) so the prefix slot carries the target axis and the
    test name floats with KIND. Both grammars co-exist deliberately.

    **Redesign trigger** — when (a) ≥ 4 inference algorithms ship
    concurrently or (b) ≥ 3 distinct test-statistic KINDs (T / J /
    Wald / F / LR) coexist, the flat ``<KIND>_<ALGO>`` enum becomes
    a (kind × algo) cardinality product and a structured shape
    (``profile.inference[Algo.X] = {test_stat, kind, p, df}``) earns
    its breaking-change cost. Below those thresholds the flat
    enum stays cheaper. **As of #191 the algorithm count is 6
    (NW / HH / GMM / NWCL / DC / BlockBootstrap) and 3 KINDs
    (T / J / WALD). The flat enum is over-budget on both axes; any
    further inference algorithm must trigger the structured-shape
    redesign discussion
    before extending the enum.**

    **Convention: ``df`` always means statistical degrees of freedom.**
    Wherever ``df`` appears in factrix StatCode descriptions, metadata
    inner-dict keys (``profile.metadata[StatCode.X]["df"]``), or
    ``profile.inference[...]`` schema fields, it carries the statistics
    sense — never a DataFrame. This matches scipy's API
    (``scipy.stats.chi2.sf(..., df=...)``) and is uniform across the
    codebase. DataFrames are spelled out as ``DataFrame`` in type hints
    and as ``df`` only in user-facing function-argument names where the
    Python variable convention is unambiguous from context.

    ``is_p_value`` returns ``True`` for any code whose
    underscore-separated tokens contain ``"p"``; the
    family-function ``estimator=`` kwarg (#170) dispatches via
    :class:`Estimator.emits_for`
    and is implicitly a p-value source by construction.
    """

    # Primary (cell main effect) — identity carried by profile.config.
    MEAN = "mean"
    # Newey-West HAC (Bartlett kernel): t-statistic + p-value pair.
    # Every primary-inference cell currently emits these two; algorithm
    # is named explicitly so the pair stays symmetric with future
    # P_<algo> / T_<algo> variants.
    T_NW = "t_nw"
    P_NW = "p_nw"
    # HH-pure rectangular-kernel HAC: t-statistic + p-value pair, parallel
    # to (T_NW, P_NW). Conditionally emitted by IC / FM PANEL when
    # forward_periods > 1 (overlap exists).
    T_HH = "t_hh"
    P_HH = "p_hh"
    # GMM J-test (Hansen 1982): J statistic is chi-square distributed
    # under H₀ — not a t-stat — so the (J_GMM, P_GMM) pair replaces the
    # (T_*, P_*) shape. Procedure metadata under either key carries
    # `{"n_moments", "n_params", "df", "weight_matrix_iter", "weight_singular"}`.
    J_GMM = "j_gmm"
    P_GMM = "p_gmm"
    # Cluster-robust Wald χ² for linear restrictions on a slice contrast
    # / joint coefficient (slice-test functions, #153 / #176). KIND = WALD (χ²
    # test statistic, parallel to T); ALGO names the cluster-SE family
    # (parallel to NW / HH naming the kernel family).
    # NWCL = NW Bartlett HAC + one-way cluster on the slice grouping
    # (emitted by `WaldNWCluster`).
    WALD_NWCL = "wald_nwcl"
    P_WALD_NWCL = "p_wald_nwcl"
    # DC = two-way cluster on (date, asset), Cameron-Gelbach-Miller
    # (2011) shape (emitted by `WaldTwoWayCluster`).
    WALD_TWOWAY = "wald_twoway"
    P_WALD_TWOWAY = "p_wald_twoway"
    # Empirical p-value from a block-bootstrap resample of a paired-diff
    # statistic (paired-diff slice tests, #153 / #176). Singleton — the
    # bootstrap distribution itself is not published as a StatCode.
    # `BlockBootstrap` is one Estimator class; fixed vs stationary
    # scheme is a ctor arg living in metadata, so a single P_BOOT key
    # serves both. Parallel to how `NeweyWest`'s lag rule lives in
    # `metadata` rather than splitting `P_NW` by lag.
    P_BOOT = "p_boot"

    # Diagnostic — factor input series.
    FACTOR_ADF_TAU = "factor_adf_tau"
    FACTOR_ADF_P = "factor_adf_p"

    # Diagnostic — regression residual.
    RESID_LJUNG_BOX_Q = "resid_ljung_box_q"
    RESID_LJUNG_BOX_P = "resid_ljung_box_p"

    # Diagnostic — event sample distribution.
    EVENT_HHI_VALUE = "event_hhi_value"

    @property
    def is_p_value(self) -> bool:
        """``True`` iff this stat is a probability in [0, 1].

        Downstream tooling uses this to distinguish probability codes
        from test statistics / point estimates. Tokenises the value on
        ``_`` and checks for a ``p`` token, so primary-inference
        variants ``P_NW`` / ``P_HH`` / ``P_GMM`` all qualify alongside
        diagnostic ``FACTOR_ADF_P`` / ``RESID_LJUNG_BOX_P``.
        """
        return "p" in self.value.split("_")

    @property
    def description(self) -> str:
        """Short gloss for ``profile.diagnose()`` consumers.

        Symmetric with ``WarningCode.description`` / ``InfoCode.description``;
        agents reading ``diagnose()["stats"]`` can resolve a key like
        ``"factor_adf_p"`` to its statistical meaning without grepping
        ``_procedures.py``.
        """
        return _STAT_DESCRIPTIONS[self]


_STAT_DESCRIPTIONS: dict[StatCode, str] = {
    StatCode.MEAN: "Cell primary point estimate (interpretation per "
    "`profile.config.metric`: IC mean, FM λ mean, CAAR event-only mean, "
    "or TS β / E[β]).",
    StatCode.T_NW: "Newey-West HAC t-stat on the cell primary estimate. "
    "Implementation convention lives in `factrix.stats.NeweyWest`.",
    StatCode.P_NW: "Two-sided p-value from the Newey-West HAC t-test on "
    "the cell primary estimate. Sibling of `T_NW`.",
    StatCode.T_HH: "Hansen-Hodrick (1980) rectangular-kernel HAC t-stat "
    "on the cell primary estimate. Sibling of `T_NW`; uses `Var(mean) = "
    "(γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n` instead of NW's Bartlett kernel.",
    StatCode.P_HH: "Two-sided p-value from the Hansen-Hodrick (1980) "
    "rectangular-kernel HAC t-test on the cell primary estimate. "
    "Implementation convention lives in `factrix.stats.HansenHodrick`.",
    StatCode.J_GMM: "Hansen (1982) GMM J-statistic for over-identifying "
    "moment restrictions; chi-square distributed under H₀ with `df = "
    "n_moments - n_params`. Implementation convention lives in "
    "`factrix.stats.GMM`.",
    StatCode.P_GMM: "Right-tail p-value from the Hansen (1982) GMM J-test "
    "(`1 - χ²_df.cdf(J_GMM)`). Sibling under the (J_GMM, P_GMM) "
    "algorithm-pair convention; computed by `factrix.stats.GMM`.",
    StatCode.WALD_NWCL: "Wald χ² statistic for a linear restriction on "
    "slice contrasts / joint coefficients, computed under NW Bartlett HAC "
    "plus one-way cluster on the slice grouping. Implementation convention "
    "lives in `factrix.stats.WaldNWCluster`.",
    StatCode.P_WALD_NWCL: "P-value from `WALD_NWCL`. Sibling under the "
    "(WALD_NWCL, P_WALD_NWCL) algorithm-pair convention.",
    StatCode.WALD_TWOWAY: "Wald χ² statistic for a linear restriction on a "
    "panel coefficient vector, computed under two-way cluster on "
    "(date, asset) (Cameron-Gelbach-Miller 2011). Implementation "
    "convention lives in `factrix.stats.WaldTwoWayCluster`.",
    StatCode.P_WALD_TWOWAY: "P-value from `WALD_TWOWAY`. Sibling under the "
    "(WALD_TWOWAY, P_WALD_TWOWAY) algorithm-pair convention.",
    StatCode.P_BOOT: "Empirical two-sided p-value from a block-bootstrap "
    "resample of a paired-diff statistic. Implementation convention lives "
    "in `factrix.stats.BlockBootstrap` (Politis-Romano stationary or "
    "Künsch fixed scheme; Politis-White auto block length). Single key "
    "for both schemes — scheme choice is metadata, not StatCode.",
    StatCode.FACTOR_ADF_TAU: "ADF τ statistic on the factor input series "
    "(constant-only specification); fed to the MacKinnon 1996 "
    "response-surface for `FACTOR_ADF_P`.",
    StatCode.FACTOR_ADF_P: "ADF unit-root test p-value on the factor input "
    "series (MacKinnon 1996 response-surface; constant-only specification). "
    "p > 0.05 flags persistent regressor regime.",
    StatCode.RESID_LJUNG_BOX_Q: "Ljung-Box Q statistic on regression "
    "residuals (TS-dummy single-asset path); compared against χ²(h) for "
    "`RESID_LJUNG_BOX_P`.",
    StatCode.RESID_LJUNG_BOX_P: "Ljung-Box p-value on residual autocorrelation "
    "(TS-dummy single-asset path); p < 0.05 flags under-set NW lag.",
    StatCode.EVENT_HHI_VALUE: "Herfindahl concentration of event counts "
    "across equal-width period bins on the panel's time axis; high values "
    "flag time-axis clumping. Does not measure within-asset event clustering.",
}
