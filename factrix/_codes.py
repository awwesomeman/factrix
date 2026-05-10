"""v0.5 enum codes for warnings, info notes, cell stats, and verdicts.

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
    # Fired when ADF p > 0.1 on a CONTINUOUS factor (Stambaugh-style
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
    FEW_EVENTS_BROWN_WARNER = "few_events_brown_warner"
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
        WarningCode.FEW_EVENTS_BROWN_WARNER: "CAAR significance test with MIN_EVENTS_HARD ≤ "
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
    by upstream mode routing; defensive).
    """
    from factrix._stats.constants import MIN_ASSETS, MIN_ASSETS_WARN

    if 2 <= n_assets < MIN_ASSETS:
        return WarningCode.SMALL_CROSS_SECTION_N
    if MIN_ASSETS <= n_assets < MIN_ASSETS_WARN:
        return WarningCode.BORDERLINE_CROSS_SECTION_N
    return None


class InfoCode(StrEnum):
    """Neutral facts surfaced to the caller — not warnings, not errors."""

    SCOPE_AXIS_COLLAPSED = "scope_axis_collapsed"

    @property
    def description(self) -> str:
        return _INFO_DESCRIPTIONS[self]


_INFO_DESCRIPTIONS: dict[InfoCode, str] = {
    InfoCode.SCOPE_AXIS_COLLAPSED: "N=1 collapsed scope axis; routed via _SCOPE_COLLAPSED sentinel.",
}


class StatCode(StrEnum):
    """Cell-specific scalar stats keyed in ``FactorProfile.stats``.

    Naming grammar (#187): each code is ``<TARGET>_<KIND>``.

    - ``TARGET`` — what is being measured. **Primary** (cell main
      effect) carries no prefix because ``FactorProfile.config``
      (``scope`` / ``signal`` / ``metric``) is the single source of
      truth for cell identity. **Diagnostic** carries an explicit
      prefix (``FACTOR_`` / ``RESID_`` / ``EVENT_``) because the
      target lives outside ``config``.
    - ``KIND`` — what kind of number. ``_MEAN`` / ``_VALUE`` for point
      estimates, statistic-named suffixes (``_T_NW`` / ``_TAU`` /
      ``_Q``) for test statistics, ``_P`` for p-values. Algorithm
      variants of the same p-value get a further suffix
      (``_P_HH`` / ``_P_GMM``).

    **Inference primary stats — algorithm-suffixed pair shape**

    Each inference algorithm emits a (test statistic, p-value) pair with
    KINDs that abbreviate the test statistic's reference distribution:
    ``T`` (Student-t / asymptotic normal), ``J`` (Hansen J / χ²),
    ``WALD`` (Wald χ²), ``F`` (Snedecor F), ``LR`` (likelihood ratio).
    Currently shipping: ``(T_NW, P)`` for Newey-West and ``(T_HH, P_HH)``
    for Hansen-Hodrick. Planned in #191: ``(J_GMM, P_GMM)`` — the
    over-identification test is χ², not t, so GMM emits J rather than T.
    The bare ``P`` alias for ``P_NW`` is tracked for rename in #192.

    **Redesign trigger** — when (a) ≥ 4 inference algorithms ship
    concurrently or (b) ≥ 3 distinct test-statistic KINDs (T / J /
    Wald / F / LR) coexist, the flat ``<KIND>_<ALGO>`` enum becomes
    a (kind × algo) cardinality product and a structured shape
    (``profile.inference[Algo.X] = {test_stat, kind, p, df}``) earns
    its breaking-change cost. Below those thresholds the flat
    enum stays cheaper.

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
    underscore-separated tokens contain ``"p"``; family-verb
    ``estimator=`` (#170) dispatches via :class:`Estimator.emits_for`
    and is implicitly a p-value source by construction.
    """

    # Primary (cell main effect) — identity carried by profile.config.
    MEAN = "mean"
    T_NW = "t_nw"
    P = "p"
    # HH-pure rectangular-kernel HAC: t-statistic + p-value emitted as
    # a pair, parallel to the (T_NW, P) NW pair. Conditionally emitted by
    # IC / FM PANEL when forward_periods > 1 (overlap exists).
    T_HH = "t_hh"
    P_HH = "p_hh"
    # GMM J-test (Hansen 1982): J statistic is chi-square distributed
    # under H₀ — not a t-stat — so the (J_GMM, P_GMM) pair replaces the
    # (T_*, P_*) shape. The pair lands together with the GMM procedure
    # in #191; the StatCode grammar in this module's docstring already
    # documents the planned shape.
    P_GMM = "p_gmm"

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

        Used by ``profile.verdict(gate=...)`` and downstream tooling to
        distinguish probability codes from test statistics / point
        estimates. Tokenises the value on ``_`` and checks for a ``p``
        token, so bare ``P`` and algorithm variants ``P_HH`` / ``P_GMM``
        all qualify alongside diagnostic ``FACTOR_ADF_P`` /
        ``RESID_LJUNG_BOX_P``.
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
    StatCode.P: "Two-sided p-value from the NW HAC t-test on the cell "
    "primary estimate.",
    StatCode.T_HH: "Hansen-Hodrick (1980) rectangular-kernel HAC t-stat "
    "on the cell primary estimate. Sibling of `T_NW`; uses `Var(mean) = "
    "(γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n` instead of NW's Bartlett kernel.",
    StatCode.P_HH: "Two-sided p-value from the Hansen-Hodrick (1980) "
    "rectangular-kernel HAC t-test on the cell primary estimate. "
    "Implementation convention lives in `factrix.stats.HansenHodrick`.",
    StatCode.P_GMM: "Two-sided p-value from a Hansen (1982) GMM J-test "
    "(over-identifying restrictions). Reserved for procedure-side landing "
    "in a follow-up issue.",
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
    "across calendar bins (cross-sectional / time-axis); high values flag "
    "calendar-time clumping. Does not measure within-asset event clustering.",
}


class Verdict(StrEnum):
    """Procedure-canonical pass/fail outcome of ``Profile.verdict()``."""

    PASS = "pass"
    FAIL = "fail"
