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

    Adding a new metric → add an enum value here + populate it in the
    procedure. Profile schema does not grow. Stats fall in three
    families:

    - **p-values**: identifier ends in ``_p`` (``IC_P`` / ``FM_LAMBDA_P``
      / ``TS_BETA_P`` / ``CAAR_P`` plus the diagnostic-only
      ``FACTOR_ADF_P`` / ``LJUNG_BOX_P``). ``is_p_value`` returns
      ``True``. These are the only codes ``multi_factor.bhy`` will
      accept as a ``p_stat=`` override (BHY step-up requires
      probabilities — feeding it t-stats yields nonsense FDR control).
    - **t-stats** / effect-size means / lag counts / HHI: ``is_p_value``
      returns ``False``. ``profile.verdict(gate=...)`` accepts these
      (the comparison is generic ``value < threshold`` — interpretation
      is the caller's call) but ``bhy(p_stat=...)`` rejects them.
    """

    IC_MEAN = "ic_mean"
    IC_T_NW = "ic_t_nw"
    IC_P = "ic_p"
    FM_LAMBDA_MEAN = "fm_lambda_mean"
    FM_LAMBDA_T_NW = "fm_lambda_t_nw"
    FM_LAMBDA_P = "fm_lambda_p"
    TS_BETA = "ts_beta"
    TS_BETA_T_NW = "ts_beta_t_nw"
    TS_BETA_P = "ts_beta_p"
    CAAR_MEAN = "caar_mean"
    CAAR_T_NW = "caar_t_nw"
    CAAR_P = "caar_p"
    FACTOR_ADF_P = "factor_adf_p"
    LJUNG_BOX_P = "ljung_box_p"
    EVENT_TEMPORAL_HHI = "event_temporal_hhi"
    NW_LAGS_USED = "nw_lags_used"

    @property
    def is_p_value(self) -> bool:
        """``True`` iff this stat is a probability in [0, 1].

        Used by ``multi_factor.bhy`` (and the shared family resolution
        layer) to gatekeep the ``p_stat=`` override — BHY step-up math
        requires p-values, so feeding a t-stat would silently corrupt
        FDR control.
        """
        return self.value.endswith("_p")

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
    StatCode.IC_MEAN: "Per-date Spearman IC, mean over the time series.",
    StatCode.IC_T_NW: "NW HAC t-stat on the IC mean. "
    "Implementation convention lives in `factrix.stats.NeweyWest`.",
    StatCode.IC_P: "Two-sided p-value from the NW HAC t-test on IC.",
    StatCode.FM_LAMBDA_MEAN: "Fama-MacBeth λ — per-date OLS slope of "
    "forward_return on factor, averaged over time.",
    StatCode.FM_LAMBDA_T_NW: "NW HAC t-stat on the FM λ mean. "
    "Implementation convention lives in `factrix.stats.NeweyWest`.",
    StatCode.FM_LAMBDA_P: "Two-sided p-value from the NW HAC t-test on FM λ.",
    StatCode.TS_BETA: "Per-asset OLS β on the broadcast factor; "
    "PANEL = cross-asset mean of β_i, TIMESERIES = single-series β.",
    StatCode.TS_BETA_T_NW: "PANEL: cross-asset t on E[β]. "
    "TIMESERIES: NW HAC t on the single-series β. "
    "TIMESERIES convention lives in `factrix.stats.NeweyWest`.",
    StatCode.TS_BETA_P: "Two-sided p-value from the TS_BETA t-test.",
    StatCode.CAAR_MEAN: "Per-event-date weighted abnormal return, "
    "averaged across event dates (event-only mean — see _procedures for "
    "the dense-calendar t-stat).",
    StatCode.CAAR_T_NW: "NW HAC t-stat on the dense-calendar CAAR series "
    "(zero-fill on non-event dates). "
    "Implementation convention lives in `factrix.stats.NeweyWest`.",
    StatCode.CAAR_P: "Two-sided p-value from the NW HAC t-test on CAAR.",
    StatCode.FACTOR_ADF_P: "ADF unit-root test p-value on the factor input "
    "series (MacKinnon 1996 response-surface; constant-only specification). "
    "p > 0.05 flags persistent regressor regime.",
    StatCode.LJUNG_BOX_P: "Ljung-Box p-value on residual autocorrelation "
    "(TS-dummy single-asset path); p < 0.05 flags under-set NW lag.",
    StatCode.EVENT_TEMPORAL_HHI: "Herfindahl concentration of event counts "
    "across calendar bins (cross-sectional / time-axis); high values flag "
    "calendar-time clumping. Does not measure within-asset event clustering.",
    StatCode.NW_LAGS_USED: "NW HAC lag count selected by the auto-bandwidth "
    "rule for the cell's primary t-test.",
}


class Verdict(StrEnum):
    """Procedure-canonical pass/fail outcome of ``Profile.verdict()``."""

    PASS = "pass"
    FAIL = "fail"
