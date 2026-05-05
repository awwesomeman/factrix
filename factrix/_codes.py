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
    # MIN_BROADCAST_EVENTS_RELIABLE. Per-asset β is identifiable but
    # the cross-event averaging is too thin for asymptotic t to be
    # trusted. Below the HARD floor raises InsufficientSampleError instead.
    SPARSE_COMMON_FEW_EVENTS = "sparse_common_few_events"

    @property
    def description(self) -> str:
        return _WARNING_DESCRIPTIONS[self]


_WARNING_DESCRIPTIONS: dict[WarningCode, str] = {}


_WARNING_DESCRIPTIONS.update(
    {
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS: "n_periods is below MIN_PERIODS_RELIABLE=30; NW HAC SE may be biased.",
        WarningCode.EVENT_WINDOW_OVERLAP: "Adjacent events sit within forward_periods; AR windows overlap.",
        WarningCode.PERSISTENT_REGRESSOR: "ADF p > 0.10 on the continuous factor; β may carry Stambaugh bias.",
        WarningCode.SERIAL_CORRELATION_DETECTED: "Ljung-Box p < 0.05 on residuals; NW lag may be under-set.",
        WarningCode.SMALL_CROSS_SECTION_N: "PANEL cross-asset t-test with n_assets < MIN_ASSETS (10); "
        "df=n_assets-1 too low — t_crit at n_assets=3 ≈ 4.30 "
        "(+119% vs asymptotic 1.96).",
        WarningCode.BORDERLINE_CROSS_SECTION_N: "PANEL cross-asset t-test with MIN_ASSETS ≤ n_assets < "
        "MIN_ASSETS_RELIABLE (10..29); residual t_crit inflation "
        "5–15% — read borderline p-values cautiously.",
        WarningCode.SPARSE_COMMON_FEW_EVENTS: "(COMMON, SPARSE, PANEL) broadcast dummy has "
        "MIN_BROADCAST_EVENTS_HARD ≤ n_events < MIN_BROADCAST_EVENTS_RELIABLE "
        "(5..19); per-asset β estimable but cross-event averaging too thin "
        "for asymptotic t.",
    }
)


def cross_section_tier(n_assets: int) -> WarningCode | None:
    """Map ``n_assets`` to the appropriate cross-asset N warning code.

    Tiers are mutually exclusive — SMALL is strictly more severe than
    BORDERLINE — so callers can membership-check the more severe code
    without an else branch. Returns ``None`` at ``n_assets ≥
    MIN_ASSETS_RELIABLE`` (clean) or ``n_assets < 2`` (PANEL impossible
    by upstream mode routing; defensive).
    """
    from factrix._stats.constants import MIN_ASSETS, MIN_ASSETS_RELIABLE

    if 2 <= n_assets < MIN_ASSETS:
        return WarningCode.SMALL_CROSS_SECTION_N
    if MIN_ASSETS <= n_assets < MIN_ASSETS_RELIABLE:
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
      accept as a ``gate=`` override (BHY step-up requires probabilities
      — feeding it t-stats yields nonsense FDR control).
    - **t-stats** / effect-size means / lag counts / HHI: ``is_p_value``
      returns ``False``. ``profile.verdict(gate=...)`` accepts these
      (the comparison is generic ``value < threshold`` — interpretation
      is the caller's call) but ``bhy(gate=...)`` rejects them.
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

        Used by ``multi_factor.bhy`` to gatekeep the ``gate=`` override
        — BHY step-up math requires p-values, so feeding a t-stat would
        silently corrupt FDR control.
        """
        return self.value.endswith("_p")


class Verdict(StrEnum):
    """Procedure-canonical pass/fail outcome of ``Profile.verdict()``."""

    PASS = "pass"
    FAIL = "fail"
