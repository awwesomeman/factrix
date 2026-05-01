"""v0.5 enum codes for warnings, info notes, cell stats, and verdicts.

``WarningCode`` / ``InfoCode`` / ``StatCode`` follow the ``*Code`` suffix
invariant (§7.5).
"""

from __future__ import annotations

from enum import StrEnum


class WarningCode(StrEnum):
    """Procedure-degradation flags (replaces v3 ``DegradedMode``)."""

    INSUFFICIENT_EVENTS = "insufficient_events"
    INSUFFICIENT_ASSETS = "insufficient_assets"
    UNRELIABLE_SE_SHORT_SERIES = "unreliable_se_short_series"
    EVENT_WINDOW_OVERLAP = "event_window_overlap"
    # Fired when ADF p > 0.1 on a CONTINUOUS factor (Stambaugh-style
    # persistent-regressor flag, §5.2 / §7.3). Not raised for SPARSE.
    PERSISTENT_REGRESSOR = "persistent_regressor"
    SERIAL_CORRELATION_DETECTED = "serial_correlation_detected"


class InfoCode(StrEnum):
    """Neutral facts surfaced to the caller — not warnings, not errors."""

    SCOPE_AXIS_COLLAPSED = "scope_axis_collapsed"


class StatCode(StrEnum):
    """Cell-specific scalar stats keyed in ``FactorProfile.stats``.

    Adding a new metric → add an enum value here + populate it in the
    procedure. Profile schema does not grow.
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


class Verdict(StrEnum):
    """Procedure-canonical pass/fail outcome of ``Profile.verdict()``."""

    PASS = "pass"
    FAIL = "fail"
