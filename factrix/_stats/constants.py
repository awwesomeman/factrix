"""TIMESERIES sample-size thresholds and Newey-West Bartlett bandwidth.

Single source of truth for the sample-size floors: no literal ``20`` /
``30`` / ``floor(4 * (T/100)**(2/9))`` may appear elsewhere in factrix.
"""

from __future__ import annotations

# ``T < MIN_PERIODS_HARD`` → :class:`factrix._errors.InsufficientSampleError`
# (no result — NW HAC SE biased beyond the floor where inference can
# be trusted at all).
MIN_PERIODS_HARD: int = 20

# ``MIN_PERIODS_HARD <= T < MIN_PERIODS_WARN`` → result still emitted, but
# tagged with :attr:`factrix._codes.WarningCode.UNRELIABLE_SE_SHORT_PERIODS`.
MIN_PERIODS_WARN: int = 30

# ``n_assets < MIN_ASSETS_WARN`` → :attr:`factrix._codes.WarningCode.FEW_ASSETS`
# from PANEL ``common_continuous`` and from ``suggest_config``. The cross-asset
# t-test on E[β] has df = n_assets - 1; as n_assets shrinks the critical t
# inflates (n_assets=3 → t_crit≈4.30 vs asymptotic 1.96; ~+15% at 10, ~+5% at
# 30). The axis intentionally never raises — N≥2 is well-defined statistics,
# only weak — so a single warn floor (no ``_HARD``) flags the whole thin
# regime; severity is read from the ``n_assets`` metadata, not encoded in tiers.
MIN_ASSETS_WARN: int = 30


def auto_bartlett(T: int) -> int:
    """[Newey-West (1994)][newey-west-1994] automatic Bartlett-kernel lag.

    ``floor(4 * (T/100) ** (2/9))``, with a minimum of 1 lag so the
    heteroskedasticity-and-autocorrelation-consistent (HAC) sum always includes the first autocovariance.
    """
    return max(1, int(4 * (T / 100) ** (2 / 9)))


# Lag-1 autocorrelation of a per-date series above which no HAC path in the
# library is calibrated. Measured (nominal 5%, true null, per-date IC series):
# at phi = 0 every inference sits at 5–9%; at phi = 0.6 Newey-West rejects
# 13–17%, the stationary bootstrap 12–19% and a plain t 32–34%; at phi = 0.85
# NW 32–34%, bootstrap 20–32%, plain t 55–61%. The kernel-level AR(0.3) row is
# already 8–11%, so 0.3 is where the excess starts rather than where it peaks.
# The screen is one-sided: only positive persistence was measured. Strong
# NEGATIVE autocorrelation also breaks HAC calibration (it is what drives
# RECT_KERNEL_NEGATIVE_VARIANCE) and is deliberately not covered here.
PERSISTENT_SERIES_AUTOCORR: float = 0.3
