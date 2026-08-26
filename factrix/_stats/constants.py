"""TIMESERIES sample-size thresholds and Newey-West Bartlett bandwidth.

Single source of truth for the sample-size floors: no literal ``20`` /
``30`` / ``floor(4 * (T/100)**(2/9))`` may appear elsewhere in factrix.
"""

from __future__ import annotations

import math

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


def har_bandwidth(T: int) -> int:
    """[Lazarus-Lewis-Stock-Watson (2018)][llsw-2018] HAR Bartlett-kernel bandwidth.

    ``ceil(1.3 * sqrt(T))``, minimum 1. This is the "rule of thumb"
    bandwidth LLSW recommend for practice, paired with fixed-b (rather
    than standard-normal) critical values — see
    :func:`factrix._stats.hac._har_dof`.

    Why this and not :func:`auto_bartlett`: the Newey-West (1994)
    ``4·(T/100)^{2/9}`` plug-in grows so slowly (4 lags at T=100, 5 at
    T=500) that the Bartlett long-run variance is badly downward-biased
    at research sample sizes, and the resulting mean t-test over-rejects
    — measured 11–21% at a nominal 5% on AR(0.6), and 12–34% on
    overlapping h-period returns. ``auto_bartlett`` is retained for the
    Driscoll-Kraay and Wald paths, which sandwich a different score
    sequence and are size-checked separately.
    """
    if T <= 0:
        return 1
    return max(1, math.ceil(1.3 * math.sqrt(T)))


# Lag-1 autocorrelation of the tested per-period series (IC series, per-period
# betas, spread series) above which a trending / regime-like series makes the
# mean test look more significant than it is. One constant for every path,
# set by the path it protects LEAST well; the prewhitened Newey-West path is
# conservative under it. Measured realised size at nominal 5%, n = 240, on
# the persistent-IC fixture whose own phi = 0 baseline is 8–9%:
#
#   fixture phi (lag-1 of IC)   NON_OVERLAPPING h=1 / bootstrap / NW plain / NW prewhitened
#   0.3  (0.26)                 —           / —          / 9.0%     / 8.0%
#   0.45 (0.39)                 —           / —          / 11.3%    / 7.3%
#   0.6  (0.53)                 32–34%      / 12–19%     / 13.7%    / 7.3%
#   0.85 (0.78)                 55–61%      / 20–32%     / 32.0%    / 15.0%
#
# 0.3 is where the plain-Bartlett excess starts; the un-prewhitened members
# (plain t on the strided sample, stationary bootstrap, Hansen-Hodrick) are
# 2.5–6x nominal by 0.5 and set the constant. Prewhitened Newey-West only
# leaves its baseline around lag-1 0.6, so the screen fires early for it —
# deliberately: a user who knows the kernel is prewhitened would otherwise
# assume a persistent series is handled.
# The screen is one-sided: only positive persistence was measured. Strong
# NEGATIVE autocorrelation also breaks HAC calibration (it is what drives
# RECT_KERNEL_NEGATIVE_VARIANCE) and is deliberately not covered here.
PERSISTENT_SERIES_AUTOCORR: float = 0.3
