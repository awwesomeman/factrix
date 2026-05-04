"""TIMESERIES sample-size thresholds and Newey-West Bartlett bandwidth.

Centralised per refactor_api.md §5.2 (A3): no literal ``20`` / ``30`` /
``floor(4 * (T/100)**(2/9))`` may appear elsewhere in factrix.
"""

from __future__ import annotations


# ``T < MIN_PERIODS_HARD`` → :class:`factrix._errors.InsufficientSampleError`
# (no verdict — NW HAC SE biased beyond the floor where verdict can be
# trusted at all).
MIN_PERIODS_HARD: int = 20

# ``MIN_PERIODS_HARD <= T < MIN_PERIODS_RELIABLE`` → verdict still emitted, but
# tagged with :attr:`factrix._codes.WarningCode.UNRELIABLE_SE_SHORT_PERIODS`.
MIN_PERIODS_RELIABLE: int = 30

# ``n_assets < MIN_ASSETS`` → :attr:`factrix._codes.WarningCode.SMALL_CROSS_SECTION_N`
# from PANEL ``common_continuous`` and from ``suggest_config``. The cross-asset
# t-test on E[β] has df = n_assets - 1; below 10 the critical t inflates
# severely (n_assets=3 → t_crit≈4.30 vs asymptotic 1.96). The axis intentionally
# never raises — N=2..9 is well-defined statistics, only weak.
# Naming deliberately omits ``_HARD``: the n_assets axis only warns, so reusing
# the n_periods ``_HARD`` (which means "raise") would be misleading.
MIN_ASSETS: int = 10

# ``MIN_ASSETS <= n_assets < MIN_ASSETS_RELIABLE`` →
# :attr:`factrix._codes.WarningCode.BORDERLINE_CROSS_SECTION_N`. Mirrors
# ``MIN_PERIODS_RELIABLE`` semantically: residual t_crit inflation 5–15%
# (n_assets=10 → +15%, n_assets=20 → +7%, n_assets=30 → +5%).
MIN_ASSETS_RELIABLE: int = 30


def auto_bartlett(T: int) -> int:
    """Newey & West (1994) automatic Bartlett-kernel lag.

    ``floor(4 * (T/100) ** (2/9))``, with a minimum of 1 lag so the
    HAC sum always includes the first autocovariance.
    """
    return max(1, int(4 * (T / 100) ** (2 / 9)))
