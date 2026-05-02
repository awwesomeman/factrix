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
# tagged with :attr:`factrix._codes.WarningCode.UNRELIABLE_SE_SHORT_SERIES`.
MIN_PERIODS_RELIABLE: int = 30


def auto_bartlett(T: int) -> int:
    """Newey & West (1994) automatic Bartlett-kernel lag.

    ``floor(4 * (T/100) ** (2/9))``, with a minimum of 1 lag so the
    HAC sum always includes the first autocovariance.
    """
    return max(1, int(4 * (T / 100) ** (2 / 9)))
