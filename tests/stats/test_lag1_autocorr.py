"""``_lag1_autocorr`` — the persistence screen behind SERIAL_CORRELATION_DETECTED."""

from __future__ import annotations

import numpy as np
from factrix._stats.constants import PERSISTENT_SERIES_AUTOCORR
from factrix._stats.diagnostics import _lag1_autocorr


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.standard_normal() * np.sqrt(1 - phi * phi)
    return e


def test_iid_is_near_zero_and_below_threshold():
    x = np.random.default_rng(0).standard_normal(240)
    assert abs(_lag1_autocorr(x)) < 0.15
    assert _lag1_autocorr(x) < PERSISTENT_SERIES_AUTOCORR


def test_persistent_series_exceeds_threshold():
    assert _lag1_autocorr(_ar1(0.85, 240, 1)) > PERSISTENT_SERIES_AUTOCORR
    assert _lag1_autocorr(_ar1(0.6, 240, 2)) > PERSISTENT_SERIES_AUTOCORR


def test_degenerate_inputs_do_not_trip_the_screen():
    assert _lag1_autocorr(np.full(50, 0.3)) == 0.0
    assert _lag1_autocorr(np.array([1.0, 2.0])) == 0.0
    assert _lag1_autocorr(np.array([])) == 0.0
