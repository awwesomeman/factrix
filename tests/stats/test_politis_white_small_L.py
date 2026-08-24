"""Politis-White: a plug-in estimate below one block clamps to 1, not the generic rule."""

from __future__ import annotations

import numpy as np
from factrix._stats.bootstrap import _politis_white_block_length


def test_iid_series_does_not_fall_back_to_generic_rule():
    rng = np.random.default_rng(0)
    T = 250
    fallback = 1.75 * T ** (1 / 3)
    hits = 0
    for _ in range(100):
        L = _politis_white_block_length(rng.normal(size=T))
        assert L >= 1.0
        hits += fallback == L
    # The generic rule fired on ~40% of iid draws before the fix.
    assert hits <= 5


def test_dependent_series_gets_long_blocks():
    rng = np.random.default_rng(1)
    T = 500
    e = rng.normal(size=T)
    x = np.empty(T)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = 0.8 * x[t - 1] + e[t]
    assert _politis_white_block_length(x) > 5.0
