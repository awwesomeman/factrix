"""Andrews-Monahan prewhitening in ``_newey_west_se`` — characterisation.

Bands are the *measured* values (kernel-level, 3000 draws when they were
set), not ``<= nominal``: the point is that a change to the kernel in either
direction is visible.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.constants import auto_bartlett
from factrix._stats.core import _p_value_from_t
from factrix._stats.hac import _newey_west_se


def _ar1(phi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    e = np.empty(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.standard_normal() * np.sqrt(1 - phi * phi)
    return e


def _variance_ratio(phi: float, n: int, prewhiten: bool, reps: int = 600) -> float:
    """E[HAC variance of the mean] / true variance of the mean."""
    rng = np.random.default_rng(0)
    true = (1 + phi) / (1 - phi) / n if phi else 1.0 / n
    acc = 0.0
    for _ in range(reps):
        x = _ar1(phi, n, rng) if phi else rng.standard_normal(n)
        acc += _newey_west_se(x, auto_bartlett(n), prewhiten=prewhiten) ** 2
    return acc / reps / true


class TestVarianceRecovery:
    def test_ar06_plain_bartlett_understates_and_prewhitening_recovers(self):
        # measured 0.615 vs 0.969 at n=150 (3000 draws)
        assert 0.50 <= _variance_ratio(0.6, 150, prewhiten=False) <= 0.72
        assert 0.85 <= _variance_ratio(0.6, 150, prewhiten=True) <= 1.10

    def test_iid_is_unaffected(self):
        # measured 0.921 vs 0.930 at n=50
        plain = _variance_ratio(0.0, 50, prewhiten=False)
        pre = _variance_ratio(0.0, 50, prewhiten=True)
        assert abs(plain - pre) < 0.05


class TestRealisedSize:
    @staticmethod
    def _size(phi: float, n: int, prewhiten: bool, reps: int = 500) -> float:
        rng = np.random.default_rng(1)
        rej = 0
        for _ in range(reps):
            x = _ar1(phi, n, rng)
            se = _newey_west_se(x, auto_bartlett(n), prewhiten=prewhiten)
            rej += _p_value_from_t(x.mean() / se, n) < 0.05
        return rej / reps

    def test_ar06_n240(self):
        # measured 0.115 (plain) vs 0.050 (prewhitened); bands ~±2.5 SE at 500
        assert 0.08 <= self._size(0.6, 240, prewhiten=False) <= 0.16
        assert 0.02 <= self._size(0.6, 240, prewhiten=True) <= 0.085

    @pytest.mark.parametrize("n", [60, 240])
    def test_prewhitened_never_worse_than_plain_on_ar06(self, n):
        assert self._size(0.6, n, prewhiten=True) <= self._size(0.6, n, prewhiten=False)
