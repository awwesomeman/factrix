"""Minimal ADF + macro_common.factor_persistent rule."""

from __future__ import annotations

import numpy as np
import pytest

from factrix._stats import _adf, _adf_pvalue_interp


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_adf_stationary_series_rejects_unit_root(rng):
    y = rng.standard_normal(300)
    tau, p = _adf(y)
    assert tau < -2.86, f"white noise should reject unit root at 5%, got τ={tau}"
    assert p < 0.05


def test_adf_random_walk_fails_to_reject(rng):
    y = np.cumsum(rng.standard_normal(300))
    tau, p = _adf(y)
    assert tau > -2.86, f"random walk should not reject unit root, got τ={tau}"
    assert p > 0.10


def test_adf_highly_persistent_ar1_flags_as_unit_root_ish(rng):
    rho = 0.98
    eps = rng.standard_normal(400)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = rho * y[t - 1] + eps[t]
    tau, p = _adf(y)
    assert p > 0.05, f"ρ=0.98 should not reject at 5%, got p={p}"


def test_adf_short_series_returns_degenerate():
    tau, p = _adf(np.array([1.0, 2.0, 3.0]))
    assert (tau, p) == (0.0, 1.0)


@pytest.mark.parametrize(
    "tau,expected_range",
    [
        (-5.0, (0.0, 0.005)),
        (-3.0, (0.02, 0.08)),
        (-2.0, (0.10, 0.45)),
        (0.5, (0.90, 1.0)),
    ],
)
def test_adf_pvalue_interpolation_ranges(tau, expected_range):
    lo, hi = expected_range
    p = _adf_pvalue_interp(tau)
    assert lo <= p <= hi, f"τ={tau}: p={p} outside {expected_range}"
