"""Hansen-Hodrick (1980) rectangular-kernel HAC primitives."""

from __future__ import annotations

import numpy as np
from factrix._stats import (
    _hansen_hodrick_se,
    _hansen_hodrick_t_test,
    _newey_west_se,
)


class TestHansenHodrickSe:
    def test_h_one_collapses_to_iid(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        se, clamped = _hansen_hodrick_se(x, forward_periods=1)
        # h=1: only γ₀ term. Compare against std/sqrt(n) using the same
        # ddof=0 convention HH uses internally (γ₀ divides by n, not n-1).
        expected = float(np.std(x, ddof=0)) / np.sqrt(len(x))
        assert clamped is False
        assert abs(se - expected) < 1e-12

    def test_positive_autocorrelation_inflates_se(self):
        rng = np.random.default_rng(1)
        x = np.cumsum(rng.standard_normal(300)) * 0.05 + 0.02
        se_iid, _ = _hansen_hodrick_se(x, forward_periods=1)
        se_overlap, clamped = _hansen_hodrick_se(x, forward_periods=12)
        assert clamped is False
        assert se_overlap > se_iid

    def test_negative_variance_clamps_and_flags(self):
        # Strong negative lag-1 autocovariance: γ₀ + 2γ₁ < 0.
        # An alternating series ±a has γ₀ ≈ a², γ₁ ≈ -a², so γ₀ + 2γ₁ ≈ -a².
        x = np.array([1.0, -1.0] * 50)
        se, clamped = _hansen_hodrick_se(x, forward_periods=2)
        assert clamped is True
        assert se == 0.0

    def test_short_sample_returns_zero(self):
        se, clamped = _hansen_hodrick_se(np.array([1.0]), forward_periods=4)
        assert se == 0.0
        assert clamped is False

    def test_lag_capped_at_n_minus_one(self):
        # forward_periods exceeds n: lag should clip to n-1, math stays finite.
        x = np.array([1.0, 2.0, 3.0, 4.0])
        se, clamped = _hansen_hodrick_se(x, forward_periods=10)
        assert np.isfinite(se)
        # Same x with forward_periods=4 hits the same effective lag (n-1=3).
        se_at_cap, _ = _hansen_hodrick_se(x, forward_periods=4)
        assert abs(se - se_at_cap) < 1e-12
        assert clamped is False


class TestHansenHodrickTTest:
    def test_h_one_matches_iid_t(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(150) + 0.3
        t, _, _, clamped = _hansen_hodrick_t_test(x, forward_periods=1)
        assert clamped is False
        # Cross-check against NW with explicit lags=0 (γ₀-only, no kernel
        # tail) — both reduce to √(γ₀ / n).
        se_nw = _newey_west_se(x, lags=0)
        se_hh, _ = _hansen_hodrick_se(x, forward_periods=1)
        assert abs(se_nw - se_hh) < 1e-12
        assert abs(t) > 1.0  # mean=0.3, n=150 — clearly non-zero

    def test_clamped_returns_p_one(self):
        x = np.array([1.0, -1.0] * 50)
        t, p, marker, clamped = _hansen_hodrick_t_test(x, forward_periods=2)
        assert clamped is True
        assert t == 0.0
        assert p == 1.0
        assert marker == ""

    def test_short_sample(self):
        t, p, marker, clamped = _hansen_hodrick_t_test(
            np.array([1.0, 2.0]), forward_periods=2
        )
        assert (t, p, marker, clamped) == (0.0, 1.0, "", False)

    def test_overlap_inflates_se_lowers_t(self):
        rng = np.random.default_rng(11)
        x = np.cumsum(rng.standard_normal(300)) * 0.02 + 0.05
        t_iid, _, _, _ = _hansen_hodrick_t_test(x, forward_periods=1)
        t_overlap, _, _, _ = _hansen_hodrick_t_test(x, forward_periods=12)
        assert abs(t_overlap) <= abs(t_iid) + 1e-9
