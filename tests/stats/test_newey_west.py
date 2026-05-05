"""Bandwidth resolution + forward_periods floor on Newey-West."""

from __future__ import annotations

import numpy as np
from factrix._stats import (
    _newey_west_se,
    _newey_west_t_test,
    _resolve_nw_lags,
)


class TestResolveNwLags:
    def test_default_rule_of_thumb(self):
        # np.floor(n^(1/3)) with float arithmetic — 64**(1/3)=3.9999... → 3
        assert _resolve_nw_lags(n=100, lags=None, forward_periods=None) == 4
        assert _resolve_nw_lags(n=30, lags=None, forward_periods=None) == 3

    def test_explicit_lags_passthrough(self):
        assert _resolve_nw_lags(n=100, lags=7, forward_periods=None) == 7

    def test_forward_periods_floors_default_lags(self):
        # floor(100^(1/3)) = 4; forward_periods=6 → floor at 5
        assert _resolve_nw_lags(n=100, lags=None, forward_periods=6) == 5

    def test_forward_periods_floors_explicit_lags(self):
        # explicit lags=2 is too small; h=5 requires at least 4
        assert _resolve_nw_lags(n=100, lags=2, forward_periods=5) == 4

    def test_forward_periods_one_is_noop(self):
        # h=1 means non-overlapping; floor reduces to h-1=0, so default wins
        default = _resolve_nw_lags(n=100, lags=None, forward_periods=None)
        assert _resolve_nw_lags(n=100, lags=None, forward_periods=1) == default

    def test_clip_to_n_minus_one(self):
        # small sample: lag can't exceed n-1 regardless of forward_periods
        assert _resolve_nw_lags(n=5, lags=None, forward_periods=10) == 4


class TestNewyWestTForwardPeriods:
    def test_forward_periods_changes_se(self):
        # Positively autocorrelated series → larger lags → larger SE
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(200)) * 0.1 + 0.05
        se_default = _newey_west_se(x)
        # forward_periods=5 forces larger lags than default floor(200^(1/3))=5,
        # so with h-1=4, already dominated by default; pick larger h to test.
        se_h20 = _newey_west_se(x, forward_periods=20)
        assert se_h20 > se_default

    def test_t_test_forward_periods_lowers_t(self):
        """Overlap-aware lag inflates SE → |t| shrinks on autocorrelated data."""
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(200)) * 0.05 + 0.02
        t_naive, _, _ = _newey_west_t_test(x)
        t_hac, _, _ = _newey_west_t_test(x, forward_periods=20)
        assert abs(t_hac) <= abs(t_naive) + 1e-9
