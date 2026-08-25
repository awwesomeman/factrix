"""Bandwidth resolution + forward_periods floor on Newey-West."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats import (
    _newey_west_se,
    _newey_west_t_test,
    _resolve_nw_lags,
)
from factrix._stats.constants import auto_bartlett


class TestResolveNwLags:
    def test_default_rule_of_thumb(self):
        assert _resolve_nw_lags(n=100, lags=None, forward_periods=None) == 4
        assert _resolve_nw_lags(n=30, lags=None, forward_periods=None) == 3
        assert _resolve_nw_lags(n=200, lags=None, forward_periods=None) == (
            auto_bartlett(200)
        )

    def test_explicit_lags_passthrough(self):
        assert _resolve_nw_lags(n=100, lags=7, forward_periods=None) == 7

    def test_forward_periods_floors_default_lags(self):
        # auto_bartlett(100) = 4; forward_periods=6 floors at 5.
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

    def test_short_sample_returns_zero(self):
        assert _resolve_nw_lags(n=0, lags=None, forward_periods=5) == 0
        assert _resolve_nw_lags(n=1, lags=None, forward_periods=5) == 0


class TestNewyWestTForwardPeriods:
    def test_forward_periods_changes_se(self):
        # Positively autocorrelated series → larger lags → larger SE
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(200)) * 0.1 + 0.05
        se_default = _newey_west_se(x)
        # forward_periods=20 forces larger lags than the auto default.
        se_h20 = _newey_west_se(x, forward_periods=20)
        assert se_h20 > se_default

    def test_forward_periods_floor_lowers_t_on_plain_bartlett(self):
        """Overlap-aware lag inflates the plain Bartlett SE → |t| shrinks.

        Asserted on ``prewhiten=False``: the property is about the bandwidth
        floor. Once the AR(1) component is prewhitened out of a random walk
        the residual carries almost no autocorrelation left for a longer
        bandwidth to pick up, so the floor is near-inert there by design.
        """
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(200)) * 0.05 + 0.02
        se_naive = _newey_west_se(x, prewhiten=False)
        se_hac = _newey_west_se(x, forward_periods=20, prewhiten=False)
        assert se_hac >= se_naive - 1e-12

    def test_prewhitening_widens_se_on_a_persistent_series(self):
        """Andrews-Monahan recolouring recovers the long-run variance the
        Bartlett kernel truncates away on a near-unit-root series."""
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(200)) * 0.05 + 0.02
        assert _newey_west_se(x) > _newey_west_se(x, prewhiten=False)


class TestRejectsNonFinite:
    """A NaN slips past the ``se < EPSILON`` guard (``max(nan, 0.0)`` is nan)
    and surfaces as a NaN t / p far from the cause; fail loudly instead."""

    def test_se_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            _newey_west_se(np.array([0.1, float("nan"), 0.2, 0.3]))

    def test_t_test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            _newey_west_t_test(np.array([0.1, float("inf"), 0.2, 0.3, 0.1]))
