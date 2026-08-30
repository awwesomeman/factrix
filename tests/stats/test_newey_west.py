"""Bandwidth resolution + overlap_periods floor on Newey-West."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats import (
    _newey_west_se,
    _newey_west_t_test,
    _resolve_nw_lags,
    _resolve_scalar_wald_hac,
)
from factrix._stats.constants import auto_bartlett, har_bandwidth


class TestResolveNwLags:
    def test_default_rule_of_thumb(self):
        assert _resolve_nw_lags(n=100, lags=None, overlap_periods=None) == 4
        assert _resolve_nw_lags(n=30, lags=None, overlap_periods=None) == 3
        assert _resolve_nw_lags(n=200, lags=None, overlap_periods=None) == (
            auto_bartlett(200)
        )

    def test_explicit_lags_passthrough(self):
        assert _resolve_nw_lags(n=100, lags=7, overlap_periods=None) == 7

    def test_forward_periods_floors_default_lags(self):
        # auto_bartlett(100) = 4; overlap_periods=6 floors at 5.
        assert _resolve_nw_lags(n=100, lags=None, overlap_periods=6) == 5

    def test_forward_periods_floors_explicit_lags(self):
        # explicit lags=2 is too small; h=5 requires at least 4
        assert _resolve_nw_lags(n=100, lags=2, overlap_periods=5) == 4

    def test_forward_periods_one_is_noop(self):
        # h=1 means non-overlapping; floor reduces to h-1=0, so default wins
        default = _resolve_nw_lags(n=100, lags=None, overlap_periods=None)
        assert _resolve_nw_lags(n=100, lags=None, overlap_periods=1) == default

    def test_clip_to_n_minus_one(self):
        # small sample: lag can't exceed n-1 regardless of overlap_periods
        assert _resolve_nw_lags(n=5, lags=None, overlap_periods=10) == 4

    def test_short_sample_returns_zero(self):
        assert _resolve_nw_lags(n=0, lags=None, overlap_periods=5) == 0
        assert _resolve_nw_lags(n=1, lags=None, overlap_periods=5) == 0


class TestResolveScalarWaldHac:
    """The restriction-count split: one restriction gets the HAR recipe."""

    def test_single_restriction_takes_the_wide_overlap_floor(self):
        # h = 21 on 240 periods: the narrow rule floors at h - 1 = 20 (and the
        # Newey-West base is smaller still); the scalar rule floors at 3(h - 1).
        lags, _, _ = _resolve_scalar_wald_hac(n=240, lags=None, overlap_periods=21)
        assert lags == 60
        assert _resolve_nw_lags(n=240, lags=None, overlap_periods=21) == 20

    def test_multi_restriction_rule_is_unchanged_by_the_split(self):
        # The K >= 2 Wald consumers keep max(auto_bartlett(T), h - 1): a wide
        # kernel on a K x K HAC matrix measured worse, not better.
        for h in (1, 5, 21):
            assert _resolve_nw_lags(n=240, lags=None, overlap_periods=h) == max(
                auto_bartlett(240), h - 1
            )

    def test_base_rule_is_the_har_bandwidth_not_the_newey_west_plug_in(self):
        lags, _, _ = _resolve_scalar_wald_hac(n=240, lags=None, overlap_periods=1)
        assert lags == har_bandwidth(240)
        assert lags > auto_bartlett(240)

    def test_returns_the_finite_sample_scale_and_effective_dof(self):
        n = 240
        lags, scale, dof = _resolve_scalar_wald_hac(n, lags=None, overlap_periods=5)
        assert scale == pytest.approx(n / (n - lags - 1))
        assert scale > 1.0
        # Fixed-b effective df, far below the regression's n - k.
        assert 1.0 <= dof < n - 1

    def test_degenerate_sample_is_safe(self):
        lags, scale, dof = _resolve_scalar_wald_hac(n=1, lags=None, overlap_periods=5)
        assert lags == 0
        assert scale == 1.0
        assert dof >= 1.0


class TestNewyWestTForwardPeriods:
    def test_forward_periods_changes_se(self):
        # Positively autocorrelated series → larger lags → larger SE
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(200)) * 0.1 + 0.05
        se_default = _newey_west_se(x)
        # overlap_periods=20 forces larger lags than the auto default.
        se_h20 = _newey_west_se(x, overlap_periods=20)
        assert se_h20 > se_default

    def test_t_test_forward_periods_lowers_t(self):
        """Overlap-aware lag inflates SE → |t| shrinks on autocorrelated data."""
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(200)) * 0.05 + 0.02
        t_naive, _, _ = _newey_west_t_test(x)
        t_hac, _, _ = _newey_west_t_test(x, overlap_periods=20)
        assert abs(t_hac) <= abs(t_naive) + 1e-9

    def test_prewhitening_widens_se_on_a_persistent_series(self):
        """Opt-in Andrews-Monahan recolouring recovers the long-run variance
        the Bartlett kernel truncates away on a near-unit-root series."""
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(200)) * 0.05 + 0.02
        assert _newey_west_se(x, prewhiten=True) > _newey_west_se(x)


class TestRejectsNonFinite:
    """A NaN slips past the ``se < EPSILON`` guard (``max(nan, 0.0)`` is nan)
    and surfaces as a NaN t / p far from the cause; fail loudly instead."""

    def test_se_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            _newey_west_se(np.array([0.1, float("nan"), 0.2, 0.3]))

    def test_t_test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            _newey_west_t_test(np.array([0.1, float("inf"), 0.2, 0.3, 0.1]))
