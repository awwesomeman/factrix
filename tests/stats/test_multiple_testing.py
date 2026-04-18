"""BHY multiple-testing procedure tests."""

from __future__ import annotations

import numpy as np
import pytest

from factorlib.stats.multiple_testing import bhy_adjust, bhy_adjusted_p


class TestBhyAdjust:
    def test_empty_input(self):
        mask = bhy_adjust([])
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_all_ones_rejects_nothing(self):
        mask = bhy_adjust(np.ones(5))
        assert not mask.any()

    def test_all_near_zero_rejects_everything(self):
        mask = bhy_adjust(np.full(5, 1e-10))
        assert mask.all()

    def test_hand_computed_example(self):
        # Five p-values; compute critical values by hand.
        # c(5) = 1 + 1/2 + 1/3 + 1/4 + 1/5 = 2.28333
        # At fdr=0.10: crit_k = k/(5 * c) * 0.10 for k=1..5
        #   crit_1 = 0.00876, crit_2 = 0.01753, crit_3 = 0.02629,
        #   crit_4 = 0.03506, crit_5 = 0.04383
        # Sorted p = [0.001, 0.01, 0.03, 0.08, 0.20]
        # Pass check: p_(1)<=crit_1 True, p_(2)<=crit_2 True, p_(3)<=crit_3 False,
        #             p_(4)<=crit_4 False, p_(5)<=crit_5 False
        # k_max = 2 → reject first two
        p = np.array([0.001, 0.01, 0.03, 0.08, 0.20])
        mask = bhy_adjust(p, fdr=0.10)
        assert list(mask) == [True, True, False, False, False]

    def test_order_invariance(self):
        p = np.array([0.20, 0.001, 0.08, 0.01, 0.03])  # shuffled
        mask = bhy_adjust(p, fdr=0.10)
        # The two smallest (indices 1 and 3, p=0.001 and 0.01) should pass
        assert list(mask) == [False, True, False, True, False]

    def test_fdr_bounds(self):
        with pytest.raises(ValueError, match="fdr must be in"):
            bhy_adjust([0.01, 0.02], fdr=0.0)
        with pytest.raises(ValueError, match="fdr must be in"):
            bhy_adjust([0.01, 0.02], fdr=1.0)

    def test_rejects_out_of_range_pvalues(self):
        with pytest.raises(ValueError, match="must all lie in"):
            bhy_adjust([-0.1, 0.5])
        with pytest.raises(ValueError, match="must all lie in"):
            bhy_adjust([0.5, 1.1])

    def test_higher_fdr_rejects_more(self):
        p = np.array([0.001, 0.01, 0.03, 0.08, 0.20])
        n_low = int(bhy_adjust(p, fdr=0.01).sum())
        n_high = int(bhy_adjust(p, fdr=0.25).sum())
        assert n_high >= n_low

    def test_single_element_below_fdr_rejects(self):
        # n=1: c(1) = 1, critical = 1/(1*1)*fdr = fdr. Any p <= fdr passes.
        assert bhy_adjust(np.array([0.01]), fdr=0.05).tolist() == [True]

    def test_single_element_above_fdr_does_not_reject(self):
        assert bhy_adjust(np.array([0.20]), fdr=0.05).tolist() == [False]


class TestBhyAdjustedP:
    def test_empty_input(self):
        adj = bhy_adjusted_p([])
        assert adj.shape == (0,)

    def test_clipped_at_one(self):
        # A loose p-value with many tests should get adjusted to 1.0
        p = np.concatenate([[0.5], np.full(9, 0.9)])
        adj = bhy_adjusted_p(p)
        assert adj.max() <= 1.0
        assert (adj >= 0.9).all()

    def test_monotonicity_in_rank(self):
        # Adjusted p must be non-decreasing when sorted by original p
        p = np.random.default_rng(0).uniform(size=50)
        adj = bhy_adjusted_p(p)
        sorted_by_p = adj[np.argsort(p)]
        # Allow tiny float noise
        diffs = np.diff(sorted_by_p)
        assert (diffs >= -1e-12).all()

    def test_hand_example_matches_scipy_like_formula(self):
        # For p_sorted = [0.001, 0.01, 0.03, 0.08, 0.20] and c_m = 2.28333
        # scaled_k = (5 * c_m / k) * p_(k)
        # then cummin from the right, clip at 1.
        p = np.array([0.001, 0.01, 0.03, 0.08, 0.20])
        adj = bhy_adjusted_p(p)
        c_m = sum(1.0 / k for k in range(1, 6))
        scaled = [(5 * c_m / k) * p[k - 1] for k in range(1, 6)]
        # cummin from right
        expected_sorted = list(scaled)
        for i in range(3, -1, -1):
            expected_sorted[i] = min(expected_sorted[i], expected_sorted[i + 1])
        expected_sorted = [min(x, 1.0) for x in expected_sorted]
        np.testing.assert_allclose(adj, expected_sorted, rtol=1e-12)

    def test_consistent_with_bhy_adjust(self):
        # A factor is significant at fdr=alpha iff adjusted_p <= alpha
        p = np.array([0.001, 0.01, 0.03, 0.08, 0.20])
        fdr = 0.10
        mask = bhy_adjust(p, fdr=fdr)
        adj = bhy_adjusted_p(p)
        np.testing.assert_array_equal(mask, adj <= fdr)


class TestNTotal:
    """Two-stage screening: n_total controls the BHY denominator."""

    def test_default_matches_explicit_len(self):
        """n_total=None and n_total=len(p) must produce identical output."""
        p = np.array([0.001, 0.01, 0.03, 0.08, 0.20])
        # bhy_adjust
        np.testing.assert_array_equal(
            bhy_adjust(p, fdr=0.10),
            bhy_adjust(p, fdr=0.10, n_total=len(p)),
        )
        # bhy_adjusted_p
        np.testing.assert_allclose(
            bhy_adjusted_p(p),
            bhy_adjusted_p(p, n_total=len(p)),
        )

    def test_larger_n_total_is_stricter(self):
        """n_total > len(p) rejects fewer and raises adjusted p-values."""
        # c(3) = 1.833; at fdr=0.05, rank-1 crit = 0.05/(3*1.833) = 0.00909
        #   → p=0.001 rejected.
        # c(1000) ≈ 7.485; rank-1 crit = 0.05/(1000*7.485) ≈ 6.68e-6
        #   → p=0.001 NOT rejected.
        p = np.array([0.001, 0.01, 0.04])
        mask_tight = bhy_adjust(p, fdr=0.05, n_total=3)
        mask_loose = bhy_adjust(p, fdr=0.05, n_total=1000)
        assert mask_tight.sum() >= mask_loose.sum()
        assert mask_tight[0]          # p=0.001 survives at n=3
        assert not mask_loose[0]      # but not at n=1000

        adj_tight = bhy_adjusted_p(p, n_total=3)
        adj_loose = bhy_adjusted_p(p, n_total=1000)
        # Adjusted p must be at least as large element-wise when m grows
        assert (adj_loose >= adj_tight - 1e-12).all()

    def test_smaller_n_total_raises(self):
        """n_total < len(p) is incoherent — BHY assumes submitted is a
        subset of the full family."""
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        with pytest.raises(ValueError, match="n_total .* must be >="):
            bhy_adjust(p, n_total=2)
        with pytest.raises(ValueError, match="n_total .* must be >="):
            bhy_adjusted_p(p, n_total=2)

    def test_adjusted_p_monotone_in_n_total(self):
        """Adjusted p-values are non-decreasing as n_total grows."""
        p = np.array([0.001, 0.01, 0.03, 0.08, 0.20])
        prev = bhy_adjusted_p(p, n_total=len(p))
        for n in [10, 100, 1000]:
            curr = bhy_adjusted_p(p, n_total=n)
            # Allow tiny float noise
            assert (curr + 1e-12 >= prev).all()
            prev = curr
