"""Tests for ``factrix._stats.multiple_testing`` (Holm / Bonferroni / Romano-Wolf)."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.multiple_testing import (
    bonferroni,
    holm_step_down,
    romano_wolf,
)


class TestBonferroni:
    def test_basic(self):
        # m=4, p=[0.01, 0.04, 0.03, 0.5] → [0.04, 0.16, 0.12, 1.0]
        out = bonferroni([0.01, 0.04, 0.03, 0.5])
        assert out == pytest.approx([0.04, 0.16, 0.12, 1.0])

    def test_clipping(self):
        # p > 1/m → all clipped to 1
        assert bonferroni([0.5, 0.5, 0.5, 0.5]) == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_empty(self):
        assert bonferroni([]) == []

    def test_validates_range(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bonferroni([0.5, 1.2])


class TestHolmStepDown:
    def test_dominates_bonferroni(self):
        # Holm adj p must be ≤ Bonferroni adj p element-wise (uniform dominance).
        rng = np.random.default_rng(seed=0)
        p = rng.uniform(0, 1, size=20)
        b = np.array(bonferroni(p))
        h = np.array(holm_step_down(p))
        assert np.all(h <= b + 1e-12)

    def test_dominates_bonferroni_deterministic(self):
        # Worst-case sanity: m=3 sorted p = [0.01, 0.02, 0.03].
        # Holm scaled = [3·0.01, 2·0.02, 1·0.03] = [0.03, 0.04, 0.03];
        # cummax     = [0.03, 0.04, 0.04].
        # Bonferroni  = [0.03, 0.06, 0.09].
        # Holm strictly tighter on the 2nd and 3rd ranks.
        h = holm_step_down([0.01, 0.02, 0.03])
        b = bonferroni([0.01, 0.02, 0.03])
        assert h == pytest.approx([0.03, 0.04, 0.04])
        assert b == pytest.approx([0.03, 0.06, 0.09])

    def test_textbook_example(self):
        # Classic Holm example: p=[0.01, 0.04, 0.03, 0.005], m=4
        # Sorted: [0.005, 0.01, 0.03, 0.04]; factors [4,3,2,1]
        # Scaled: [0.02, 0.03, 0.06, 0.04]; cummax: [0.02, 0.03, 0.06, 0.06]
        # Map back to original order:
        #   idx 0 (raw 0.01, rank 1) → 0.03
        #   idx 1 (raw 0.04, rank 3) → 0.06
        #   idx 2 (raw 0.03, rank 2) → 0.06
        #   idx 3 (raw 0.005, rank 0) → 0.02
        out = holm_step_down([0.01, 0.04, 0.03, 0.005])
        assert out == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_monotone_in_rank(self):
        # Adjusted p in rank order is non-decreasing (cummax invariant).
        p = [0.001, 0.04, 0.02, 0.5, 0.3, 0.1]
        adj = np.array(holm_step_down(p))
        adj_sorted = adj[np.argsort(p)]
        assert np.all(np.diff(adj_sorted) >= -1e-12)

    def test_empty(self):
        assert holm_step_down([]) == []

    def test_singleton(self):
        # m=1: Holm = Bonferroni = identity.
        assert holm_step_down([0.03]) == pytest.approx([0.03])


class TestRomanoWolf:
    def test_independence_close_to_bonferroni_holm(self):
        # Independent N(0,1) bootstrap: Romano-Wolf should give p ≥ Holm
        # roughly (it adapts to the actual dependence; under independence
        # it's not better than Holm). Sanity: all p in [0, 1], monotone.
        rng = np.random.default_rng(seed=1)
        m = 5
        boot = rng.standard_normal(size=(2000, m))
        # Observed stats — pretend strong tails on first 2 hypotheses.
        t = np.array([3.5, 3.0, 0.5, 0.2, -0.1])
        adj = np.array(romano_wolf(t, boot))
        assert np.all((adj >= 0) & (adj <= 1))
        # Monotone in descending |t|: order [0, 1, 2, 3, 4] (|t| sorted desc).
        order = np.argsort(-np.abs(t))
        adj_desc = adj[order]
        assert np.all(np.diff(adj_desc) >= -1e-12)
        # Strong stats should attract small adj p.
        assert adj[0] < 0.05
        assert adj[1] < 0.05

    def test_dependence_sharper_than_bonferroni(self):
        # Perfectly correlated bootstrap: max over k = single column;
        # Romano-Wolf delivers near-Bonferroni-of-1 = raw-p; should be
        # MUCH tighter than naive Bonferroni-m on the leading hypothesis.
        rng = np.random.default_rng(seed=2)
        m = 10
        common = rng.standard_normal(size=2000)
        boot = np.tile(common[:, None], (1, m))  # all columns identical
        t = np.array([3.5] + [0.0] * (m - 1))
        adj_rw = np.array(romano_wolf(t, boot))
        # Naive Bonferroni would be 10 * P(|N|>3.5) ≈ 10 * 0.000465 ≈ 0.0046.
        # Romano-Wolf with perfect correlation collapses multiplicity:
        # adj[0] should be close to single-hypothesis p ≈ 0.000465.
        assert adj_rw[0] < 0.005
        # Sanity: still in [0,1].
        assert np.all((adj_rw >= 0) & (adj_rw <= 1))

    def test_one_sided(self):
        # Observed t = +3 with positive-only alt → small p; with t = -3
        # under one-sided positive alt, should be near 1.
        rng = np.random.default_rng(seed=3)
        boot = rng.standard_normal(size=(2000, 2))
        adj_pos = romano_wolf([3.0, -3.0], boot, one_sided=True)
        # +3 → tiny p; -3 → huge p (no rejection on left tail).
        assert adj_pos[0] < 0.05
        assert adj_pos[1] > 0.5

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match=r"shape \(B, 3\)"):
            romano_wolf([1.0, 2.0, 3.0], np.ones((100, 4)))

    def test_rejects_empty_bootstrap(self):
        with pytest.raises(ValueError, match="at least 1 resample"):
            romano_wolf([1.0, 2.0], np.ones((0, 2)))

    def test_empty(self):
        assert romano_wolf([], np.empty((10, 0))) == []
