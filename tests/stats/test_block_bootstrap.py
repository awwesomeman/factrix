"""Tests for ``factrix._stats.bootstrap`` (Künsch / PR / PW)."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.bootstrap import (
    _block_bootstrap_diff_p,
    _fixed_block_indices,
    _politis_white_block_length,
    _stationary_block_indices,
)


class TestPolitisWhiteBlockLength:
    def test_iid_returns_short_block(self):
        # IID series: optimal block length should be small (< T^(1/3) * something).
        rng = np.random.default_rng(seed=0)
        x = rng.standard_normal(size=500)
        L = _politis_white_block_length(x, scheme="stationary")
        assert 1.0 <= L <= 50.0  # generous upper bound for IID

    def test_persistent_returns_longer_block(self):
        # AR(1) with phi=0.7 should pick a block longer than IID equivalent.
        rng = np.random.default_rng(seed=1)
        n = 500
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + rng.standard_normal()
        L_persist = _politis_white_block_length(x, scheme="stationary")
        # Generate matched-length IID and compare.
        x_iid = rng.standard_normal(size=n)
        L_iid = _politis_white_block_length(x_iid, scheme="stationary")
        assert L_persist > L_iid

    def test_fixed_smaller_than_stationary(self):
        # PW eq 9 vs 12: D_CB = (4/3)·g(0)² > D_SB = 2·g(0)²? No — 4/3 < 2,
        # so fixed L = (2G²/D_CB)^(1/3) > stationary L. Verify direction.
        rng = np.random.default_rng(seed=2)
        n = 400
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        L_sb = _politis_white_block_length(x, scheme="stationary")
        L_cb = _politis_white_block_length(x, scheme="fixed")
        assert L_cb >= L_sb

    def test_fallback_on_short_series(self):
        # n=3 < 4 → fallback to 1.75 * n^(1/3).
        L = _politis_white_block_length(np.array([1.0, 2.0, 3.0]))
        assert pytest.approx(max(1.0, 1.75 * 3 ** (1.0 / 3.0))) == L

    def test_fallback_on_zero_variance(self):
        L = _politis_white_block_length(np.zeros(100))
        assert pytest.approx(max(1.0, 1.75 * 100 ** (1.0 / 3.0))) == L

    def test_rejects_unknown_scheme(self):
        rng = np.random.default_rng(seed=7)
        with pytest.raises(ValueError, match="scheme must be"):
            _politis_white_block_length(rng.standard_normal(200), scheme="other")  # type: ignore[arg-type]


class TestPolitisWhiteUpperBound:
    """The upper clamp is arch's ``b_max = ceil(min(3·√n, n/3))``.

    Verified against ``arch.bootstrap._single_optimal_block`` source (line
    ``b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))``), replacing an
    earlier looser ``n / 2``. The bound exists to keep enough effective
    blocks per resample: at ``L = n/2`` a resample is ~2 blocks and the
    empirical p is built on coin flips.
    """

    def test_bound_binds_on_short_trending_series(self):
        # A short random walk pushes the plug-in L past the bound; at n=20
        # the old n/2 bound allowed L=10 (2 blocks per resample), the arch
        # bound caps it at ceil(min(3*sqrt(20), 20/3)) = 7. Seed chosen so
        # the unclamped estimate genuinely exceeds the bound.
        n = 20
        x = np.cumsum(np.random.default_rng(156).standard_normal(n))
        L = _politis_white_block_length(x)
        assert pytest.approx(np.ceil(min(3 * np.sqrt(n), n / 3))) == L
        assert n / 2 > L

    def test_iid_series_is_untouched_by_the_bound(self):
        # The bound only matters for extreme persistence; iid resolves to
        # the lower clamp region far below it.
        x = np.random.default_rng(3).standard_normal(120)
        assert _politis_white_block_length(x) < 5.0

    def test_moderately_persistent_series_is_untouched(self):
        # AR(0.6) at n=300 sits well inside both the old and new bounds:
        # the alignment must not move the common case.
        rng = np.random.default_rng(7)
        x = np.empty(300)
        x[0] = rng.standard_normal()
        for t in range(1, 300):
            x[t] = 0.6 * x[t - 1] + rng.standard_normal()
        L = _politis_white_block_length(x)
        assert 1.0 < L < np.ceil(min(3 * np.sqrt(300), 100))


class TestFixedBlockIndices:
    def test_shape_and_range(self):
        rng = np.random.default_rng(seed=0)
        idx = _fixed_block_indices(100, 50, block_length=5, rng=rng)
        assert idx.shape == (50, 100)
        assert idx.min() >= 0 and idx.max() < 100

    def test_block_contiguity(self):
        # Within each block of length L, indices should be consecutive
        # modulo n. Sample one resample.
        rng = np.random.default_rng(seed=42)
        n, L = 50, 4
        idx = _fixed_block_indices(n, 1, block_length=L, rng=rng)[0]
        # Each block of length L (except maybe last) should have
        # idx[t+1] = (idx[t] + 1) % n.
        n_full_blocks = len(idx) // L
        for b in range(n_full_blocks):
            block = idx[b * L : (b + 1) * L]
            diffs = (block[1:] - block[:-1]) % n
            assert np.all(diffs == 1)

    def test_rejects_zero_block_length(self):
        rng = np.random.default_rng()
        with pytest.raises(ValueError, match="block_length must be >= 1"):
            _fixed_block_indices(10, 5, block_length=0, rng=rng)

    def test_empty(self):
        rng = np.random.default_rng()
        idx = _fixed_block_indices(0, 5, block_length=3, rng=rng)
        assert idx.shape == (5, 0)


class TestStationaryBlockIndices:
    def test_shape_and_range(self):
        rng = np.random.default_rng(seed=0)
        idx = _stationary_block_indices(100, 50, mean_block_length=5.0, rng=rng)
        assert idx.shape == (50, 100)
        assert idx.min() >= 0 and idx.max() < 100

    def test_geometric_block_length_mean(self):
        # With p_new = 1/L, the count of new-block events per resample
        # of length n should average n/L. Statistical sanity (loose).
        rng = np.random.default_rng(seed=99)
        n = 1000
        L_target = 10.0
        idx = _stationary_block_indices(n, 200, mean_block_length=L_target, rng=rng)
        # Block boundary count = number of jumps that aren't "+1 mod n".
        diffs = (idx[:, 1:] - idx[:, :-1]) % n
        n_jumps = (diffs != 1).sum() + 200  # +1 boundary at t=0 per resample
        avg_blocks = n_jumps / 200
        # Expected ≈ n/L = 100.
        assert 70 < avg_blocks < 130

    def test_rejects_short_block(self):
        rng = np.random.default_rng()
        with pytest.raises(ValueError, match=r"mean_block_length must be >= 1\.0"):
            _stationary_block_indices(10, 5, mean_block_length=0.5, rng=rng)


class TestBlockBootstrapDiffP:
    def test_calibration_under_null(self):
        # Under H0 (true mean = 0), p should be roughly uniform — in
        # particular, large p on a series with mean ≈ 0.
        rng = np.random.default_rng(seed=0)
        diff = rng.standard_normal(size=200)  # mean ≈ 0
        p, _meta = _block_bootstrap_diff_p(diff, n_resamples=499, rng_seed=0)
        assert 0.0 < p <= 1.0
        # mean ≈ 0 → not significant.
        assert p > 0.1

    def test_power_under_strong_alt(self):
        rng = np.random.default_rng(seed=1)
        diff = rng.standard_normal(size=200) + 0.5  # strong positive shift
        p, _ = _block_bootstrap_diff_p(diff, n_resamples=499, rng_seed=0)
        assert p < 0.01

    def test_seed_recorded_when_none(self):
        diff = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.05, 0.15])
        _p, meta = _block_bootstrap_diff_p(diff, n_resamples=199, rng_seed=None)
        assert isinstance(meta["rng_seed"], int)
        assert meta["rng_seed"] >= 0
        assert meta["n_resamples"] == 199

    def test_explicit_seed_reproducible(self):
        diff = np.array([0.3, -0.1, 0.4, -0.2, 0.1, 0.05, -0.15, 0.2, 0.0, 0.1])
        p1, m1 = _block_bootstrap_diff_p(diff, n_resamples=199, rng_seed=123)
        p2, m2 = _block_bootstrap_diff_p(diff, n_resamples=199, rng_seed=123)
        assert p1 == p2
        assert m1["rng_seed"] == m2["rng_seed"] == 123

    def test_fixed_scheme(self):
        rng = np.random.default_rng(seed=2)
        diff = rng.standard_normal(size=100) + 0.4
        p_fixed, m_fixed = _block_bootstrap_diff_p(
            diff,
            block_length=5,
            scheme="fixed",
            n_resamples=499,
            rng_seed=0,
        )
        assert p_fixed < 0.05
        assert m_fixed["scheme"] == "fixed"
        assert m_fixed["block_length"] == 5

    def test_stationary_auto_block_length_is_not_discretized(self):
        """The stationary L is a geometric MEAN, so it must stay fractional.

        Rounding it discretizes the renewal probability ``p_new = 1 / L``
        for no reason, and put this function out of step with the two other
        consumers of the same estimate.
        """
        rng = np.random.default_rng(seed=7)
        # AR(1): persistent enough that Politis-White lands off an integer.
        x = np.empty(300)
        x[0] = rng.standard_normal()
        for t in range(1, 300):
            x[t] = 0.6 * x[t - 1] + rng.standard_normal()

        expected = _politis_white_block_length(x, scheme="stationary")
        assert expected != round(expected), "fixture must exercise a fractional L"

        _p, meta = _block_bootstrap_diff_p(x, n_resamples=199, rng_seed=0)
        assert meta["block_length"] == pytest.approx(expected)

    def test_stationary_auto_matches_period_inference(self):
        """Both auto paths resolve the same series to the same block length.

        ``slicing.period_inference`` passes the Politis-White float straight
        to ``_stationary_block_indices``; before the fix this function
        rounded it first, so the two re-sampled the same series with
        different effective block lengths.
        """
        from factrix.slicing.period_inference import _SCHEME

        rng = np.random.default_rng(seed=11)
        x = np.empty(300)
        x[0] = rng.standard_normal()
        for t in range(1, 300):
            x[t] = 0.6 * x[t - 1] + rng.standard_normal()

        _p, meta = _block_bootstrap_diff_p(
            x, n_resamples=199, rng_seed=0, scheme=_SCHEME
        )
        assert meta["block_length"] == pytest.approx(
            _politis_white_block_length(x, scheme=_SCHEME)
        )

    def test_fixed_auto_block_length_stays_integral(self):
        """The fixed scheme's L IS a block size, so rounding is correct there."""
        rng = np.random.default_rng(seed=7)
        x = np.empty(300)
        x[0] = rng.standard_normal()
        for t in range(1, 300):
            x[t] = 0.6 * x[t - 1] + rng.standard_normal()

        _p, meta = _block_bootstrap_diff_p(
            x, n_resamples=199, rng_seed=0, scheme="fixed"
        )
        L = meta["block_length"]
        assert int(L) == L

    def test_short_series_returns_one(self):
        p, meta = _block_bootstrap_diff_p(np.array([0.5]))
        assert p == 1.0
        assert meta["n_resamples"] == 0

    def test_p_floor_smoothing(self):
        # p should never be exactly 0 (Davison-Hinkley smoothing).
        diff = np.full(100, 100.0)  # huge mean → all bootstrap means 0
        p, _ = _block_bootstrap_diff_p(diff, n_resamples=99, rng_seed=0)
        assert p == pytest.approx(1.0 / 100.0)


class TestRejectsNonFinite:
    def test_diff_p_rejects_nan(self):
        """An all-NaN centring makes every ``|boot| >= |obs|`` test False and
        the empirical p collapse to ``1 / (B + 1)`` — spurious maximal
        significance — so a NaN must be refused, not tolerated."""
        diff = np.array([0.1, 0.2, float("nan"), 0.3, 0.1, 0.2])
        with pytest.raises(ValueError, match="finite"):
            _block_bootstrap_diff_p(diff, rng_seed=0)

    def test_politis_white_falls_back_on_nan(self):
        x = np.array([0.1, float("nan"), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert _politis_white_block_length(x) == pytest.approx(1.75 * 8 ** (1 / 3))
