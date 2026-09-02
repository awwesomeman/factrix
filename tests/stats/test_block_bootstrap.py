"""Tests for ``factrix._stats.bootstrap`` (Politis-Romano / Politis-White)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._stats.bootstrap import (
    _block_bootstrap_diff_p,
    _count_extreme,
    _politis_white_block_length,
    _stationary_block_indices,
)


class TestPolitisWhiteBlockLength:
    def test_iid_returns_short_block(self):
        # IID series: optimal block length should be small (< T^(1/3) * something).
        rng = np.random.default_rng(seed=0)
        x = rng.standard_normal(size=500)
        L = _politis_white_block_length(x)
        assert 1.0 <= L <= 50.0  # generous upper bound for IID

    def test_persistent_returns_longer_block(self):
        # AR(1) with phi=0.7 should pick a block longer than IID equivalent.
        rng = np.random.default_rng(seed=1)
        n = 500
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + rng.standard_normal()
        L_persist = _politis_white_block_length(x)
        # Generate matched-length IID and compare.
        x_iid = rng.standard_normal(size=n)
        L_iid = _politis_white_block_length(x_iid)
        assert L_persist > L_iid

    def test_fallback_on_short_series(self):
        # n=3 < 4 → fallback to 1.75 * n^(1/3).
        L = _politis_white_block_length(np.array([1.0, 2.0, 3.0]))
        assert pytest.approx(max(1.0, 1.75 * 3 ** (1.0 / 3.0))) == L

    def test_fallback_on_zero_variance(self):
        L = _politis_white_block_length(np.zeros(100))
        assert pytest.approx(max(1.0, 1.75 * 100 ** (1.0 / 3.0))) == L


class TestPolitisWhiteUpperBound:
    """The upper clamp is arch's ``b_max = ceil(min(3·√n, n/3))``.

    Verified against ``arch.bootstrap._single_optimal_block`` source (line
    ``b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))``), replacing an
    earlier looser ``n / 2``. The bound exists to keep enough effective
    blocks per resample: at ``L = n/2`` a resample is ~2 blocks and the
    empirical p is built on coin flips.

    It binds on short series regardless of persistence — measured
    against the old ``n / 2``, the resolved L moves on ~6% of iid draws
    at n=20 and under 1% by n=120 — so these are point checks on
    representative seeds, not a claim that any family is universally
    untouched.
    """

    def test_bound_binds_on_short_trending_series(self):
        # A short random walk pushes the plug-in L past the bound; at n=20
        # the old n/2 bound allowed L=10 (2 blocks per resample), the arch
        # bound caps it at ceil(min(3*sqrt(20), 20/3)) = 7. Rng chosen so
        # the unclamped estimate genuinely exceeds the bound.
        n = 20
        x = np.cumsum(np.random.default_rng(156).standard_normal(n))
        L = _politis_white_block_length(x)
        assert pytest.approx(np.ceil(min(3 * np.sqrt(n), n / 3))) == L
        assert n / 2 > L

    def test_iid_series_is_untouched_by_the_bound(self):
        # A long iid series resolves to the lower clamp region far below
        # the bound. This is a point check, not a general claim: at short
        # n the bound does bind on iid draws (~6% at n=20), because the
        # plug-in estimate is noisiest where there is least dependence.
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
        with pytest.raises(ValueError, match="invalid block_length"):
            _stationary_block_indices(10, 5, mean_block_length=0.5, rng=rng)

    def test_rejects_block_length_past_the_bound(self):
        rng = np.random.default_rng()
        with pytest.raises(ValueError, match="invalid block_length"):
            _stationary_block_indices(30, 5, mean_block_length=200.0, rng=rng)


class TestTailCount:
    @pytest.mark.parametrize(
        ("alternative", "expected"),
        [("greater", 2), ("less", 2), ("two-sided", 2)],
    )
    def test_ties_count_as_extreme(self, alternative, expected):
        draws = np.array([1.0, 2.0, 3.0])
        assert _count_extreme(draws, 2.0, alternative) == expected


class TestStudentizedDiffP:
    def test_calibration_under_null(self):
        # Under H0 (true mean = 0), p should be roughly uniform — in
        # particular, large p on a series with mean ≈ 0.
        rng = np.random.default_rng(seed=0)
        diff = rng.standard_normal(size=200)  # mean ≈ 0
        p, _meta = _block_bootstrap_diff_p(diff, n_resamples=499, rng=0)
        assert 0.0 < p <= 1.0
        # mean ≈ 0 → not significant.
        assert p > 0.1

    def test_power_under_strong_alt(self):
        rng = np.random.default_rng(seed=1)
        diff = rng.standard_normal(size=200) + 0.5  # strong positive shift
        p, _ = _block_bootstrap_diff_p(diff, n_resamples=499, rng=0)
        assert p < 0.01

    def test_seed_recorded_when_none(self):
        diff = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.05, 0.15])
        _p, meta = _block_bootstrap_diff_p(diff, n_resamples=199, rng=None)
        assert isinstance(meta["seed"], int)
        assert meta["seed"] >= 0
        assert meta["n_resamples"] == 199
        assert 0 < meta["n_resamples_used"] <= 199

    def test_explicit_seed_reproducible(self):
        diff = np.array([0.3, -0.1, 0.4, -0.2, 0.1, 0.05, -0.15, 0.2, 0.0, 0.1])
        p1, m1 = _block_bootstrap_diff_p(diff, n_resamples=199, rng=123)
        p2, m2 = _block_bootstrap_diff_p(diff, n_resamples=199, rng=123)
        assert p1 == p2
        assert m1["seed"] == m2["seed"] == 123

    def test_metadata_counts_only_resamples_used_by_p(self):
        """A normal finite sample can drop roots; metadata exposes that denominator."""
        diff = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.05, 0.15])
        p_value, metadata = _block_bootstrap_diff_p(diff, n_resamples=199, rng=0)

        assert metadata["n_resamples"] == 199
        n_used = metadata["n_resamples_used"]
        assert 0 < n_used < 199
        smoothed_extreme = p_value * (n_used + 1) - 1
        assert smoothed_extreme == pytest.approx(round(smoothed_extreme))
        assert metadata["p_value_mc_se"] == pytest.approx(
            np.sqrt(p_value * (1.0 - p_value) / n_used)
        )

    def test_zero_usable_resamples_withholds_p(self, monkeypatch):
        """Zero valid bootstrap-t roots admit no inference, not p=1."""
        import factrix._stats.bootstrap as bootstrap

        def fake_batch_means_se(values, _block_length):
            if values.ndim == 1:
                return np.array([1.0])
            return np.full(values.shape[0], np.nan)

        monkeypatch.setattr(bootstrap, "_batch_means_se", fake_batch_means_se)
        p_value, metadata = _block_bootstrap_diff_p(
            np.arange(10.0), n_resamples=3, rng=0
        )

        assert math.isnan(p_value)
        assert metadata["n_resamples"] == 3
        assert metadata["n_resamples_used"] == 0
        assert math.isnan(metadata["p_value_mc_se"])

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

        expected = _politis_white_block_length(x)
        assert expected != round(expected), "fixture must exercise a fractional L"

        _p, meta = _block_bootstrap_diff_p(x, n_resamples=199, rng=0)
        assert meta["block_length"] == pytest.approx(expected)

    def test_stationary_auto_matches_period_inference(self):
        """Both auto paths resolve the same series to the same block length.

        ``slicing.period_inference`` passes the Politis-White float straight
        to ``_stationary_block_indices``; before the fix this function
        rounded it first, so the two re-sampled the same series with
        different effective block lengths.
        """
        rng = np.random.default_rng(seed=11)
        x = np.empty(300)
        x[0] = rng.standard_normal()
        for t in range(1, 300):
            x[t] = 0.6 * x[t - 1] + rng.standard_normal()

        _p, meta = _block_bootstrap_diff_p(x, n_resamples=199, rng=0)
        assert meta["block_length"] == pytest.approx(_politis_white_block_length(x))

    def test_short_series_withholds_the_test(self):
        """n < 2 admits no test. NaN, not 1.0: an untestable sample is not
        evidence for the null, and the old 0.0 block_length sentinel broke
        the documented ``>= 1`` invariant for metadata readers."""
        p, meta = _block_bootstrap_diff_p(np.array([0.5]))
        assert math.isnan(p)
        assert math.isnan(meta["block_length"])
        assert meta["n_resamples"] == 0
        assert meta["n_resamples_used"] == 0
        assert meta["studentized"] is False

    def test_short_series_still_rejects_unknown_alternative(self):
        with pytest.raises(UserInputError, match="greater"):
            _block_bootstrap_diff_p(
                np.array([0.5]),
                alternative="grater",  # type: ignore[arg-type]
            )

    def test_p_floor_smoothing(self):
        # p should never be exactly 0 (Davison-Hinkley smoothing).
        diff = np.full(100, 100.0)  # huge mean → all bootstrap means 0
        p, _ = _block_bootstrap_diff_p(diff, n_resamples=99, rng=0)
        assert p == pytest.approx(1.0 / 100.0)


class TestRejectsNonFinite:
    def test_diff_p_rejects_nan(self):
        """An all-NaN centring makes every ``|boot| >= |obs|`` test False and
        the empirical p collapse to ``1 / (B + 1)`` — spurious maximal
        significance — so a NaN must be refused, not tolerated."""
        diff = np.array([0.1, 0.2, float("nan"), 0.3, 0.1, 0.2])
        with pytest.raises(ValueError, match="finite"):
            _block_bootstrap_diff_p(diff, rng=0)

    def test_politis_white_falls_back_on_nan(self):
        x = np.array([0.1, float("nan"), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert _politis_white_block_length(x) == pytest.approx(1.75 * 8 ** (1 / 3))


class TestStudentizedRoot:
    """Götze-Künsch: the block bootstrap refines only for a studentized root."""

    @staticmethod
    def _ar1(n, phi, rng):
        e = rng.normal(size=n + 200)
        x = np.zeros(n + 200)
        for t in range(1, n + 200):
            x[t] = phi * x[t - 1] + e[t]
        return x[200:]

    def test_root_is_studentized_by_the_block_se(self):
        """Hand-check: p counts |t*| >= |t_obs| with the batch-means SE."""
        from factrix._stats.bootstrap import (
            _batch_means_se,
            _politis_white_block_length,
            _stationary_block_indices,
        )

        rng = np.random.default_rng(0)
        diff = rng.standard_normal(120)
        p, meta = _block_bootstrap_diff_p(diff, n_resamples=199, rng=7)
        assert meta["studentized"] is True

        L = _politis_white_block_length(diff)
        assert meta["block_length"] == pytest.approx(L)
        idx = _stationary_block_indices(120, 199, L, np.random.default_rng(7))
        resamples = (diff - diff.mean())[idx]
        se_boot = _batch_means_se(resamples, L)
        usable = np.isfinite(se_boot) & (se_boot > 0)
        t_boot = resamples.mean(axis=1)[usable] / se_boot[usable]
        t_obs = diff.mean() / _batch_means_se(diff, L)[0]
        expected = (np.sum(np.abs(t_boot) >= abs(t_obs)) + 1.0) / (usable.sum() + 1.0)
        assert p == pytest.approx(expected)

    def test_batch_means_se_matches_a_hand_computation(self):
        from factrix._stats.bootstrap import _batch_means_se

        x = np.arange(12.0)
        # L=4 -> batches [0..3], [4..7], [8..11] with means 1.5, 5.5, 9.5.
        batch_means = np.array([1.5, 5.5, 9.5])
        assert _batch_means_se(x, 4)[0] == pytest.approx(
            np.sqrt(batch_means.var(ddof=1) / 3)
        )

    def test_block_bound_always_leaves_enough_batches_to_studentize(self):
        """The L <= ceil(min(3*sqrt(n), n/3)) bound guarantees >= 2 batches,
        so the unstudentized fallback is unreachable from admissible input."""
        from factrix._stats.bootstrap import _max_block_length

        for n in range(2, 400):
            assert n // _max_block_length(n) >= 2

        rng = np.random.default_rng(2)
        for n in (6, 20, 120):
            _, meta = _block_bootstrap_diff_p(
                rng.standard_normal(n), n_resamples=99, rng=0
            )
            assert meta["studentized"] is True

    def test_degenerate_series_falls_back_to_the_raw_mean_root(self):
        """A zero-dispersion series leaves no SE to divide by."""
        _, meta = _block_bootstrap_diff_p(np.full(40, 2.0), n_resamples=99, rng=0)
        assert meta["studentized"] is False

    @pytest.mark.parametrize(
        ("alternative", "expected_p"),
        [("greater", 0.01), ("less", 1.0), ("two-sided", 0.01)],
    )
    def test_raw_mean_fallback_respects_alternative(self, alternative, expected_p):
        p, meta = _block_bootstrap_diff_p(
            np.full(40, 2.0),
            n_resamples=99,
            alternative=alternative,
            rng=0,
        )
        assert meta["studentized"] is False
        assert p == pytest.approx(expected_p)

    @pytest.mark.parametrize(("n", "phi"), [(120, 0.5), (500, 0.8)])
    def test_size_beats_the_unstudentized_root(self, n, phi):
        """Studentizing roughly halves the excess rejection under dependence."""
        from factrix._stats.bootstrap import (
            _politis_white_block_length,
            _stationary_block_indices,
        )

        n_reps = 200
        rng = np.random.default_rng(11)
        studentized = plain = 0
        for _ in range(n_reps):
            diff = self._ar1(n, phi, rng)
            p, _ = _block_bootstrap_diff_p(diff, n_resamples=299, rng=1)
            studentized += p < 0.05
            # Same draws, unstudentized root (the pre-fix behaviour).
            L = _politis_white_block_length(diff)
            idx = _stationary_block_indices(n, 299, L, np.random.default_rng(1))
            boot = (diff - diff.mean())[idx].mean(axis=1)
            p_plain = (np.sum(np.abs(boot) >= abs(diff.mean())) + 1.0) / 300.0
            plain += p_plain < 0.05
        assert studentized <= plain
        assert studentized / n_reps <= 0.13


class TestOverlapHorizonFloor:
    def test_block_length_is_floored_at_the_horizon(self):
        rng = np.random.default_rng(3)
        diff = rng.standard_normal(200)
        _, meta = _block_bootstrap_diff_p(
            diff, overlap_periods=21, n_resamples=99, rng=0
        )
        assert meta["block_length"] >= 21

    def test_floor_respects_the_max_block_bound(self):
        """h beyond ceil(min(3*sqrt(n), n/3)) clamps rather than raising."""
        from factrix._stats.bootstrap import _max_block_length

        rng = np.random.default_rng(3)
        diff = rng.standard_normal(30)
        _, meta = _block_bootstrap_diff_p(
            diff, overlap_periods=100, n_resamples=99, rng=0
        )
        assert meta["block_length"] == _max_block_length(30)

    def test_horizon_one_leaves_the_plug_in_alone(self):
        rng = np.random.default_rng(3)
        diff = rng.standard_normal(200)
        _, floored = _block_bootstrap_diff_p(
            diff, overlap_periods=1, n_resamples=99, rng=0
        )
        _, plain = _block_bootstrap_diff_p(diff, n_resamples=99, rng=0)
        assert floored["block_length"] == plain["block_length"]


class TestPolitisWhiteCorrections:
    def test_k_t_window_is_wide_enough_to_see_a_lag_five_spike(self):
        """K_T = max(5, ceil(sqrt(log10 n))) per Politis-White section 4.

        Behavioural pin rather than a source-text one: on an MA(5) series
        whose only significant autocorrelation sits at lag 5, the lag
        selector's first window (lags 1..K_T) must reach that spike, so m
        advances past it and the plug-in returns a long block. A narrower
        window would stop at m = 0, see only the (near-zero) lag-1 term and
        return the L ~ 1 of an independent series.
        """
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            e = rng.standard_normal(505)
            ma5 = e[5:] + 0.9 * e[:-5]
            assert _politis_white_block_length(ma5) > 5.0
            assert _politis_white_block_length(rng.standard_normal(500)) < 5.0

    def test_no_detectable_dependence_falls_through_to_the_unit_clamp(self):
        """Ghat ~ 0 must give L = 1, not the inflated 1.75*T^(1/3) rule.

        A white-noise draw whose sample lag-1 autocorrelation is ~0 drives
        the estimated spectral derivative to ~0, so the plug-in L underflows
        and has to land on the ``max(L, 1.0)`` clamp. The contradictory
        early return this replaced sent the same case to the generic rule
        (~10 at n = 200), an order of magnitude too long for a series with
        no dependence at all.
        """
        x = np.random.default_rng(166).standard_normal(200)
        centred = x - x.mean()
        rho_1 = float(np.dot(centred[1:], centred[:-1]) / np.dot(centred, centred))
        assert abs(rho_1) < 2e-3  # the regime the branch exists for
        assert _politis_white_block_length(x) == 1.0
        assert 1.75 * 200 ** (1 / 3) > 10.0  # what the removed branch returned

    def test_auto_block_length_respects_the_documented_clamp(self):
        from factrix._stats.bootstrap import (
            _max_block_length,
            _politis_white_block_length,
        )

        rng = np.random.default_rng(0)
        for n in (50, 120, 500):
            x = np.cumsum(rng.standard_normal(n)) * 0.01
            assert 1.0 <= _politis_white_block_length(x) <= _max_block_length(n)
        assert _max_block_length(50) == 17
        assert _max_block_length(120) == 33
        assert _max_block_length(500) == 68
