"""Stationary bootstrap resampling + mean CI."""

from __future__ import annotations

import numpy as np
import pytest
from factrix.stats.bootstrap import (
    _default_block_length,
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)


class TestDefaultBlockLength:
    def test_grows_as_t_cuberoot(self):
        # L ≈ 1.75 · T^(1/3): 1000 → ~17.5, 8000 → ~35
        assert _default_block_length(1000) == pytest.approx(17.5, rel=0.01)
        assert _default_block_length(8000) == pytest.approx(35.0, rel=0.01)

    def test_degenerate_small_n(self):
        assert _default_block_length(1) == 1.0
        assert _default_block_length(0) == 1.0


class TestStationaryBootstrapResamples:
    def test_shape_and_same_length(self):
        x = np.arange(100, dtype=float)
        resamples = stationary_bootstrap_resamples(
            x,
            n_bootstrap=50,
            seed=0,
        )
        assert resamples.shape == (50, 100)

    def test_values_are_a_subset_of_input(self):
        x = np.arange(50, dtype=float)
        resamples = stationary_bootstrap_resamples(
            x,
            n_bootstrap=20,
            seed=0,
        )
        # Every draw must come from the original series.
        assert np.isin(resamples, x).all()

    def test_reproducible_with_seed(self):
        x = np.random.default_rng(0).standard_normal(200)
        a = stationary_bootstrap_resamples(x, n_bootstrap=30, seed=42)
        b = stationary_bootstrap_resamples(x, n_bootstrap=30, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_block_length_one_is_iid_bootstrap(self):
        """L=1: resamples lose the within-block dependence of L>1."""
        rng = np.random.default_rng(0)
        # Build a clearly-autocorrelated series so the test can detect
        # block vs iid behaviour on resampled paths.
        noise = rng.standard_normal(500)
        x = np.empty(500)
        x[0] = noise[0]
        for t in range(1, 500):
            x[t] = 0.8 * x[t - 1] + noise[t]

        def mean_lag1_acf(samples: np.ndarray) -> float:
            acfs = [
                float(np.corrcoef(s[:-1], s[1:])[0, 1])
                for s in samples
                if np.std(s) > 0.0
            ]
            return float(np.mean(acfs)) if acfs else 0.0

        iid = stationary_bootstrap_resamples(
            x,
            n_bootstrap=100,
            block_length=1.0,
            seed=1,
        )
        block = stationary_bootstrap_resamples(
            x,
            n_bootstrap=100,
            block_length=30.0,
            seed=1,
        )
        # Mean preservation holds for both.
        assert np.mean(iid) == pytest.approx(np.mean(x), rel=0.05)
        # Block-preserving resamples keep more of the autocorrelation
        # than iid shuffles — the key guarantee of a stationary bootstrap.
        assert mean_lag1_acf(block) > mean_lag1_acf(iid) + 0.1

    def test_mean_preserved_on_iid_series(self):
        rng = np.random.default_rng(123)
        x = rng.standard_normal(500) + 0.2
        resamples = stationary_bootstrap_resamples(
            x,
            n_bootstrap=1000,
            seed=0,
        )
        # Grand mean across all resamples ≈ sample mean.
        assert float(resamples.mean()) == pytest.approx(x.mean(), abs=0.05)

    def test_rejects_block_length_below_one(self):
        with pytest.raises(ValueError, match="block_length"):
            stationary_bootstrap_resamples(
                np.arange(10.0),
                n_bootstrap=5,
                block_length=0.5,
            )


class TestBootstrapMeanCI:
    def test_basic_ci_brackets_sample_mean(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300) + 0.1
        lo, hi, point = bootstrap_mean_ci(x, n_bootstrap=500, seed=1)
        assert lo < point < hi
        # Narrow range — 300 iid normals give CI ≈ ±0.11.
        assert hi - lo < 0.5

    def test_statistic_callable(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        lo, hi, point = bootstrap_mean_ci(
            x,
            n_bootstrap=300,
            seed=2,
            statistic=np.median,
        )
        assert lo < point < hi
        assert point == pytest.approx(float(np.median(x)))

    def test_rejects_ci_out_of_range(self):
        with pytest.raises(ValueError, match="ci"):
            bootstrap_mean_ci(np.arange(20.0), ci=1.5)
        with pytest.raises(ValueError, match="ci"):
            bootstrap_mean_ci(np.arange(20.0), ci=0.0)
