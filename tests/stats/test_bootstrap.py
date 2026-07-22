"""Stationary bootstrap resampling + mean CI."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.bootstrap import _politis_white_block_length
from factrix.stats.bootstrap import (
    _resolve_auto_block_length,
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)


class TestResolveAutoBlockLength:
    def test_vector_matches_politis_white(self):
        x = np.random.default_rng(0).standard_normal(200)
        assert _resolve_auto_block_length(x) == _politis_white_block_length(
            x, scheme="stationary"
        )

    def test_matrix_takes_max_of_per_column_estimates(self):
        rng = np.random.default_rng(0)
        persistent = np.empty(300)
        persistent[0] = rng.standard_normal()
        for t in range(1, 300):
            persistent[t] = 0.9 * persistent[t - 1] + rng.standard_normal()
        iid = rng.standard_normal(300)
        values = np.column_stack([iid, persistent])

        expected = max(
            _politis_white_block_length(iid, scheme="stationary"),
            _politis_white_block_length(persistent, scheme="stationary"),
        )
        assert _resolve_auto_block_length(values) == expected

    def test_zero_column_matrix_falls_back_like_degenerate_series(self):
        assert _resolve_auto_block_length(
            np.empty((10, 0))
        ) == _politis_white_block_length(np.zeros(10), scheme="stationary")


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

    def test_matrix_columns_share_resampled_rows(self):
        base = np.arange(40, dtype=float)
        values = np.column_stack([base, 10.0 * base + 3.0])
        resamples = stationary_bootstrap_resamples(
            values,
            n_bootstrap=20,
            seed=42,
        )
        assert resamples.shape == (20, 40, 2)
        np.testing.assert_array_equal(
            resamples[:, :, 1], 10.0 * resamples[:, :, 0] + 3.0
        )

    def test_matrix_first_column_matches_vector_with_same_seed(self):
        base = np.arange(40, dtype=float)
        values = np.column_stack([base, -base])
        vector = stationary_bootstrap_resamples(base, n_bootstrap=20, seed=42)
        matrix = stationary_bootstrap_resamples(values, n_bootstrap=20, seed=42)
        np.testing.assert_array_equal(matrix[:, :, 0], vector)

    def test_empty_matrix_preserves_column_axis(self):
        resamples = stationary_bootstrap_resamples(
            np.empty((0, 3)), n_bootstrap=5, seed=0
        )
        assert resamples.shape == (5, 0, 3)

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

    def test_rejects_unsupported_shape(self):
        with pytest.raises(ValueError, match=r"shape \(T,\) or \(T, m\)"):
            stationary_bootstrap_resamples(np.ones((2, 3, 4)), n_bootstrap=5)

    @pytest.mark.parametrize("n_bootstrap", [0, -1, True, 2.5])
    def test_rejects_invalid_resample_count(self, n_bootstrap):
        with pytest.raises(ValueError, match="positive integer"):
            stationary_bootstrap_resamples(np.arange(10.0), n_bootstrap=n_bootstrap)

    def test_rejects_non_finite_values(self):
        with pytest.raises(ValueError, match="finite"):
            stationary_bootstrap_resamples(np.array([[1.0, np.nan]]), n_bootstrap=5)


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

    def test_rejects_matrix_input(self):
        with pytest.raises(ValueError, match="values must be 1-D"):
            bootstrap_mean_ci(np.ones((20, 2)))
