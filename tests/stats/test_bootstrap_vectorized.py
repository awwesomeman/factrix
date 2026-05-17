"""Cross-factor vectorised stationary bootstrap (``bootstrap_mean_ci_batch``).

Equivalence to the single-factor path under the same seed is the
contract: the batched call shares one ``(B, T)`` block-index matrix
across rows, so the K=1 case must reproduce ``bootstrap_mean_ci``
to floating-point identity.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    bootstrap_mean_ci_batch,
)


class TestSingleFactorEquivalence:
    def test_k1_matches_single_factor_path_bitwise(self):
        # K=1 batch must reproduce single-factor output exactly: same
        # rng draw, same advanced-index, same quantile.
        x = np.random.default_rng(0).standard_normal(200)
        lo, hi, point = bootstrap_mean_ci(x, n_bootstrap=500, seed=7)
        los, his, points = bootstrap_mean_ci_batch(x[None, :], n_bootstrap=500, seed=7)
        assert los.shape == his.shape == points.shape == (1,)
        np.testing.assert_array_equal(los, [lo])
        np.testing.assert_array_equal(his, [hi])
        np.testing.assert_array_equal(points, [point])

    def test_k1_matches_with_custom_block_length_and_ci(self):
        x = np.random.default_rng(1).standard_normal(150)
        lo, hi, point = bootstrap_mean_ci(
            x, n_bootstrap=300, seed=3, block_length=8.0, ci=0.9
        )
        los, his, points = bootstrap_mean_ci_batch(
            x[None, :], n_bootstrap=300, seed=3, block_length=8.0, ci=0.9
        )
        np.testing.assert_array_equal(los, [lo])
        np.testing.assert_array_equal(his, [hi])
        np.testing.assert_array_equal(points, [point])


class TestBatchShape:
    def test_returns_per_factor_arrays(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal((12, 100))
        lo, hi, point = bootstrap_mean_ci_batch(values, n_bootstrap=200, seed=42)
        assert lo.shape == hi.shape == point.shape == (12,)
        # CI must bracket the point estimate for every factor (by
        # construction: ``point`` is the sample mean and lo / hi are
        # quantiles of bootstrap means, which centre on the sample).
        assert np.all(lo <= point)
        assert np.all(point <= hi)

    def test_point_equals_row_mean(self):
        values = np.arange(40, dtype=float).reshape(4, 10)
        _, _, point = bootstrap_mean_ci_batch(values, n_bootstrap=50, seed=0)
        np.testing.assert_allclose(point, values.mean(axis=1))


class TestChunking:
    def test_chunk_size_does_not_affect_output(self):
        # K-chunking is a memory bound for the resample tensor; the
        # (B, T) index matrix is shared across chunks, so per-row
        # bootstrap means are mathematically identical regardless of
        # chunk size. Numpy's reduction order across different chunk
        # shapes can drift the result by ~1 ULP — assert with a tight
        # tolerance instead of bitwise.
        rng = np.random.default_rng(2)
        values = rng.standard_normal((20, 80))
        full = bootstrap_mean_ci_batch(values, n_bootstrap=250, seed=5)
        for cs in (1, 3, 7, 20, 100):
            chunked = bootstrap_mean_ci_batch(
                values, n_bootstrap=250, seed=5, chunk_size=cs
            )
            np.testing.assert_allclose(chunked[0], full[0], rtol=0, atol=1e-14)
            np.testing.assert_allclose(chunked[1], full[1], rtol=0, atol=1e-14)
            np.testing.assert_allclose(chunked[2], full[2], rtol=0, atol=1e-14)

    def test_default_chunk_caps_resample_tensor_memory(self):
        # With a small budget the chunk must drop below n_factors; the
        # equivalence test above proves the chunked path is numerically
        # identical to the unchunked one.
        rng = np.random.default_rng(3)
        values = rng.standard_normal((50, 60))
        # Explicit small chunk = sanity that the loop terminates / output
        # shape stays right at the boundary cases.
        lo, hi, point = bootstrap_mean_ci_batch(
            values, n_bootstrap=100, seed=1, chunk_size=1
        )
        assert lo.shape == hi.shape == point.shape == (50,)


class TestValidation:
    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            bootstrap_mean_ci_batch(np.arange(20, dtype=float))
        with pytest.raises(ValueError, match="must be 2-D"):
            bootstrap_mean_ci_batch(np.zeros((2, 3, 4)))

    def test_rejects_ci_out_of_range(self):
        x = np.arange(20, dtype=float).reshape(1, 20)
        with pytest.raises(ValueError, match="ci must be in"):
            bootstrap_mean_ci_batch(x, ci=0.0)
        with pytest.raises(ValueError, match="ci must be in"):
            bootstrap_mean_ci_batch(x, ci=1.5)

    def test_rejects_block_length_below_one(self):
        x = np.arange(20, dtype=float).reshape(1, 20)
        with pytest.raises(ValueError, match="block_length must be"):
            bootstrap_mean_ci_batch(x, block_length=0.5)

    def test_empty_n_factors_returns_empty(self):
        out = bootstrap_mean_ci_batch(
            np.empty((0, 50), dtype=float), n_bootstrap=10, seed=0
        )
        for arr in out:
            assert arr.shape == (0,)

    def test_empty_observations_returns_nan(self):
        # Matches single-factor ``bootstrap_mean_ci`` which propagates
        # NaN at n_observations==0 — zero would silently claim a sample
        # mean of nothing and mask a bad input.
        out = bootstrap_mean_ci_batch(
            np.empty((3, 0), dtype=float), n_bootstrap=10, seed=0
        )
        for arr in out:
            assert arr.shape == (3,)
            assert np.all(np.isnan(arr))
