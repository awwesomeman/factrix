"""Tests for cluster-Wald primitives in ``factrix._stats.wald``."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.wald import (
    _nw_hac_vector_mean,
    _wald_double_cluster,
    _wald_nw_cluster_means,
)


class TestNWHACVectorMean:
    def test_iid_diagonal_covariance(self):
        # IID multivariate normal: V_HAC ≈ Σ / T (no off-diagonal).
        rng = np.random.default_rng(seed=0)
        T, K = 1000, 3
        Y = rng.multivariate_normal(mean=np.zeros(K), cov=np.eye(K), size=T)
        mean, V = _nw_hac_vector_mean(Y)
        assert mean.shape == (K,)
        assert V.shape == (K, K)
        # Off-diagonals should be small (no cross-corr in DGP).
        np.testing.assert_allclose(V - np.diag(np.diag(V)), 0.0, atol=2e-3)
        # Diagonals ≈ 1/T = 0.001.
        np.testing.assert_allclose(np.diag(V), 1.0 / T, atol=3e-4)

    def test_psd_under_persistence(self):
        # AR(1) joint series: V_HAC must be PSD even with positive
        # autocorrelation pumping up the off-diagonals.
        rng = np.random.default_rng(seed=1)
        T = 500
        e = rng.standard_normal(size=(T, 2))
        Y = np.empty_like(e)
        Y[0] = e[0]
        for t in range(1, T):
            Y[t] = 0.6 * Y[t - 1] + e[t]
        _, V = _nw_hac_vector_mean(Y)
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals >= -1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(seed=2)
        Y = rng.standard_normal(size=(200, 4))
        _, V = _nw_hac_vector_mean(Y)
        np.testing.assert_allclose(V, V.T, atol=1e-14)

    def test_short_sample(self):
        Y = np.array([[1.0, 2.0]])  # T=1
        mean, V = _nw_hac_vector_mean(Y)
        np.testing.assert_array_equal(mean, [1.0, 2.0])
        np.testing.assert_array_equal(V, np.zeros((2, 2)))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _nw_hac_vector_mean(np.arange(10.0))

    def test_matches_scalar_nw_for_k1(self):
        # K=1 case: joint NW HAC variance should match the scalar
        # _newey_west_se² used by the existing HAC module.
        from factrix._stats.hac import _newey_west_se

        rng = np.random.default_rng(seed=3)
        x = rng.standard_normal(200)
        se = _newey_west_se(x)
        _, V = _nw_hac_vector_mean(x.reshape(-1, 1))
        assert V[0, 0] == pytest.approx(se * se, rel=1e-12)


class TestWaldNWClusterMeans:
    def test_null_holds_when_means_equal(self):
        # Two slices with identical IID generation → contrast (1, -1)
        # should have large p (cannot reject equality).
        rng = np.random.default_rng(seed=0)
        T = 500
        Y = rng.standard_normal(size=(T, 2))
        _, p = _wald_nw_cluster_means(Y, R=np.array([[1.0, -1.0]]), q=0.0)
        assert p > 0.1

    def test_alt_detected_when_means_differ(self):
        rng = np.random.default_rng(seed=1)
        T = 500
        Y = np.column_stack([rng.standard_normal(T), rng.standard_normal(T) + 0.5])
        _, p = _wald_nw_cluster_means(Y, R=np.array([[1.0, -1.0]]), q=0.0)
        assert p < 0.001

    def test_omnibus_three_slices(self):
        # K=3, all means equal → joint Wald on R=[[1,-1,0],[1,0,-1]] big p.
        rng = np.random.default_rng(seed=2)
        T = 400
        Y = rng.standard_normal(size=(T, 3))
        R = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
        _, p = _wald_nw_cluster_means(Y, R=R, q=np.zeros(2))
        assert p > 0.1

    def test_short_sample_returns_unity(self):
        Y = np.array([[1.0, 2.0]])  # T=1
        W, p = _wald_nw_cluster_means(Y, R=np.array([[1.0, -1.0]]))
        assert (W, p) == (0.0, 1.0)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _wald_nw_cluster_means(np.arange(10.0), R=np.array([[1.0]]))


class TestWaldDoubleCluster:
    def _make_panel(self, n_dates=40, n_assets=25, beta=0.0, seed=0):
        rng = np.random.default_rng(seed=seed)
        date_ids = np.repeat(np.arange(n_dates), n_assets)
        asset_ids = np.tile(np.arange(n_assets), n_dates)
        n = n_dates * n_assets
        # x and eps both carry date + asset shocks → genuine two-way
        # cluster structure. Pure-iid x would make CGM ≈ HC0 and the
        # finite-sample subtraction can flip negative.
        x_date = rng.standard_normal(n_dates)
        x_asset = rng.standard_normal(n_assets) * 0.5
        x = x_date[date_ids] + x_asset[asset_ids] + rng.standard_normal(n) * 0.5
        e_date = rng.standard_normal(n_dates)
        e_asset = rng.standard_normal(n_assets) * 0.5
        eps = e_date[date_ids] + e_asset[asset_ids] + rng.standard_normal(n) * 0.5
        y = beta * x + eps
        X = np.column_stack([np.ones(n), x])
        return y, X, date_ids, asset_ids

    def test_null_holds(self):
        y, X, d, a = self._make_panel(beta=0.0, seed=0)
        # Test slope = 0.
        R = np.array([[0.0, 1.0]])
        _, p = _wald_double_cluster(y, X, R=R, date_ids=d, asset_ids=a)
        assert p > 0.05

    def test_alt_detected(self):
        y, X, d, a = self._make_panel(beta=0.5, seed=1)
        R = np.array([[0.0, 1.0]])
        _, p = _wald_double_cluster(y, X, R=R, date_ids=d, asset_ids=a)
        assert p < 0.01

    def test_symmetric_V(self):
        # Side-effect check via behaviour: identical (d, a) and (a, d)
        # cluster orderings → identical p (CGM is symmetric by
        # construction).
        y, X, d, a = self._make_panel(beta=0.3, seed=2)
        R = np.array([[0.0, 1.0]])
        _, p_da = _wald_double_cluster(y, X, R=R, date_ids=d, asset_ids=a)
        _, p_ad = _wald_double_cluster(y, X, R=R, date_ids=a, asset_ids=d)
        assert p_da == pytest.approx(p_ad)

    def test_rejects_id_length_mismatch(self):
        y, X, d, a = self._make_panel()
        with pytest.raises(ValueError, match="length must match"):
            _wald_double_cluster(
                y,
                X,
                R=np.array([[0.0, 1.0]]),
                date_ids=d[:-1],
                asset_ids=a,
            )

    def test_singular_returns_unity(self):
        # Perfectly collinear regressors → X'X singular → (0, 1).
        n = 50
        x = np.arange(n, dtype=float)
        X = np.column_stack([x, 2 * x])  # collinear
        y = x + np.random.default_rng(0).standard_normal(n)
        d = np.repeat(np.arange(10), 5)
        a = np.tile(np.arange(5), 10)
        out = _wald_double_cluster(
            y, X, R=np.array([[1.0, 0.0]]), date_ids=d, asset_ids=a
        )
        assert out == (0.0, 1.0)
