"""Tests for cluster-Wald primitives in ``factrix._stats.wald``."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from factrix._stats.wald import (
    _nw_hac_vector_mean,
    _wald_nw_cluster_means,
    _wald_p_linear,
    _wald_two_way_cluster,
)
from scipy import stats as sp_stats


class TestWaldFiniteSampleF:
    def test_df_denom_matches_f_reference(self):
        # With df_denom the p-value is the F survival of W/r, strictly
        # larger (more conservative) than the asymptotic chi2 survival.
        beta = np.array([2.0, 0.0])
        V = np.eye(2)
        R = np.array([[1.0, 0.0]])
        W, p_f = _wald_p_linear(beta, V, R, df_denom=9)
        _, p_chi2 = _wald_p_linear(beta, V, R)
        assert p_f == pytest.approx(sp_stats.f.sf(W / 1, dfn=1, dfd=9))
        assert p_f > p_chi2

    def test_df_denom_non_positive_is_not_computable(self):
        beta = np.array([5.0])
        V = np.array([[1.0]])
        R = np.array([[1.0]])
        W, p = _wald_p_linear(beta, V, R, df_denom=0)
        assert np.isnan(W) and np.isnan(p)

    def test_singular_middle_is_not_computable(self):
        # Zero contrast variance with a non-zero contrast: maximum-evidence
        # degeneracy, not the null — NaN, never (0, 1).
        beta = np.array([5.0, 1.0])
        V = np.zeros((2, 2))
        R = np.array([[1.0, -1.0]])
        W, p = _wald_p_linear(beta, V, R)
        assert np.isnan(W) and np.isnan(p)

    def test_cluster_means_small_T_more_conservative(self):
        # Same per-date panel, the finite-sample F reference must not
        # under-state p relative to the asymptotic chi2 the means imply.
        rng = np.random.default_rng(seed=7)
        T = 12
        Y = np.column_stack([rng.standard_normal(T), rng.standard_normal(T) + 0.4])
        R = np.array([[1.0, -1.0]])
        W, p = _wald_nw_cluster_means(Y, R=R, q=0.0)
        assert p == pytest.approx(sp_stats.f.sf(W / 1, dfn=1, dfd=T - 1))
        assert p > sp_stats.chi2.sf(W, df=1)


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

    def test_short_sample_is_not_computable(self):
        Y = np.array([[1.0, 2.0]])  # T=1
        W, p = _wald_nw_cluster_means(Y, R=np.array([[1.0, -1.0]]))
        assert np.isnan(W) and np.isnan(p)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _wald_nw_cluster_means(np.arange(10.0), R=np.array([[1.0]]))


class TestWaldTwoWayCluster:
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
        _, p = _wald_two_way_cluster(y, X, R=R, date_ids=d, asset_ids=a)
        assert p > 0.05

    def test_alt_detected(self):
        y, X, d, a = self._make_panel(beta=0.5, seed=1)
        R = np.array([[0.0, 1.0]])
        _, p = _wald_two_way_cluster(y, X, R=R, date_ids=d, asset_ids=a)
        assert p < 0.01

    def test_symmetric_V(self):
        # Side-effect check via behaviour: identical (d, a) and (a, d)
        # cluster orderings → identical p (CGM is symmetric by
        # construction).
        y, X, d, a = self._make_panel(beta=0.3, seed=2)
        R = np.array([[0.0, 1.0]])
        _, p_da = _wald_two_way_cluster(y, X, R=R, date_ids=d, asset_ids=a)
        _, p_ad = _wald_two_way_cluster(y, X, R=R, date_ids=a, asset_ids=d)
        assert p_da == pytest.approx(p_ad)

    def test_rejects_id_length_mismatch(self):
        y, X, d, a = self._make_panel()
        with pytest.raises(ValueError, match="length must match"):
            _wald_two_way_cluster(
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
        out = _wald_two_way_cluster(
            y, X, R=np.array([[1.0, 0.0]]), date_ids=d, asset_ids=a
        )
        assert np.isnan(out[0]) and np.isnan(out[1])


class TestTwoWayClusterLabelDtypes:
    """REGRESSION: np.unique(column_stack, axis=0) raises on object/datetime."""

    @staticmethod
    def _panel(n_dates: int = 20, n_assets: int = 8, seed: int = 5):
        rng = np.random.default_rng(seed)
        dates, assets, x, y = [], [], [], []
        base = np.datetime64("2024-01-01")
        for t in range(n_dates):
            for a in range(n_assets):
                xv = float(rng.standard_normal())
                dates.append(base + np.timedelta64(t, "D"))
                assets.append(f"ASSET_{a}")
                x.append(xv)
                y.append(0.4 * xv + float(rng.normal(0, 0.3)))
        X = np.column_stack([np.ones(len(x)), np.asarray(x)])
        return (
            np.asarray(y),
            X,
            np.asarray(dates, dtype="datetime64[D]"),
            np.asarray(assets, dtype=object),
        )

    def test_datetime_dates_and_string_assets(self):
        y, X, date_ids, asset_ids = self._panel()
        R = np.array([[0.0, 1.0]])
        # Old code raised TypeError/ValueError inside np.unique(..., axis=0).
        W, p = _wald_two_way_cluster(y, X, R=R, date_ids=date_ids, asset_ids=asset_ids)
        assert np.isfinite(W) and W > 0.0
        assert 0.0 <= p <= 1.0

    def test_label_encoding_is_dtype_invariant(self):
        y, X, date_ids, asset_ids = self._panel()
        R = np.array([[0.0, 1.0]])
        ref = _wald_two_way_cluster(y, X, R=R, date_ids=date_ids, asset_ids=asset_ids)
        # Same partition, integer labels -> identical statistic.
        int_dates = np.unique(date_ids, return_inverse=True)[1].astype(np.int64)
        int_assets = np.unique(asset_ids, return_inverse=True)[1].astype(np.int64)
        got = _wald_two_way_cluster(y, X, R=R, date_ids=int_dates, asset_ids=int_assets)
        assert got[0] == pytest.approx(ref[0])
        assert got[1] == pytest.approx(ref[1])

    def test_single_cluster_margin_is_degenerate(self):
        y, X, date_ids, _ = self._panel()
        one_asset = np.array(["ONLY"] * len(y), dtype=object)
        out = _wald_two_way_cluster(
            y, X, R=np.array([[0.0, 1.0]]), date_ids=date_ids, asset_ids=one_asset
        )
        assert np.isnan(out[0]) and np.isnan(out[1])


class TestTwoWayClusterAgreesWithPooledBeta:
    """The CGM/Stata small-sample factors must match ``pooled_beta``'s."""

    def test_wald_statistic_equals_pooled_beta_t_squared(self):
        import polars as pl
        from factrix.metrics.fm_beta import pooled_beta

        rng = np.random.default_rng(19)
        n_dates, n_assets = 25, 10
        rows = []
        base = datetime(2024, 1, 1)
        for t in range(n_dates):
            for a in range(n_assets):
                f = float(rng.standard_normal())
                rows.append(
                    {
                        "date": base + timedelta(days=t),
                        "asset_id": f"A{a}",
                        "factor": f,
                        "forward_return": 0.3 * f + float(rng.normal(0, 0.5)),
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = pooled_beta(panel, cluster_col="date", two_way_cluster_col="asset_id")

        y = panel["forward_return"].to_numpy()
        X = np.column_stack([np.ones(panel.height), panel["factor"].to_numpy()])
        W, _ = _wald_two_way_cluster(
            y,
            X,
            R=np.array([[0.0, 1.0]]),
            date_ids=panel["date"].to_numpy(),
            asset_ids=panel["asset_id"].to_numpy(),
        )
        # Both estimators now apply G/(G-1) per dimension and (N-1)/(N-K)
        # overall, so W == t^2 exactly. Before the fix the Wald V_DC omitted
        # every correction and W came out ~10% too large.
        assert pytest.approx(result.stat**2, rel=1e-9) == W
