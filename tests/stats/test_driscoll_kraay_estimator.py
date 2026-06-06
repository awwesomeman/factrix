"""Tests for the Driscoll-Kraay estimator surface.

Covers the selection-only ``DriscollKraay`` Estimator (protocol /
registry / ``list_estimators`` / ``emits_for``), the numeric primitives
``_bartlett_lrcov`` / ``_driscoll_kraay_cov``, and the lowercase
``factrix.estimators.driscoll_kraay`` callable.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._axis import FactorDensity, FactorScope
from factrix._codes import StatCode
from factrix._stats import _bartlett_lrcov, _driscoll_kraay_cov
from factrix.estimators import driscoll_kraay
from factrix.stats import _ESTIMATOR_REGISTRY, DriscollKraay, Estimator, list_estimators


def _dk_cov_reference(
    X: np.ndarray, resid: np.ndarray, time_ids: np.ndarray, lags: int
) -> np.ndarray:
    """Independent rederivation of the DK covariance for parity checks."""
    uniq = np.unique(time_ids)
    n_periods = len(uniq)
    H = np.zeros((n_periods, X.shape[1]))
    for ti, t in enumerate(uniq):
        mask = time_ids == t
        H[ti] = (X[mask] * resid[mask][:, None]).sum(axis=0)
    S = H.T @ H
    for j in range(1, lags + 1):
        omega_j = H[j:].T @ H[:-j]
        w = 1.0 - j / (lags + 1)
        S = S + w * (omega_j + omega_j.T)
    xtx_inv = np.linalg.inv(X.T @ X)
    return xtx_inv @ S @ xtx_inv


class TestDriscollKraayEstimator:
    def test_satisfies_estimator_protocol(self):
        assert isinstance(DriscollKraay(), Estimator)

    def test_name(self):
        assert DriscollKraay().name == "DriscollKraay"

    def test_description_mentions_driscoll_and_cross_section(self):
        d = DriscollKraay().description.lower()
        assert "driscoll-kraay" in d
        assert "cross-section" in d

    def test_emits_p_dk(self):
        code = DriscollKraay().emits_for(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
        assert code is StatCode.P_DK

    def test_applicable_only_to_individual_dense(self):
        est = DriscollKraay()
        assert est.applicable_to(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
        assert not est.applicable_to(FactorScope.COMMON, FactorDensity.DENSE)
        assert not est.applicable_to(FactorScope.INDIVIDUAL, FactorDensity.SPARSE)

    def test_in_registry(self):
        assert "DriscollKraay" in {type(e).__name__ for e in _ESTIMATOR_REGISTRY}

    def test_surfaced_by_list_estimators_for_individual_dense(self):
        names = list_estimators(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
        assert "DriscollKraay" in names

    def test_excluded_from_common_cell(self):
        names = list_estimators(FactorScope.COMMON, FactorDensity.DENSE)
        assert "DriscollKraay" not in names


class TestBartlettLrcov:
    def test_zero_lags_is_white_outer_product(self):
        rng = np.random.default_rng(0)
        H = rng.normal(size=(20, 3))
        np.testing.assert_allclose(_bartlett_lrcov(H, 0), H.T @ H)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        H = rng.normal(size=(15, 2))
        S = _bartlett_lrcov(H, 3)
        np.testing.assert_allclose(S, S.T)

    def test_lags_beyond_sample_are_skipped(self):
        rng = np.random.default_rng(2)
        H = rng.normal(size=(4, 2))
        # lags far beyond T-1: extra lags contribute nothing (empty slices),
        # so the result is finite and symmetric, not an error.
        S = _bartlett_lrcov(H, 50)
        assert np.all(np.isfinite(S))


class TestDriscollKraayCov:
    def _panel(self, seed=7, n_dates=80, n_assets=12, rho=0.0):
        rng = np.random.default_rng(seed)
        g = 0.0
        rows_x, rows_y, rows_t = [], [], []
        for d in range(n_dates):
            g = rho * g + rng.normal(0, 1)
            u = rng.normal(0, 0.5, n_assets)
            v = rng.normal(0, 0.5, n_assets)
            f = g + u
            r = g + v
            rows_x.append(f)
            rows_y.append(r)
            rows_t.append(np.full(n_assets, d))
        x = np.concatenate(rows_x)
        y = np.concatenate(rows_y)
        t = np.concatenate(rows_t)
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return X, resid, t

    def test_matches_independent_rederivation(self):
        X, resid, t = self._panel()
        cov, n_periods, lags = _driscoll_kraay_cov(X, resid, t)
        ref = _dk_cov_reference(X, resid, t, lags)
        np.testing.assert_allclose(cov, ref, rtol=1e-10, atol=1e-12)
        assert n_periods == 80

    def test_auto_bandwidth_on_period_count(self):
        # auto_bartlett(80) = max(1, floor(4*(80/100)^(2/9))) = 3.
        X, resid, t = self._panel()
        _, _, lags = _driscoll_kraay_cov(X, resid, t)
        assert lags == 3

    def test_explicit_lags_respected_and_clipped(self):
        X, resid, t = self._panel(n_dates=10)
        _, n_periods, lags = _driscoll_kraay_cov(X, resid, t, lags=2)
        assert lags == 2
        # bandwidth above T-1 clips to n_periods - 1.
        _, _, lags_big = _driscoll_kraay_cov(X, resid, t, lags=999)
        assert lags_big == n_periods - 1

    def test_reduces_to_hc0_when_one_obs_per_period_no_lags(self):
        # With one observation per period and lags=0, the cross-sectional
        # score sum is the score itself and DK collapses to White HC0:
        # (X'X)^-1 (Σ e_t^2 x_t x_t') (X'X)^-1.
        rng = np.random.default_rng(11)
        n = 40
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        y = 0.5 * x + rng.normal(size=n)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        time_ids = np.arange(n)  # one obs per period
        cov, _, _ = _driscoll_kraay_cov(X, resid, time_ids, lags=0)
        xtx_inv = np.linalg.inv(X.T @ X)
        meat = (X * resid[:, None]).T @ (X * resid[:, None])
        hc0 = xtx_inv @ meat @ xtx_inv
        np.testing.assert_allclose(cov, hc0, rtol=1e-10)

    def test_singular_design_raises(self):
        # Constant regressor → singular X'X.
        n = 30
        X = np.column_stack([np.ones(n), np.ones(n)])
        resid = np.zeros(n)
        with pytest.raises(np.linalg.LinAlgError):
            _driscoll_kraay_cov(X, resid, np.arange(n))


class TestLowercaseCallable:
    def test_returns_cov_and_metadata(self):
        rng = np.random.default_rng(3)
        n = 60
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        resid = rng.normal(size=n)
        t = np.repeat(np.arange(20), 3)
        cov, meta = driscoll_kraay(X, resid, t)
        assert cov.shape == (2, 2)
        assert meta["n_periods"] == 20
        assert meta["kernel"] == "bartlett"
        assert isinstance(meta["driscoll_kraay_lags"], int)

    def test_negative_lags_rejected(self):
        X = np.ones((4, 2))
        with pytest.raises(ValueError, match="lags must be"):
            driscoll_kraay(X, np.zeros(4), np.arange(4), lags=-1)
