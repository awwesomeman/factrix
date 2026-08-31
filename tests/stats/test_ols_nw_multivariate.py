"""Unit tests for ``_ols_nw_multivariate`` + ``_wald_p_linear``."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats import (
    _ols_nw_multivariate,
    _ols_scalar_wald_hac,
    _resolve_scalar_wald_hac,
    _wald_p_linear,
)


class TestPointEstimates:
    def test_recovers_ols_at_lags_zero(self):
        rng = np.random.default_rng(0)
        n = 500
        X = np.column_stack(
            [np.ones(n), rng.standard_normal(n), rng.standard_normal(n)]
        )
        true_beta = np.array([0.5, 1.2, -0.7])
        y = X @ true_beta + rng.standard_normal(n) * 0.3
        beta, _V, _ = _ols_nw_multivariate(y, X, lags=0)
        beta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(beta, beta_lstsq, atol=1e-12)

    def test_singular_design_is_not_computable(self):
        """Zero beta / zero V produced a t = 0/0 downstream that read as a
        non-rejection; a rank-deficient design has no estimate at all."""
        n = 100
        x1 = np.arange(n, dtype=float)
        # Two columns identical → X'X singular.
        X = np.column_stack([x1, x1])
        y = np.arange(n, dtype=float)
        beta, V, resid = _ols_nw_multivariate(y, X, lags=0)
        assert np.isnan(beta).all()
        assert np.isnan(V).all()
        np.testing.assert_array_equal(resid, np.zeros(n))

    def test_n_below_k_plus_one_is_not_computable(self):
        # 2 obs, 3 params — under-identified.
        X = np.array([[1.0, 0.5, -0.3], [1.0, 1.5, 0.2]])
        y = np.array([0.1, 0.2])
        beta, V, _ = _ols_nw_multivariate(y, X, lags=0)
        assert np.isnan(beta).all()
        assert np.isnan(V).all()


class TestHACVariance:
    def test_lags_increase_se_under_autocorrelated_resid(self):
        # Construct AR(1) errors so non-zero lags inflate the HAC SE
        # versus the iid estimate (lags=0).
        rng = np.random.default_rng(1)
        n = 1000
        x = rng.standard_normal(n)
        eps = np.zeros(n)
        for t in range(1, n):
            eps[t] = 0.7 * eps[t - 1] + rng.standard_normal()
        y = 0.5 + 1.0 * x + eps
        X = np.column_stack([np.ones(n), x])
        _, V0, _ = _ols_nw_multivariate(y, X, lags=0)
        _, V8, _ = _ols_nw_multivariate(y, X, lags=8)
        # SE on slope coefficient (index 1) should grow with lags.
        assert np.sqrt(V8[1, 1]) > np.sqrt(V0[1, 1])


class TestScalarWaldHacFit:
    def test_applies_the_resolved_scale_inside_the_consumer(self):
        rng = np.random.default_rng(12)
        n, h = 120, 5
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = 0.4 + 0.7 * x + rng.standard_normal(n)

        fit = _ols_scalar_wald_hac(y, X, overlap_periods=h)
        lags, scale, dof = _resolve_scalar_wald_hac(n, None, h)
        beta, covariance, resid = _ols_nw_multivariate(y, X, lags=lags)

        np.testing.assert_allclose(fit.beta, beta)
        np.testing.assert_allclose(fit.covariance, covariance * scale)
        np.testing.assert_allclose(fit.resid, resid)
        assert fit.lags == lags
        assert fit.dof == pytest.approx(dof)

    @pytest.mark.parametrize(
        ("n", "expected"),
        [(29, False), (30, True), (120, True)],
    )
    def test_bandwidth_warning_policy_suppresses_the_short_sample(
        self, n: int, expected: bool
    ) -> None:
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = 0.2 * x + rng.standard_normal(n)

        fit = _ols_scalar_wald_hac(y, X, overlap_periods=21)

        assert fit.bandwidth_ill_conditioned is expected


class TestWaldLinear:
    def test_size_under_null(self):
        # Repeated draws under H0 should give roughly uniform p-values.
        # Loose check: average rejection at α=0.05 within [0.02, 0.10]
        # for 200 reps with T=300.
        rng = np.random.default_rng(2024)
        rejects = 0
        n_trials = 200
        n = 300
        for _ in range(n_trials):
            x1 = rng.standard_normal(n)
            x2 = rng.standard_normal(n)
            X = np.column_stack([np.ones(n), x1, x2])
            y = rng.standard_normal(n) * 0.5  # no density
            beta, V, _ = _ols_nw_multivariate(y, X, lags=0)
            R = np.array([[0.0, 1.0, -1.0]])
            _, p = _wald_p_linear(beta, V, R, q=0.0)
            if p < 0.05:
                rejects += 1
        rate = rejects / n_trials
        assert 0.02 < rate < 0.10, f"empirical rejection rate {rate} out of band"

    def test_power_under_alternative(self):
        # When the restriction is clearly violated, Wald rejects.
        rng = np.random.default_rng(7)
        n = 500
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x1, x2])
        y = 1.0 * x1 + (-1.0) * x2 + rng.standard_normal(n) * 0.3
        beta, V, _ = _ols_nw_multivariate(y, X, lags=0)
        R = np.array([[0.0, 1.0, -1.0]])  # H0: β1 = β2
        _, p = _wald_p_linear(beta, V, R, q=0.0)
        assert p < 1e-6
