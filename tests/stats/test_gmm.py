"""Two-step efficient GMM J-statistic + Bartlett-kernel long-run covariance.

Pure overid (n_params = 0) on multivariate moment systems; tests cover
the H₀ chi-square calibration, autocorrelation handling via the
overlap-aware bandwidth, and the singular-weight fallback path.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats import _long_run_covariance, _two_step_gmm_j_stat
from scipy import stats as sp_stats


class TestLongRunCovariance:
    def test_zero_lags_recovers_sample_covariance(self):
        # With lags=0, long-run cov collapses to Γ_0 (sample cov, ddof=0).
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((500, 3))
        s = _long_run_covariance(moments, lags=0)
        sample_cov = ((moments - moments.mean(0)).T @ (moments - moments.mean(0))) / 500
        np.testing.assert_allclose(s, sample_cov, atol=1e-12)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        moments = rng.standard_normal((200, 4))
        s = _long_run_covariance(moments, forward_periods=5)
        np.testing.assert_allclose(s, s.T, atol=1e-12)

    def test_overlap_inflates_diagonal_on_persistent_series(self):
        # MA(h-1) overlap from rolling-sum returns inflates long-run var.
        rng = np.random.default_rng(2)
        raw = rng.standard_normal((500, 1))
        h = 10
        kernel = np.ones(h)
        overlapped = np.convolve(raw[:, 0], kernel, mode="valid").reshape(-1, 1)
        s_short = _long_run_covariance(overlapped, forward_periods=1)
        s_long = _long_run_covariance(overlapped, forward_periods=h)
        assert s_long[0, 0] > s_short[0, 0]


class TestTwoStepGmmJStat:
    def test_chi_square_calibration_under_null(self):
        # H₀: E[g] = 0 with iid Gaussian moments. J should distribute as
        # χ²(K). Check coverage of the 5% upper tail across many seeds.
        k = 4
        n = 400
        rejections = 0
        trials = 400
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            moments = rng.standard_normal((n, k))
            j, df, _, _ = _two_step_gmm_j_stat(moments, forward_periods=1)
            assert df == k
            crit = sp_stats.chi2.ppf(0.95, df=k)
            if j > crit:
                rejections += 1
        # Expected ~5%; allow a generous Monte-Carlo band.
        assert 0.02 <= rejections / trials <= 0.10

    def test_strong_signal_rejects(self):
        # E[g] far from 0 → J large → reject H₀.
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((300, 3)) + 0.5
        j, df, _, singular = _two_step_gmm_j_stat(moments, forward_periods=1)
        p = float(sp_stats.chi2.sf(j, df=df))
        assert p < 0.001
        assert not singular

    @pytest.mark.parametrize("max_iter,expected", [(1, 1), (2, 2), (10, 2)])
    def test_n_iter_capped_at_two(self, max_iter, expected):
        # Pure-overid has no parameter to update, so iteration beyond
        # step 2 has no effect on θ; n_iter is capped accordingly.
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((200, 2))
        _, _, n_iter, _ = _two_step_gmm_j_stat(
            moments, forward_periods=1, max_iter=max_iter
        )
        assert n_iter == expected

    def test_singular_weight_uses_pinv_and_flags(self):
        # Two perfectly collinear moment columns → Ŝ rank-deficient.
        rng = np.random.default_rng(0)
        base = rng.standard_normal((100, 1))
        moments = np.hstack([base, 2.0 * base])
        j, df, _, singular = _two_step_gmm_j_stat(moments, forward_periods=1)
        assert singular is True
        assert df == 2
        assert np.isfinite(j)

    def test_n_params_nonzero_not_implemented(self):
        rng = np.random.default_rng(0)
        moments = rng.standard_normal((100, 3))
        with pytest.raises(NotImplementedError, match="Parametric GMM"):
            _two_step_gmm_j_stat(moments, n_params=1, forward_periods=1)

    def test_chi_square_calibration_small_sample(self):
        # Asymptotics should kick in by T=100 with K=3 iid moments.
        k = 3
        n = 100
        rejections = 0
        trials = 400
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            moments = rng.standard_normal((n, k))
            j, df, _, _ = _two_step_gmm_j_stat(moments, forward_periods=1)
            if j > sp_stats.chi2.ppf(0.95, df=df):
                rejections += 1
        # Looser band than the asymptotic test — finite-sample size
        # distortion is real but should stay within 2x nominal.
        assert 0.02 <= rejections / trials <= 0.12

    def test_ar1_moments_with_overlap_aware_bandwidth(self):
        # AR(1) ρ=0.7 moments under H₀: E[g]=0. With forward_periods set
        # high enough to capture the persistence, J should remain
        # χ²-calibrated; with forward_periods=1 (no HAC widening beyond
        # default), Type-I error inflates noticeably. Test asserts the
        # overlap-aware version is the more conservative one.
        k = 2
        n = 400
        rho = 0.7
        rejections_short = 0
        rejections_long = 0
        trials = 200
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            innov = rng.standard_normal((n, k))
            moments = np.zeros_like(innov)
            moments[0] = innov[0]
            for t in range(1, n):
                moments[t] = rho * moments[t - 1] + innov[t]
            j_short, df, _, _ = _two_step_gmm_j_stat(moments, forward_periods=1)
            j_long, _, _, _ = _two_step_gmm_j_stat(moments, forward_periods=20)
            crit = sp_stats.chi2.ppf(0.95, df=df)
            if j_short > crit:
                rejections_short += 1
            if j_long > crit:
                rejections_long += 1
        # Overlap-aware bandwidth widens Ŝ → smaller J → fewer rejections.
        assert rejections_long <= rejections_short

    def test_long_run_covariance_is_exactly_symmetric(self):
        # Floating-point symmetry safeguard — important for downstream
        # solvers that assume Hermitian Ŝ.
        rng = np.random.default_rng(3)
        moments = rng.standard_normal((300, 5))
        s = _long_run_covariance(moments, forward_periods=10)
        assert np.array_equal(s, s.T)

    def test_overlap_inflates_long_run_cov_lowers_j(self):
        # Positively autocorrelated moments → larger Ŝ → smaller J for
        # the same g_bar (the test "knows" SE is wider).
        rng = np.random.default_rng(7)
        raw = rng.standard_normal(400)
        h = 10
        kernel = np.ones(h)
        series = np.convolve(raw, kernel, mode="valid")
        # Two correlated moment columns built from the same persistent series.
        moments = np.column_stack(
            [series, series + 0.1 * rng.standard_normal(len(series))]
        )
        j_short, _, _, _ = _two_step_gmm_j_stat(moments, forward_periods=1)
        j_long, _, _, _ = _two_step_gmm_j_stat(moments, forward_periods=h)
        assert j_long < j_short
