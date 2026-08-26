"""Stambaugh (1999) bias in ``predictive_beta``, and the Amihud-Hurvich fix.

The ADF screen the metric shipped with proxies persistence only and carries
no information about the innovation correlation that drives the bias, so it
missed the main channel: at ``phi = 0.5`` the bias is present and the flag
fired on 1% of runs; at ``T = 240`` the test still rejected 11% while the
screen stayed silent on 62% of draws.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._stats.ols import _amihud_hurvich_beta
from factrix.metrics.predictive_beta import (
    _STAMBAUGH_CHANNEL_WARN,
    predictive_beta,
)


def _stambaugh_draw(
    n_periods: int, phi: float, rho: float, horizon: int, rng
) -> tuple[np.ndarray, np.ndarray]:
    """Predictive system with a TRUE beta of zero.

    ``x`` is AR(phi) with innovation ``u``; the return innovation ``e``
    correlates ``rho`` with ``u``. That correlation is the whole bias
    channel — at ``rho = 0`` OLS is already unbiased.
    """
    total = n_periods + horizon + 1
    u = rng.normal(size=total)
    w = rng.normal(size=total)
    e = rho * u + np.sqrt(1.0 - rho**2) * w
    x = np.zeros(total)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + u[t]
    cumulative = np.cumsum(np.concatenate([[0.0], e]))
    returns = (
        cumulative[1 + horizon : 1 + horizon + n_periods]
        - cumulative[1 : 1 + n_periods]
    )
    return x[:n_periods], returns


def _panel(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    dates = [dt.date(2000, 1, 1) + dt.timedelta(days=i) for i in range(len(x))]
    return pl.DataFrame(
        {"date": dates, "asset_id": ["A"] * len(x), "factor": x, "forward_return": y}
    )


class TestAugmentedRegressionEstimator:
    def test_recovers_a_known_slope_without_persistence(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = 0.5 * x + rng.standard_normal(200)
        fit = _amihud_hurvich_beta(y, x, lags=2)
        assert fit.beta == pytest.approx(0.5, abs=0.15)
        assert fit.p_value < 0.01

    def test_phi_is_bias_corrected_upward(self):
        """AH eq. 6: phi_c = phi + (1+3phi)/n + 3(1+3phi)/n^2."""
        rng = np.random.default_rng(1)
        n = 120
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + rng.standard_normal()
        fit = _amihud_hurvich_beta(rng.standard_normal(n), x, lags=2)
        expected = fit.phi + (1 + 3 * fit.phi) / n + 3 * (1 + 3 * fit.phi) / n**2
        assert fit.phi_corrected == pytest.approx(expected)
        assert fit.phi_corrected > fit.phi

    def test_short_or_degenerate_sample_is_not_computable(self):
        assert np.isnan(_amihud_hurvich_beta(np.arange(3.0), np.arange(3.0), lags=1))[0]
        constant = np.full(40, 2.0)
        assert np.isnan(_amihud_hurvich_beta(np.arange(40.0), constant, lags=1).beta)

    def test_generated_regressor_term_widens_the_standard_error(self):
        """Without it the proxy absorbs the correlated part of e and the
        test rejects ~50% of true nulls."""
        from factrix._stats.ols import _ols_nw_multivariate

        rng = np.random.default_rng(7)
        x, y = _stambaugh_draw(200, 0.95, -0.9, 1, rng)
        fit = _amihud_hurvich_beta(y, x, lags=4)

        # Rebuild the augmented design's raw OLS SE for the same fit.
        x_lag, x_cur = x[:-1], x[1:]
        proxy = (
            x_cur
            - (x_cur.mean() - fit.phi_corrected * x_lag.mean())
            - fit.phi_corrected * x_lag
        )
        design = np.column_stack([np.ones(len(proxy)), x[: len(proxy)], proxy])
        _, cov, _ = _ols_nw_multivariate(y[: len(proxy)], design, lags=4)
        se_aug = float(np.sqrt(cov[1, 1]))
        assert fit.se > se_aug


class TestBiasAndSize:
    """Monte Carlo on the null: bias and rejection rate, before and after."""

    @staticmethod
    def _run(n_periods, phi, rho, horizon, n_reps=200):
        rng = np.random.default_rng(3)
        betas_ols, betas_ah, rej_ah = [], [], 0
        for _ in range(n_reps):
            x, y = _stambaugh_draw(n_periods, phi, rho, horizon, rng)
            result = predictive_beta(
                _panel(x, y), forward_periods=horizon, adf_threshold=None
            )
            betas_ah.append(result.value)
            betas_ols.append(result.metadata["beta_ols_uncorrected"])
            rej_ah += result.p_value is not None and result.p_value < 0.05
        return np.mean(betas_ols), np.mean(betas_ah), rej_ah / n_reps

    @pytest.mark.parametrize(
        ("n_periods", "phi", "rho", "horizon"),
        [(120, 0.95, -0.9, 1), (120, 0.95, -0.9, 5), (120, 0.5, -0.9, 1)],
    )
    def test_bias_shrinks(self, n_periods, phi, rho, horizon):
        bias_ols, bias_ah, _ = self._run(n_periods, phi, rho, horizon)
        assert abs(bias_ah) < abs(bias_ols) / 2.0

    @pytest.mark.parametrize(
        ("n_periods", "phi", "rho", "horizon"),
        [(120, 0.5, -0.9, 1), (240, 0.95, -0.9, 1)],
    )
    def test_size_improves(self, n_periods, phi, rho, horizon):
        _, _, size = self._run(n_periods, phi, rho, horizon)
        assert size <= 0.12

    def test_no_bias_channel_leaves_the_estimate_alone(self):
        """rho = 0 means no Stambaugh bias, so the correction must be inert."""
        bias_ols, bias_ah, size = self._run(120, 0.95, 0.0, 1)
        assert abs(bias_ah) < 0.02
        assert abs(bias_ols) < 0.02
        assert size <= 0.10


class TestBiasChannelWarning:
    def test_fires_on_the_product_not_the_adf_screen(self):
        rng = np.random.default_rng(11)
        x, y = _stambaugh_draw(240, 0.95, -0.9, 1, rng)
        result = predictive_beta(_panel(x, y), forward_periods=1, adf_threshold=None)
        assert result.metadata["stambaugh_bias_channel"] > _STAMBAUGH_CHANNEL_WARN
        assert WarningCode.PERSISTENT_REGRESSOR.value in result.warning_codes

    def test_silent_when_the_channel_is_absent(self):
        """A persistent regressor with rho = 0 carries no Stambaugh bias."""
        rng = np.random.default_rng(11)
        x, y = _stambaugh_draw(240, 0.5, 0.0, 1, rng)
        result = predictive_beta(_panel(x, y), forward_periods=1, adf_threshold=None)
        assert result.metadata["stambaugh_bias_channel"] <= _STAMBAUGH_CHANNEL_WARN
        assert WarningCode.PERSISTENT_REGRESSOR.value not in result.warning_codes

    def test_metadata_reports_the_correction_applied(self):
        rng = np.random.default_rng(12)
        x, y = _stambaugh_draw(200, 0.9, -0.8, 1, rng)
        result = predictive_beta(_panel(x, y), forward_periods=1, adf_threshold=None)
        assert result.metadata["stambaugh_adjusted"] is True
        assert result.metadata["stambaugh_bias_estimate"] == pytest.approx(
            result.metadata["beta_ols_uncorrected"] - result.value
        )
        assert result.metadata["innovation_corr"] < 0.0
