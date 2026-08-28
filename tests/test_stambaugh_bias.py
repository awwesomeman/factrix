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
from factrix._stats import _lag1_autocorr
from factrix._stats.constants import PERSISTENT_SERIES_AUTOCORR
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
        # Read field by field rather than ``np.isnan`` over the whole tuple:
        # the fit now carries an array field (``resid``), so the tuple is no
        # longer a flat float vector numpy can test elementwise.
        too_short = _amihud_hurvich_beta(np.arange(3.0), np.arange(3.0), lags=1)
        assert np.isnan(too_short.beta)
        assert np.isnan(too_short.alpha)
        assert too_short.n_used == 0
        assert too_short.resid.size == 0
        constant = np.full(40, 2.0)
        assert np.isnan(_amihud_hurvich_beta(np.arange(40.0), constant, lags=1).beta)

    def test_reports_the_rows_and_structural_residual_of_its_own_design(self):
        """The fit exposes the sample it actually used, not the caller's ``n``.

        The augmented design drops the first observation at ``h = 1`` and the
        last ``h - 1`` windows on top of that at ``h > 1``, and its residual
        is the STRUCTURAL one — ``y - alpha_c - beta_c x`` — not the augmented
        regression's, which has the innovation proxy projected out of it.
        """
        rng = np.random.default_rng(21)
        x, y = _stambaugh_draw(120, 0.95, -0.9, 5, rng)
        fit = _amihud_hurvich_beta(y, x, lags=4, overlap_periods=5)

        assert fit.n_used == 120 - 5
        assert fit.resid.shape == (fit.n_used,)
        m = fit.n_used
        assert fit.alpha == pytest.approx(
            float(np.mean(y[:m]) - fit.beta * np.mean(x[:m]))
        )
        assert fit.resid == pytest.approx(y[:m] - fit.alpha - fit.beta * x[:m])
        # The structural residual is centred by construction; the augmented
        # regression's own residual is not (the proxy carries a non-zero mean).
        assert float(np.mean(fit.resid)) == pytest.approx(0.0, abs=1e-10)

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
                _panel(x, y), overlap_periods=horizon, adf_threshold=None
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
        # Tightened from 0.12 once the h=1 path moved to the plain OLS
        # covariance the source paper uses: these cells measure 0.05-0.06
        # over 2000 draws, so 0.09 is well clear of MC noise at this
        # replication count without pinning a seed's exact value.
        assert size <= 0.09

    def test_no_bias_channel_leaves_the_estimate_alone(self):
        """rho = 0 means no Stambaugh bias, so the correction must be inert."""
        bias_ols, bias_ah, size = self._run(120, 0.95, 0.0, 1)
        assert abs(bias_ah) < 0.02
        assert abs(bias_ols) < 0.02
        assert size <= 0.09


class TestBiasChannelWarning:
    def test_fires_on_the_product_not_the_adf_screen(self):
        rng = np.random.default_rng(11)
        x, y = _stambaugh_draw(240, 0.95, -0.9, 1, rng)
        result = predictive_beta(_panel(x, y), overlap_periods=1, adf_threshold=None)
        assert result.metadata["stambaugh_bias_channel"] > _STAMBAUGH_CHANNEL_WARN
        assert WarningCode.PERSISTENT_REGRESSOR.value in result.warning_codes

    def test_silent_when_the_channel_is_absent(self):
        """A persistent regressor with rho = 0 carries no Stambaugh bias."""
        rng = np.random.default_rng(11)
        x, y = _stambaugh_draw(240, 0.5, 0.0, 1, rng)
        result = predictive_beta(_panel(x, y), overlap_periods=1, adf_threshold=None)
        assert result.metadata["stambaugh_bias_channel"] <= _STAMBAUGH_CHANNEL_WARN
        assert WarningCode.PERSISTENT_REGRESSOR.value not in result.warning_codes

    def test_metadata_reports_the_correction_applied(self):
        rng = np.random.default_rng(12)
        x, y = _stambaugh_draw(200, 0.9, -0.8, 1, rng)
        result = predictive_beta(_panel(x, y), overlap_periods=1, adf_threshold=None)
        assert result.metadata["stambaugh_adjusted"] is True
        assert result.metadata["stambaugh_bias_estimate"] == pytest.approx(
            result.metadata["beta_ols_uncorrected"] - result.value
        )
        assert result.metadata["innovation_corr"] < 0.0


class TestCorrectedFitDiagnostics:
    """REGRESSION (#873): every headline diagnostic describes the fit that
    produced ``value``.

    The metric used to report the corrected slope alongside an ``alpha``, an
    ``r_squared`` and a residual screen taken off the preceding UNCORRECTED
    OLS fit, on the full pair count rather than the rows the augmented design
    kept. In the Stambaugh regime the two coefficients differ materially, so
    those diagnostics described a model the caller was never shown.
    """

    @staticmethod
    def _ols_residual(x: np.ndarray, y: np.ndarray, beta_ols: float) -> np.ndarray:
        """OLS residual on the full sample, rebuilt from the reported slope."""
        alpha_ols = float(np.mean(y) - beta_ols * np.mean(x))
        return y - alpha_ols - beta_ols * x

    def test_alpha_and_r_squared_name_the_fit_they_belong_to(self):
        rng = np.random.default_rng(0)
        x, y = _stambaugh_draw(60, 0.95, -0.9, 1, rng)
        result = predictive_beta(_panel(x, y), overlap_periods=1, adf_threshold=None)

        assert result.metadata["stambaugh_adjusted"] is True
        beta_ols = result.metadata["beta_ols_uncorrected"]
        # The whole point of the cell: the two slopes are materially apart, so
        # a diagnostic taken off the wrong one is visibly wrong.
        assert abs(result.value - beta_ols) > 0.02

        # ``alpha`` closes over the corrected slope on the fit's own rows.
        m = result.n_obs
        assert m == len(x) - 1
        assert result.metadata["alpha"] == pytest.approx(
            float(np.mean(y[:m]) - result.value * np.mean(x[:m]))
        )

        # R-squared stays an OLS quantity and says so in its name. A
        # "corrected R-squared" is not a least-squares object at all - it can
        # go negative - so the key is labelled rather than recomputed.
        resid_ols = self._ols_residual(x, y, beta_ols)
        y_c = y - float(np.mean(y))
        expected_r2 = 1.0 - float(resid_ols @ resid_ols) / float(y_c @ y_c)
        assert result.metadata["r_squared_ols_uncorrected"] == pytest.approx(
            expected_r2
        )
        assert "r_squared" not in result.metadata

    def test_residual_screen_reads_the_corrected_models_residuals(self):
        rng = np.random.default_rng(0)
        x, y = _stambaugh_draw(60, 0.95, -0.9, 1, rng)
        result = predictive_beta(_panel(x, y), overlap_periods=1, adf_threshold=None)

        m = result.n_obs
        corrected_resid = y[:m] - result.metadata["alpha"] - result.value * x[:m]
        assert result.metadata["residual_lag1_autocorr"] == pytest.approx(
            _lag1_autocorr(corrected_resid)
        )

    def test_serial_correlation_verdict_follows_the_corrected_residuals(self):
        """A draw where the two residual series straddle the threshold.

        Seed 57 of this DGP puts the uncorrected OLS residual at 0.322 and the
        corrected model's at 0.292, either side of
        ``PERSISTENT_SERIES_AUTOCORR`` = 0.3: the metric used to flag serial
        correlation for a residual series it did not report.
        """
        rng = np.random.default_rng(57)
        x, y = _stambaugh_draw(60, 0.95, -0.9, 1, rng)
        result = predictive_beta(_panel(x, y), overlap_periods=1, adf_threshold=None)

        beta_ols = result.metadata["beta_ols_uncorrected"]
        ols_autocorr = _lag1_autocorr(self._ols_residual(x, y, beta_ols))
        assert ols_autocorr > PERSISTENT_SERIES_AUTOCORR

        assert result.metadata["residual_lag1_autocorr"] <= PERSISTENT_SERIES_AUTOCORR
        assert WarningCode.SERIAL_CORRELATION_DETECTED.value not in result.warning_codes

    def test_fallback_to_plain_ols_keeps_every_diagnostic_on_the_ols_fit(self):
        """Too short for the augmented design: the reported model IS the OLS
        one, so ``alpha``, the residual screen and the counts stay on it."""
        rng = np.random.default_rng(31)
        n = 24
        x, y = _stambaugh_draw(n, 0.5, 0.0, 20, rng)
        with pytest.warns(UserWarning, match="effective sample"):
            result = predictive_beta(
                _panel(x, y), overlap_periods=20, adf_threshold=None
            )

        assert result.metadata["stambaugh_adjusted"] is False
        assert result.n_obs == n
        assert result.metadata["n_periods_finite"] == n
        assert result.metadata["alpha"] == pytest.approx(
            float(np.mean(y) - result.value * np.mean(x))
        )
        assert result.metadata["residual_lag1_autocorr"] == pytest.approx(
            _lag1_autocorr(self._ols_residual(x, y, result.value)[::20])
        )
