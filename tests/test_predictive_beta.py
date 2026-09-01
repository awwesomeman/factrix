"""Tests for single-asset dense ``predictive_beta``."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix._stats import _resolve_nw_lags
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN
from factrix.metrics.predictive_beta import predictive_beta


def _ts_panel(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    n = len(x)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {"date": dates, "asset_id": ["A"] * n, "factor": x, "forward_return": y}
    )


class TestPredictiveBetaStatistic:
    def test_estimates_positive_predictive_slope(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=240)
        y = 0.7 * x + 0.35 * rng.normal(size=240)

        result = predictive_beta(_ts_panel(x, y), overlap_periods=5)
        reference_beta = np.polyfit(x, y, 1)[0]

        # ``value`` is the Stambaugh-corrected slope; the raw OLS slope
        # this reference reproduces is kept in metadata.
        assert result.metadata["beta_ols_uncorrected"] == pytest.approx(reference_beta)
        assert result.value == pytest.approx(reference_beta, rel=0.02)
        assert result.metadata["stambaugh_adjusted"] is True
        assert result.stat > 0
        assert result.p_value < 0.01
        # 240 finite pairs, 235 rows in the corrected fit: the Amihud-Hurvich
        # design needs a lagged predictor and a horizon-summed innovation
        # proxy, so it drops ``overlap_periods`` rows. ``n_obs`` reports the
        # rows the headline test actually ran on; the pair count stays
        # auditable in ``n_periods_finite``.
        assert result.n_obs == 240 - 5
        assert result.metadata["n_periods_finite"] == 240
        assert result.n_obs_axis == "periods"
        assert result.metadata["h0"] == "beta=0"
        assert result.metadata["newey_west_lags"] == _resolve_nw_lags(240, None, 5)
        assert result.metadata["unit_root_suspected"] is False
        assert WarningCode.PERSISTENT_REGRESSOR.value not in result.warning_codes

    def test_metadata_names_the_covariance_that_ran(self) -> None:
        rng = np.random.default_rng(1009)
        x = rng.normal(size=240)
        y = 0.2 * x + rng.normal(size=240)
        panel = _ts_panel(x, y)

        h1_default = predictive_beta(panel, overlap_periods=1, newey_west_lags=None)
        h1_explicit = predictive_beta(panel, overlap_periods=1, newey_west_lags=30)
        h5 = predictive_beta(panel, overlap_periods=5, newey_west_lags=1)

        assert h1_default.stat == h1_explicit.stat
        assert h1_default.p_value == h1_explicit.p_value
        assert "homoskedastic" in h1_default.metadata["method"]
        assert "Newey-West" not in h1_default.metadata["method"]
        assert h1_default.metadata["hac_applied"] is False
        assert h1_default.metadata["har_lags"] is None
        assert (
            WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED.value
            not in h1_explicit.warning_codes
        )

        assert "Newey-West" in h5.metadata["method"]
        assert h5.metadata["hac_applied"] is True
        assert isinstance(h5.metadata["har_lags"], int)

    def test_persistent_factor_sets_adf_warning(self) -> None:
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(size=240))
        y = 0.05 * x + rng.normal(size=240)

        result = predictive_beta(_ts_panel(x, y), overlap_periods=1)

        assert result.metadata["adf_p"] > 0.10
        assert result.metadata["unit_root_suspected"] is True
        assert WarningCode.PERSISTENT_REGRESSOR.value in result.warning_codes

    def test_adf_threshold_none_disables_persistence_check(self) -> None:
        rng = np.random.default_rng(0)
        result = predictive_beta(
            _ts_panel(rng.normal(size=120), rng.normal(size=120)),
            overlap_periods=1,
            adf_threshold=None,
        )
        assert "adf_stat" not in result.metadata
        assert "adf_p" not in result.metadata
        assert "unit_root_suspected" not in result.metadata

    def test_adf_threshold_out_of_range_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="adf_threshold"):
            predictive_beta(
                _ts_panel(rng.normal(size=120), rng.normal(size=120)),
                adf_threshold=1.0,
            )

    def test_independent_factor_not_significant(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=300)
        y = rng.normal(size=300)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=1)
        assert result.p_value > 0.05

    def test_pairwise_complete_rows_define_sample(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=80)
        y = x + rng.normal(size=80)
        panel = _ts_panel(x, y).with_columns(
            pl.when(pl.int_range(pl.len()) % 10 == 0)
            .then(None)
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )

        result = predictive_beta(panel, overlap_periods=1)
        # 72 finite pairs; the corrected fit spends the first one on the AR(1)
        # lag, so the reported sample is 71.
        assert result.metadata["n_periods_finite"] == 72
        assert result.n_obs == 71


class TestPredictiveBetaShortCircuits:
    def test_insufficient_periods(self) -> None:
        rng = np.random.default_rng(3)
        n = MIN_PERIODS_HARD - 1
        result = predictive_beta(_ts_panel(rng.normal(size=n), rng.normal(size=n)))
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_predictive_periods"
        assert result.n_obs_axis == "periods"

    def test_degenerate_factor_variance(self) -> None:
        x = np.ones(MIN_PERIODS_HARD)
        y = np.arange(MIN_PERIODS_HARD, dtype=float)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=1)
        assert math.isnan(result.value)
        assert result.p_value is None
        assert result.metadata["signal_status"] == "degenerate_zero_variance"
        assert "reason" not in result.metadata
        assert WarningCode.DEGENERATE_VARIANCE.value in result.warning_codes
        assert result.n_obs == MIN_PERIODS_HARD

    def test_missing_return_column(self) -> None:
        rng = np.random.default_rng(4)
        panel = _ts_panel(rng.normal(size=40), rng.normal(size=40)).drop(
            "forward_return"
        )
        result = predictive_beta(panel)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_return_column"

    def test_warns_between_hard_and_warn_periods(self) -> None:
        rng = np.random.default_rng(5)
        n = MIN_PERIODS_WARN - 5
        with pytest.warns(UserWarning, match="MIN_PERIODS_WARN"):
            result = predictive_beta(
                _ts_panel(rng.normal(size=n), rng.normal(size=n)),
                overlap_periods=1,
            )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in result.warning_codes


class TestPredictiveBetaDispatch:
    def _single_asset_forward_panel(self) -> pl.DataFrame:
        raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=160, rng=10)
        first = raw["asset_id"].unique().sort()[0]
        return fx.preprocess.compute_forward_return(
            raw.filter(pl.col("asset_id") == first),
            forward_periods=5,
        )

    def test_evaluate_runs_on_single_asset_dense_timeseries(self) -> None:
        panel = self._single_asset_forward_panel()
        er = fx.evaluate(
            panel,
            metrics={"pb": predictive_beta()},
            factor_cols=["factor"],
            forward_periods=5,
        )["factor"]

        out = er.metrics["pb"]
        assert er.cell[2] is fx.DataStructure.TIMESERIES
        assert out.name == "pb"
        assert not math.isnan(out.value)

    def test_panel_data_rejects_predictive_beta_by_structure(self) -> None:
        panel = fx.preprocess.compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=20, n_dates=80, rng=11),
            forward_periods=5,
        )
        with pytest.raises(fx.IncompatibleAxisError, match="TIMESERIES"):
            fx.evaluate(
                panel,
                metrics={"pb": predictive_beta()},
                factor_cols=["factor"],
                forward_periods=5,
            )

    def test_inspect_data_marks_single_asset_only(self) -> None:
        single = self._single_asset_forward_panel()
        single_info = fx.inspect_data(single, factor_cols=["factor"])
        panel_info = fx.inspect_data(
            fx.datasets.make_cs_panel(n_assets=20, n_dates=80), factor_cols=["factor"]
        )

        single_pb = next(m for m in single_info.metrics if m.name == "predictive_beta")
        panel_pb = next(m for m in panel_info.metrics if m.name == "predictive_beta")
        assert single_pb.usable is True
        assert panel_pb.usable is False
        assert any("cell mismatch" in b for b in panel_pb.blockers)


class TestPredictiveBetaNonFinite:
    """REGRESSION: polars drop_nulls keeps float NaN, poisoning the NW slope."""

    @staticmethod
    def _series(n: int = 120, seed: int = 3) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        x = rng.standard_normal(n)
        y = 0.7 * x + rng.normal(0, 0.2, n)
        return pl.DataFrame(
            {"date": dates, "asset_id": ["A"] * n, "factor": x, "forward_return": y}
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_nan_return_cell_is_dropped(self):
        data = self._series()
        poisoned = data.with_columns(
            pl.when(pl.int_range(pl.len()) == 10)
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        result = predictive_beta(poisoned)
        assert np.isfinite(result.value)
        assert np.isfinite(result.stat)
        # One poisoned cell leaves 119 finite pairs; the corrected fit at the
        # default ``overlap_periods=5`` runs on 114 of them.
        assert result.metadata["n_periods_finite"] == data.height - 1
        assert result.n_obs == data.height - 1 - 5

    def test_nan_factor_cell_is_dropped(self):
        data = self._series()
        poisoned = data.with_columns(
            pl.when(pl.int_range(pl.len()) == 4)
            .then(float("nan"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        result = predictive_beta(poisoned)
        assert np.isfinite(result.value)
        assert result.metadata["n_periods_finite"] == data.height - 1
        assert result.n_obs == data.height - 1 - 5

    def test_matches_manually_pruned_input(self):
        data = self._series()
        poisoned = data.with_columns(
            pl.when(pl.int_range(pl.len()) == 7)
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        pruned = data.with_row_index("_i").filter(pl.col("_i") != 7).drop("_i")
        assert predictive_beta(poisoned).value == pytest.approx(
            predictive_beta(pruned).value
        )


class TestPredictiveBetaReportedSample:
    """REGRESSION (#873): the reported counts are the corrected fit's rows.

    ``n_obs`` / ``n_periods`` used to report the finite-pair count while the
    headline slope came from the Amihud-Hurvich augmented design, which spends
    the first observation on the AR(1) lag and, at ``h > 1``, the last
    ``h - 1`` windows on the horizon-summed innovation proxy. The two counts
    are both auditable now: ``n_periods_finite`` is the pre-fit pair count.
    """

    @pytest.mark.parametrize("overlap_periods", [1, 2, 5, 21])
    def test_n_obs_counts_the_rows_the_headline_test_used(
        self, overlap_periods: int
    ) -> None:
        rng = np.random.default_rng(9)
        n_finite = 200
        result = predictive_beta(
            _ts_panel(rng.normal(size=n_finite), rng.normal(size=n_finite)),
            overlap_periods=overlap_periods,
        )
        assert result.metadata["stambaugh_adjusted"] is True
        assert result.metadata["n_periods_finite"] == n_finite
        assert result.n_obs == n_finite - overlap_periods
        assert result.metadata["n_periods"] == result.n_obs
        assert result.metadata["n_periods_effective"] == (
            result.n_obs // overlap_periods
        )


class TestPredictiveBetaOverlapAndPersistenceScreens:
    """The two screens that read the sample the HAC standard error actually
    has: the effective (non-overlapping) observation count, and the residual
    persistence above which no path in the library is calibrated.
    """

    def test_short_series_with_a_long_horizon_flags_the_effective_sample(self):
        rng = np.random.default_rng(3)
        n = 120
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        with pytest.warns(UserWarning, match="effective sample"):
            result = predictive_beta(_ts_panel(x, y), overlap_periods=21)
        # The corrected fit drops ``overlap_periods`` rows, so the effective
        # count reads 99 // 21, not 120 // 21.
        assert result.metadata["n_periods_finite"] == n
        assert result.metadata["n_periods"] == n - 21
        assert result.metadata["n_periods_effective"] == (n - 21) // 21
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in result.warning_codes
        assert (
            WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE.value in result.warning_codes
        )

    def test_same_length_at_horizon_one_stays_clean(self):
        # h = 1 makes the effective and reported counts identical, so the gate
        # is exactly the pre-existing one. Both sit one below the finite-pair
        # count, which the AR(1) lag consumes.
        rng = np.random.default_rng(4)
        n = 120
        result = predictive_beta(
            _ts_panel(rng.normal(size=n), rng.normal(size=n)), overlap_periods=1
        )
        assert result.metadata["n_periods_effective"] == n - 1
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value not in result.warning_codes
        assert (
            WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE.value
            not in result.warning_codes
        )

    def test_overlapping_horizon_echoes_the_known_size_warning(self):
        rng = np.random.default_rng(41)
        with pytest.warns(UserWarning) as caught:
            result = predictive_beta(
                _ts_panel(rng.normal(size=240), rng.normal(size=240)),
                overlap_periods=5,
            )
        overlap_warning = next(
            str(item.message)
            for item in caught
            if WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE.value in str(item.message)
        )
        assert "Amihud-Hurvich-adjusted HAR test" in overlap_warning
        assert "7.5-14.5%" in overlap_warning
        assert (
            WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE.value in result.warning_codes
        )

    def test_persistent_residuals_flag_serial_correlation(self):
        # Returns are an AR(1) with phi = 0.8 and the predictor is independent
        # noise, so the regression residuals inherit the persistence.
        rng = np.random.default_rng(5)
        n = 400
        y = np.zeros(n)
        eps = rng.normal(size=n)
        for i in range(1, n):
            y[i] = 0.8 * y[i - 1] + eps[i]
        x = rng.normal(size=n)
        result = predictive_beta(_ts_panel(x, y), overlap_periods=1)
        assert result.metadata["residual_lag1_autocorr"] > 0.3
        assert WarningCode.SERIAL_CORRELATION_DETECTED.value in result.warning_codes

    def test_overlap_alone_does_not_flag_serial_correlation(self):
        # Overlapping forward returns make the raw residuals an MA(h-1) by
        # construction, which the h-1 Bartlett floor already absorbs. The
        # screen reads the strided residuals, so a clean series at h=21 stays
        # silent rather than flagging every long-horizon run.
        rng = np.random.default_rng(8)
        n = 2500
        d = pl.DataFrame(
            {
                "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "asset_id": ["A"] * n,
                "factor": rng.normal(size=n),
                "price": 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n)),
            }
        )
        panel = fx.preprocess.compute_forward_return(d, forward_periods=21)
        result = predictive_beta(panel, overlap_periods=21)
        assert WarningCode.SERIAL_CORRELATION_DETECTED.value not in result.warning_codes

    def test_iid_residuals_do_not_flag_serial_correlation(self):
        rng = np.random.default_rng(6)
        n = 400
        result = predictive_beta(
            _ts_panel(rng.normal(size=n), rng.normal(size=n)), overlap_periods=1
        )
        assert WarningCode.SERIAL_CORRELATION_DETECTED.value not in result.warning_codes

    def test_declaring_the_code_stops_the_echo_but_keeps_the_record(self):
        import warnings

        rng = np.random.default_rng(7)
        n = 120
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = predictive_beta(
                _ts_panel(rng.normal(size=n), rng.normal(size=n)),
                overlap_periods=21,
                expected_warnings=(
                    "overlapping_predictive_inference",
                    "unreliable_se_short_periods",
                ),
            )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in result.warning_codes
        assert (
            WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE.value in result.warning_codes
        )
