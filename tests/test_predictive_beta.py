"""Tests for single-asset dense ``predictive_beta``."""

from __future__ import annotations

import math
from datetime import date, timedelta

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

        result = predictive_beta(_ts_panel(x, y), forward_periods=5)
        reference_beta = np.polyfit(x, y, 1)[0]

        assert result.value == pytest.approx(reference_beta)
        assert result.stat > 0
        assert result.p_value < 0.01
        assert result.n_obs == 240
        assert result.n_obs_axis == "periods"
        assert result.metadata["h0"] == "beta=0"
        assert result.metadata["newey_west_lags"] == _resolve_nw_lags(240, None, 5)

    def test_independent_factor_not_significant(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=300)
        y = rng.normal(size=300)
        result = predictive_beta(_ts_panel(x, y), forward_periods=1)
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

        result = predictive_beta(panel, forward_periods=1)
        assert result.n_obs == 72


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
        result = predictive_beta(_ts_panel(x, y), forward_periods=1)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "degenerate_factor_variance"
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
                forward_periods=1,
            )
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS.value in result.warning_codes


class TestPredictiveBetaDispatch:
    def _single_asset_forward_panel(self) -> pl.DataFrame:
        raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=160, seed=10)
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
            fx.datasets.make_cs_panel(n_assets=20, n_dates=80, seed=11),
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
