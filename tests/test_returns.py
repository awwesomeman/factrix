"""Tests for factorlib.preprocessing.returns."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.preprocessing.returns import (
    compute_abnormal_return,
    compute_forward_return,
    winsorize_forward_return,
)


def _make_price_data():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    return pl.DataFrame({
        "datetime": dates * 2,
        "ticker": ["A"] * 5 + ["B"] * 5,
        "close": [100.0, 110.0, 121.0, 133.1, 146.41] + [200.0, 190.0, 180.5, 171.475, 162.9],
    }).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))


class TestComputeForwardReturn:
    def test_basic(self):
        df = _make_price_data()
        result = compute_forward_return(df, "datetime", "ticker", "close", forward_periods=1)
        # A: [0.10, 0.10, 0.10, 0.10] B: [-0.05, -0.05, -0.05, -0.05]
        a_returns = result.filter(pl.col("ticker") == "A")["forward_return"].to_list()
        assert a_returns[0] == pytest.approx(0.10)

    def test_drops_nulls(self):
        df = _make_price_data()
        result = compute_forward_return(df, "datetime", "ticker", "close", forward_periods=1)
        assert result["forward_return"].null_count() == 0
        # Each asset loses 1 row (last), so 5*2 - 2 = 8
        assert len(result) == 8

    def test_per_asset_independent(self):
        df = _make_price_data()
        result = compute_forward_return(df, "datetime", "ticker", "close", forward_periods=1)
        a_ret = result.filter(pl.col("ticker") == "A")["forward_return"][0]
        b_ret = result.filter(pl.col("ticker") == "B")["forward_return"][0]
        # A goes up, B goes down
        assert a_ret > 0
        assert b_ret < 0


class TestWinsorizeForwardReturn:
    def test_noop(self):
        df = pl.DataFrame({
            "datetime": [datetime(2024, 1, 1)] * 5,
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        }).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
        result = winsorize_forward_return(df, "datetime", lower=0.0, upper=1.0)
        assert result["forward_return"].to_list() == df["forward_return"].to_list()

    def test_clips_extreme(self):
        df = pl.DataFrame({
            "datetime": [datetime(2024, 1, 1)] * 10,
            "forward_return": [0.01] * 8 + [0.50, -0.50],  # extremes
        }).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
        result = winsorize_forward_return(df, "datetime", lower=0.1, upper=0.9)
        vals = result["forward_return"].to_list()
        assert max(vals) < 0.50  # upper extreme clipped
        assert min(vals) > -0.50  # lower extreme clipped


class TestComputeAbnormalReturn:
    def test_zero_mean_per_date(self):
        df = pl.DataFrame({
            "datetime": [datetime(2024, 1, 1)] * 5,
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        }).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
        result = compute_abnormal_return(df, "datetime")
        mean_abnormal = result["abnormal_return"].mean()
        assert abs(mean_abnormal) < 1e-10

    def test_known_values(self):
        df = pl.DataFrame({
            "datetime": [datetime(2024, 1, 1)] * 3,
            "forward_return": [0.01, 0.02, 0.03],
        }).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
        result = compute_abnormal_return(df, "datetime")
        # mean = 0.02, abnormal = [-0.01, 0.0, 0.01]
        expected = [-0.01, 0.0, 0.01]
        actual = result["abnormal_return"].to_list()
        for a, e in zip(actual, expected):
            assert a == pytest.approx(e)
