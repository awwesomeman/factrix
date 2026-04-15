"""Tests for factorlib.tools.series.trend."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.tools.series.trend import ic_trend


def _make_series(values: list[float]) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "value": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestTheilSenSlope:
    def test_perfect_uptrend(self):
        values = [0.01 * i for i in range(20)]
        result = ic_trend(_make_series(values))
        assert result.value > 0
        assert result.metadata["ci_excludes_zero"] is True

    def test_flat(self):
        values = [0.05] * 20
        result = ic_trend(_make_series(values))
        assert result.value == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        values = [0.01] * 5  # < 10
        result = ic_trend(_make_series(values))
        assert result.value == 0.0
        assert result.significance == ""

    def test_negative_slope(self):
        values = [0.10 - 0.005 * i for i in range(20)]
        result = ic_trend(_make_series(values))
        assert result.value < 0
