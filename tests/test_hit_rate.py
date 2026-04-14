"""Tests for factorlib.tools.series.hit_rate."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.tools.series.hit_rate import compute_hit_rate


def _make_series(values: list[float]) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "value": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestComputeHitRate:
    def test_all_positive(self):
        series = _make_series([0.01] * 20)
        result = compute_hit_rate(series, forward_periods=1)
        assert result.value == pytest.approx(1.0)
        assert result.t_stat > 0

    def test_all_negative(self):
        series = _make_series([-0.01] * 20)
        result = compute_hit_rate(series, forward_periods=1)
        assert result.value == pytest.approx(0.0)
        assert result.t_stat < 0

    def test_half_and_half(self):
        values = [0.01] * 10 + [-0.01] * 10
        series = _make_series(values)
        result = compute_hit_rate(series, forward_periods=1)
        assert result.value == pytest.approx(0.5)
        assert abs(result.t_stat) < 0.5

    def test_insufficient_data(self):
        series = _make_series([0.01] * 5)  # < MIN_IC_PERIODS=10
        result = compute_hit_rate(series, forward_periods=1)
        assert result.value == 0.0
        assert result.significance == "○"
