"""Tests for factrix.metrics.hit_rate."""

import math
import polars as pl
import pytest
from datetime import datetime, timedelta

from factrix.metrics.hit_rate import hit_rate


def _make_series(values: list[float]) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "value": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestComputeHitRate:
    def test_all_positive(self):
        series = _make_series([0.01] * 20)
        result = hit_rate(series, forward_periods=1)
        assert result.value == pytest.approx(1.0)
        assert result.stat > 0

    def test_all_negative(self):
        series = _make_series([-0.01] * 20)
        result = hit_rate(series, forward_periods=1)
        assert result.value == pytest.approx(0.0)
        assert result.stat < 0

    def test_half_and_half(self):
        values = [0.01] * 10 + [-0.01] * 10
        series = _make_series(values)
        result = hit_rate(series, forward_periods=1)
        assert result.value == pytest.approx(0.5)
        assert abs(result.stat) < 0.5

    def test_insufficient_data(self):
        series = _make_series([0.01] * 5)  # < MIN_IC_PERIODS=10
        result = hit_rate(series, forward_periods=1)
        assert math.isnan(result.value)
        assert result.significance == ""

    def test_small_n_uses_exact_binomial(self):
        # n=15 → below _BINOMIAL_EXACT_CUTOFF=20. All hits → exact p is
        # 2 * 0.5**15 ≈ 6.1e-5, whereas the normal approx gives ≈ 6.3e-5
        # for z = √15. Any difference confirms the exact branch.
        series = _make_series([0.01] * 15)
        result = hit_rate(series, forward_periods=1)
        assert result.metadata["method"] == "binomial exact test"
        # Exact p for 15/15 successes under H₀: p=0.5 is 2 * 0.5**15.
        assert result.metadata["p_value"] == pytest.approx(2 * 0.5 ** 15)

    def test_large_n_uses_normal_approximation(self):
        series = _make_series([0.01] * 100 + [-0.01] * 100)
        result = hit_rate(series, forward_periods=1)
        assert (
            result.metadata["method"]
            == "binomial score test (normal approximation)"
        )
