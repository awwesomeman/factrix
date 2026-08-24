"""Tests for factrix.metrics.positive_rate."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
from factrix.metrics.positive_rate import positive_rate


def _make_series(values: list[float]) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "value": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )


class TestComputePositiveRate:
    def test_all_positive(self):
        series = _make_series([0.01] * 20)
        result = positive_rate(series, forward_periods=1)
        assert result.value == pytest.approx(1.0)
        # stat is the exact test's sufficient statistic: the hit count.
        assert result.stat == 20
        assert result.metadata["stat_type"] == "binomial_hits"

    def test_all_negative(self):
        series = _make_series([-0.01] * 20)
        result = positive_rate(series, forward_periods=1)
        assert result.value == pytest.approx(0.0)
        assert result.stat == 0
        assert result.p_value == pytest.approx(2 * 0.5**20)

    def test_half_and_half(self):
        values = [0.01] * 10 + [-0.01] * 10
        series = _make_series(values)
        result = positive_rate(series, forward_periods=1)
        assert result.value == pytest.approx(0.5)
        assert result.stat == 10
        assert result.p_value == pytest.approx(1.0)

    def test_insufficient_data(self):
        series = _make_series([0.01] * 5)  # < MIN_SERIES_PERIODS_HARD=10
        result = positive_rate(series, forward_periods=1)
        assert math.isnan(result.value)
        assert result.p_value is None or result.p_value >= 0.10

    def test_small_n_uses_exact_binomial(self):
        # n=15 → below _BINOMIAL_EXACT_CUTOFF=20. All hits → exact p is
        # 2 * 0.5**15 ≈ 6.1e-5, whereas the normal approx gives ≈ 6.3e-5
        # for z = √15. Any difference confirms the exact branch.
        series = _make_series([0.01] * 15)
        result = positive_rate(series, forward_periods=1)
        assert result.metadata["method"] == "binomial exact test"
        # Exact p for 15/15 successes under H₀: p=0.5 is 2 * 0.5**15.
        assert result.p_value == pytest.approx(2 * 0.5**15)

    def test_large_n_still_exact(self):
        """No normal-approximation branch: the exact test runs at every n and
        the p matches scipy.stats.binomtest (the old score-test branch was
        anti-conservative by ~35-40% in the 0.02-0.15 band)."""
        from scipy.stats import binomtest

        series = _make_series([0.01] * 115 + [-0.01] * 85)
        result = positive_rate(series, forward_periods=1)
        assert result.metadata["method"] == "binomial exact test"
        assert result.p_value == pytest.approx(binomtest(115, 200, 0.5).pvalue)


class TestNaNHandling:
    def test_nan_is_dropped_not_counted_as_miss(self):
        """A float NaN is not a null for polars ``drop_nulls``; it would count
        as a non-hit in ``value > 0`` and bias the rate toward 0."""
        base = [0.01] * 30
        dirty = _make_series(base + [float("nan")] * 10)
        clean = _make_series(base)
        r_dirty = positive_rate(dirty, forward_periods=1)
        r_clean = positive_rate(clean, forward_periods=1)
        assert r_dirty.value == r_clean.value == 1.0
        assert r_dirty.n_obs == r_clean.n_obs == 30
        assert r_dirty.p_value == pytest.approx(r_clean.p_value)
        assert r_dirty.metadata["dropped_periods"] == 10
