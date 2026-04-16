"""Tests for factorlib.metrics.quantile."""

import pytest

from factorlib.metrics.quantile import (
    quantile_spread,
    compute_spread_series,
)


class TestQuantileSpreadSeries:
    def test_perfect_panel(self, tiny_panel):
        series = compute_spread_series(tiny_panel, forward_periods=1, n_groups=5)
        assert "spread" in series.columns
        assert "q1_return" in series.columns
        assert "q5_return" in series.columns
        # factor=[1..5], return=[0.01..0.05], 5 groups → q1=0.05, q5=0.01
        for row in series.iter_rows(named=True):
            assert row["q1_return"] == pytest.approx(0.05)
            assert row["q5_return"] == pytest.approx(0.01)
            assert row["spread"] == pytest.approx(0.04)


class TestQuantileSpread:
    def test_noisy_panel(self, noisy_panel):
        series = compute_spread_series(noisy_panel, forward_periods=1, n_groups=5)
        assert len(series) >= 5
        assert series["spread"].null_count() == 0

    def test_insufficient_periods(self):
        import polars as pl
        from datetime import datetime

        # 2 dates < MIN_PORTFOLIO_PERIODS=5
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5 + [datetime(2024, 1, 2)] * 5,
            "asset_id": ["A", "B", "C", "D", "E"] * 2,
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05] * 2,
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = quantile_spread(df, forward_periods=1, n_groups=5)
        assert result.value == 0.0

    def test_decomposition_in_metadata(self, tiny_panel):
        """spread = long_alpha + short_alpha (per-period)."""
        series = compute_spread_series(tiny_panel, forward_periods=1, n_groups=5)
        for row in series.iter_rows(named=True):
            long = row["q1_return"] - row["universe_return"]
            short = row["universe_return"] - row["q5_return"]
            assert long + short == pytest.approx(row["spread"])

    def test_metadata_has_long_short(self, noisy_panel):
        result = quantile_spread(noisy_panel, forward_periods=1, n_groups=5)
        if result.value != 0.0:
            assert "long_alpha" in result.metadata
            assert "short_alpha" in result.metadata
            assert "long_stat" in result.metadata
            assert "short_stat" in result.metadata
            assert "p_value" in result.metadata
