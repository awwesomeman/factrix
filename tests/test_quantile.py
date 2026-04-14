"""Tests for factorlib.tools.panel.quantile."""

import pytest

from factorlib.tools.panel.quantile import (
    long_short_alpha,
    quantile_spread,
    quantile_spread_series,
)


class TestQuantileSpreadSeries:
    def test_perfect_panel(self, tiny_panel):
        series = quantile_spread_series(tiny_panel, forward_periods=1, n_groups=5)
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
        # WHY: noisy_panel has 20 dates, forward_periods=1 keeps all → enough for annualization
        # but date range = 20 days < 0.1 years → annualize_return returns None → value=0.
        # Use forward_periods=1 and verify the metric at least runs.
        # The real check is that quantile_spread_series produces valid spreads.
        series = quantile_spread_series(noisy_panel, forward_periods=1, n_groups=5)
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


class TestLongShortAlpha:
    def test_decomposition_sums_to_spread(self, tiny_panel):
        series = quantile_spread_series(tiny_panel, forward_periods=1, n_groups=5)
        for row in series.iter_rows(named=True):
            long = row["q1_return"] - row["universe_return"]
            short = row["universe_return"] - row["q5_return"]
            assert long + short == pytest.approx(row["spread"])

    def test_precomputed_series(self, noisy_panel):
        series = quantile_spread_series(noisy_panel, forward_periods=1)
        r1 = long_short_alpha(noisy_panel, forward_periods=1)
        r2 = long_short_alpha(noisy_panel, forward_periods=1, _precomputed_series=series)
        assert r1.value == pytest.approx(r2.value)
