"""Tests for factrix.metrics.monotonicity."""

import math
import pytest

from factrix.metrics.monotonicity import monotonicity


class TestComputeMonotonicity:
    def test_perfect_monotonic(self, noisy_panel):
        # WHY: tiny_panel only has 3 dates, < MIN_MONOTONICITY_PERIODS after sampling
        # Use noisy_panel (20 dates × 30 assets) with perfect factor-return alignment
        import polars as pl
        perfect = noisy_panel.with_columns(
            pl.col("factor").rank(method="average").over("date").alias("forward_return")
        )
        result = monotonicity(perfect, forward_periods=1, n_groups=5)
        assert result.value == pytest.approx(1.0)

    def test_inverse_monotonic(self, noisy_panel):
        import polars as pl
        inverted = noisy_panel.with_columns(
            (-pl.col("factor").rank(method="average").over("date")).alias("forward_return")
        )
        result = monotonicity(inverted, forward_periods=1, n_groups=5)
        assert result.value == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(-1.0)

    def test_insufficient_periods(self):
        import polars as pl
        from datetime import datetime

        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5,
            "asset_id": ["A", "B", "C", "D", "E"],
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = monotonicity(df, forward_periods=1, n_groups=5)
        # Only 1 date < MIN_MONOTONICITY_PERIODS=5
        assert math.isnan(result.value)
        assert result.significance == ""
