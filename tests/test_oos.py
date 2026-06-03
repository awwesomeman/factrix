"""Tests for factrix.metrics.oos_decay."""

import math

from factrix._results import MetricResult
from factrix.metrics.oos_decay import oos_decay


class TestOOSDecay:
    def test_stable_series_passes(self, ic_series_positive):
        result = oos_decay(ic_series_positive)
        assert result.metadata["status"] == "PASS"
        assert result.value > 0.5
        assert result.metadata["sign_flipped"] is False

    def test_sign_flip_vetoed(self, ic_series_sign_flip):
        result = oos_decay(ic_series_sign_flip)
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["sign_flipped"] is True

    def test_insufficient_data(self):
        from datetime import datetime, timedelta

        import polars as pl

        # Only 6 rows — below MIN_OOS_PERIODS * 2 = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
        series = pl.DataFrame({"date": dates, "value": [0.01] * 6}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = oos_decay(series)
        assert result.metadata["status"] == "VETOED"
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_oos_periods"

    def test_custom_is_ratio(self, ic_series_positive):
        result = oos_decay(ic_series_positive, is_ratio=0.5)
        assert result.metadata["is_ratio"] == 0.5

    def test_survival_below_threshold_vetoed(self):
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(99)
        # IS strong, OOS very weak
        is_vals = rng.normal(0.10, 0.01, 30)
        oos_vals = rng.normal(0.01, 0.01, 20)
        values = np.concatenate([is_vals, oos_vals])
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        series = pl.DataFrame({"date": dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = oos_decay(series, is_ratio=0.6, survival_threshold=0.5)
        # OOS mean / IS mean ≈ 0.01/0.10 = 0.1 < 0.5
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["sign_flipped"] is False

    def test_returns_metric_output(self, ic_series_positive):
        """Single-contract check: oos_decay returns MetricResult."""

        result = oos_decay(ic_series_positive)
        assert isinstance(result, MetricResult)
        assert result.stat is None  # descriptive, not hypothesis test
        # Descriptive-only: no p_value emitted (would invite mis-routing
        # the diagnostic into BHY / gate logic).
        assert "p_value" not in result.metadata

    def test_metadata_shape(self, ic_series_positive):
        """metadata carries the single-split fields."""
        result = oos_decay(ic_series_positive)
        assert set(result.metadata.keys()) >= {
            "sign_flipped",
            "status",
            "is_ratio",
            "mean_is",
            "mean_oos",
            "survival_threshold",
        }
