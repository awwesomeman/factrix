"""Tests for factorlib.metrics.oos."""

import pytest

from factorlib.metrics.oos import SplitDetail, multi_split_oos_decay


class TestMultiSplitOOSDecay:
    def test_stable_series_passes(self, ic_series_positive):
        result = multi_split_oos_decay(ic_series_positive)
        assert result.name == "oos_decay"
        assert result.metadata["status"] == "PASS"
        assert result.value > 0.5
        assert result.metadata["sign_flipped"] is False

    def test_sign_flip_vetoed(self, ic_series_sign_flip):
        result = multi_split_oos_decay(ic_series_sign_flip)
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["sign_flipped"] is True

    def test_insufficient_data(self):
        import polars as pl
        from datetime import datetime, timedelta

        # Only 6 rows — below MIN_OOS_PERIODS * 2 = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
        series = pl.DataFrame({"date": dates, "value": [0.01] * 6}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = multi_split_oos_decay(series)
        assert result.metadata["status"] == "VETOED"
        assert result.value == 0.0
        assert result.metadata["reason"] == "insufficient_oos_periods"

    def test_custom_single_split(self, ic_series_positive):
        result = multi_split_oos_decay(ic_series_positive, splits=[(0.5, 0.5)])
        assert len(result.metadata["per_split"]) == 1

    def test_survival_below_threshold_vetoed(self):
        import polars as pl
        from datetime import datetime, timedelta
        import numpy as np

        rng = np.random.default_rng(99)
        # IS strong, OOS very weak
        is_vals = rng.normal(0.10, 0.01, 30)
        oos_vals = rng.normal(0.01, 0.01, 20)
        values = np.concatenate([is_vals, oos_vals])
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        series = pl.DataFrame({"date": dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = multi_split_oos_decay(series, survival_threshold=0.5)
        # OOS mean / IS mean ≈ 0.01/0.10 = 0.1 < 0.5
        assert result.metadata["status"] == "VETOED"
        assert result.metadata["sign_flipped"] is False

    def test_median_not_mean(self):
        """Verify value is median of per-split ratios, not mean."""
        import polars as pl
        from datetime import datetime, timedelta
        import numpy as np

        rng = np.random.default_rng(123)
        values = rng.normal(0.05, 0.02, 60)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        series = pl.DataFrame({"date": dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = multi_split_oos_decay(series)
        ratios = sorted(sd["survival_ratio"] for sd in result.metadata["per_split"])
        assert result.value == ratios[1]

    def test_returns_metric_output(self, ic_series_positive):
        """Single-contract check: oos_decay returns MetricOutput, not OOSResult."""
        from factorlib._types import MetricOutput
        result = multi_split_oos_decay(ic_series_positive)
        assert isinstance(result, MetricOutput)
        assert result.name == "oos_decay"
        assert result.stat is None  # descriptive, not hypothesis test
        assert result.metadata["p_value"] == 1.0
        assert result.metadata["method"] == "multi-split OOS decay"

    def test_per_split_shape(self, ic_series_positive):
        """per_split entries are serializable dicts, not dataclass instances."""
        result = multi_split_oos_decay(ic_series_positive)
        for entry in result.metadata["per_split"]:
            assert isinstance(entry, dict)
            assert set(entry.keys()) == {
                "is_ratio", "mean_is", "mean_oos", "survival_ratio", "sign_flipped",
            }


class TestSplitDetail:
    def test_oos_ratio_property(self):
        detail = SplitDetail(
            is_ratio=0.7,
            mean_is=0.05,
            mean_oos=0.04,
            survival_ratio=0.8,
            sign_flipped=False,
        )
        assert detail.oos_ratio == pytest.approx(0.3)

    def test_to_dict(self):
        detail = SplitDetail(
            is_ratio=0.7,
            mean_is=0.05,
            mean_oos=0.04,
            survival_ratio=0.8,
            sign_flipped=False,
        )
        assert detail.to_dict() == {
            "is_ratio": 0.7,
            "mean_is": 0.05,
            "mean_oos": 0.04,
            "survival_ratio": 0.8,
            "sign_flipped": False,
        }
