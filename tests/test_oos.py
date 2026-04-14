"""Tests for factorlib.tools.series.oos."""

import pytest

from factorlib.tools.series.oos import SplitDetail, multi_split_oos_decay


class TestMultiSplitOOSDecay:
    def test_stable_series_passes(self, ic_series_positive):
        result = multi_split_oos_decay(ic_series_positive)
        assert result.status == "PASS"
        assert result.decay_ratio > 0.5
        assert result.sign_flipped is False

    def test_sign_flip_vetoed(self, ic_series_sign_flip):
        result = multi_split_oos_decay(ic_series_sign_flip)
        assert result.status == "VETOED"
        assert result.sign_flipped is True

    def test_insufficient_data(self):
        import polars as pl
        from datetime import datetime, timedelta

        # Only 6 rows — below MIN_OOS_PERIODS * 2 = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
        series = pl.DataFrame({"date": dates, "value": [0.01] * 6}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = multi_split_oos_decay(series)
        assert result.status == "VETOED"
        assert result.decay_ratio == 0.0

    def test_custom_single_split(self, ic_series_positive):
        result = multi_split_oos_decay(ic_series_positive, splits=[(0.5, 0.5)])
        assert len(result.per_split) == 1

    def test_decay_below_threshold_vetoed(self):
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
        result = multi_split_oos_decay(series, decay_threshold=0.5)
        # OOS mean / IS mean ≈ 0.01/0.10 = 0.1 < 0.5
        assert result.status == "VETOED"
        assert result.sign_flipped is False

    def test_median_not_mean(self):
        """Verify decay_ratio is median of per-split ratios, not mean."""
        import polars as pl
        from datetime import datetime, timedelta
        import numpy as np

        # Construct series where 3 splits give different decays
        rng = np.random.default_rng(123)
        values = rng.normal(0.05, 0.02, 60)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        series = pl.DataFrame({"date": dates, "value": values}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = multi_split_oos_decay(series)
        ratios = sorted(s.decay_ratio for s in result.per_split)
        # Median of 3 = middle element
        assert result.decay_ratio == ratios[1]


class TestSplitDetail:
    def test_oos_ratio_property(self):
        detail = SplitDetail(
            is_ratio=0.7,
            mean_is=0.05,
            mean_oos=0.04,
            decay_ratio=0.8,
            sign_flipped=False,
        )
        assert detail.oos_ratio == pytest.approx(0.3)
