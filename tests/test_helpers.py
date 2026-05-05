"""Tests for factrix.metrics._helpers."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from factrix.metrics._helpers import (
    TIE_RATIO_WARN_THRESHOLD,
    _assign_quantile_groups,
    _compute_tie_ratio,
    _sample_non_overlapping,
    _warn_high_tie_ratio,
)


class TestSampleNonOverlapping:
    def test_every_1_keeps_all(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        df = pl.DataFrame({"date": dates, "v": range(10)}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = _sample_non_overlapping(df, forward_periods=1)
        assert len(result) == 10

    def test_every_3(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        df = pl.DataFrame({"date": dates, "v": range(10)}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = _sample_non_overlapping(df, forward_periods=3)
        # 10 unique dates, every 3rd → indices [0,3,6,9] = 4 dates
        assert result["date"].n_unique() == 4

    def test_preserves_all_assets_on_sampled_dates(self, tiny_panel):
        result = _sample_non_overlapping(tiny_panel, forward_periods=1)
        for dt in result["date"].unique():
            n = result.filter(pl.col("date") == dt)["asset_id"].n_unique()
            assert n == 5


class TestAssignQuantileGroups:
    def test_5_groups(self, tiny_panel):
        one_date = tiny_panel.filter(pl.col("date") == tiny_panel["date"][0])
        result = _assign_quantile_groups(one_date, n_groups=5)
        groups = result["_group"].sort().to_list()
        assert groups == [0, 1, 2, 3, 4]

    def test_per_date_independent(self, tiny_panel):
        result = _assign_quantile_groups(tiny_panel, n_groups=5)
        for dt in result["date"].unique():
            chunk = result.filter(pl.col("date") == dt)
            assert set(chunk["_group"].to_list()) == {0, 1, 2, 3, 4}

    def test_ordinal_splits_tied_values_across_groups(self):
        # 6 assets all sharing factor = 1.0. Ordinal ranks by row order,
        # so each gets a unique rank → balanced groups spanning {0..2}.
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2024, 1, 1)] * 6, dtype=pl.Datetime("ms")),
                "asset_id": [f"a{i}" for i in range(6)],
                "factor": [1.0] * 6,
            }
        )
        result = _assign_quantile_groups(df, n_groups=3, tie_policy="ordinal")
        # ordinal → 6 distinct ranks over 3 groups → two per bucket
        assert sorted(result["_group"].to_list()) == [0, 0, 1, 1, 2, 2]

    def test_average_keeps_tied_values_in_same_group(self):
        # Same input: all tied. Average gives every row the same rank,
        # so the bucket formula puts them all in the same bucket.
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2024, 1, 1)] * 6, dtype=pl.Datetime("ms")),
                "asset_id": [f"a{i}" for i in range(6)],
                "factor": [1.0] * 6,
            }
        )
        result = _assign_quantile_groups(df, n_groups=3, tie_policy="average")
        # average rank = 3.5 for all → ((3.5-1)*3/6).cast(Int32) = 1 → all same
        groups = set(result["_group"].to_list())
        assert len(groups) == 1, (
            f"average tie_policy should keep tied values in one bucket, got {groups}"
        )


class TestComputeTieRatio:
    def test_unique_values_returns_zero(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2024, 1, 1)] * 5, dtype=pl.Datetime("ms")),
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        assert _compute_tie_ratio(df) == 0.0

    def test_all_tied_returns_near_one(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2024, 1, 1)] * 5, dtype=pl.Datetime("ms")),
                "factor": [1.0] * 5,
            }
        )
        # 5 rows, 1 unique → 1 - 1/5 = 0.8
        assert _compute_tie_ratio(df) == pytest.approx(0.8)

    def test_empty_returns_nan(self):
        df = pl.DataFrame(
            schema={
                "date": pl.Datetime("ms"),
                "factor": pl.Float64,
            }
        )
        assert math.isnan(_compute_tie_ratio(df))

    def test_median_across_dates(self):
        # Day 1: all tied (ratio 1.0 - 1/5 = 0.8). Day 2: all unique (0.0).
        # Median = (0.8 + 0.0) / 2 = 0.4.
        df = pl.DataFrame(
            {
                "date": pl.Series(
                    [datetime(2024, 1, 1)] * 5 + [datetime(2024, 1, 2)] * 5,
                    dtype=pl.Datetime("ms"),
                ),
                "factor": [1.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        assert _compute_tie_ratio(df) == pytest.approx(0.4)


class TestWarnHighTieRatio:
    def test_warns_when_above_threshold(self, recwarn):
        _warn_high_tie_ratio(0.5, "quantile_spread", "ordinal")
        hits = [
            w
            for w in recwarn.list
            if issubclass(w.category, UserWarning)
            and "tie_ratio=0.500" in str(w.message)
            and "quantile_spread" in str(w.message)
        ]
        assert len(hits) == 1

    def test_silent_below_threshold(self, recwarn):
        _warn_high_tie_ratio(
            TIE_RATIO_WARN_THRESHOLD - 0.01,
            "quantile_spread",
            "ordinal",
        )
        assert not recwarn.list

    def test_silent_on_average_policy(self, recwarn):
        # "average" already handles ties honestly — warning would be noise.
        _warn_high_tie_ratio(0.9, "quantile_spread", "average")
        assert not recwarn.list

    def test_silent_on_nan(self, recwarn):
        _warn_high_tie_ratio(float("nan"), "quantile_spread", "ordinal")
        assert not recwarn.list
