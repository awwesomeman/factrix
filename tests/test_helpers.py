"""Tests for factorlib.tools._helpers."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.tools._helpers import (
    annualize_return,
    assign_quantile_groups,
    sample_non_overlapping,
)


class TestSampleNonOverlapping:
    def test_every_1_keeps_all(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        df = pl.DataFrame({"date": dates, "v": range(10)}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = sample_non_overlapping(df, forward_periods=1)
        assert len(result) == 10

    def test_every_3(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        df = pl.DataFrame({"date": dates, "v": range(10)}).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        result = sample_non_overlapping(df, forward_periods=3)
        # 10 unique dates, every 3rd → indices [0,3,6,9] = 4 dates
        assert result["date"].n_unique() == 4

    def test_preserves_all_assets_on_sampled_dates(self, tiny_panel):
        result = sample_non_overlapping(tiny_panel, forward_periods=1)
        for dt in result["date"].unique():
            n = result.filter(pl.col("date") == dt)["asset_id"].n_unique()
            assert n == 5


class TestAssignQuantileGroups:
    def test_5_groups(self, tiny_panel):
        one_date = tiny_panel.filter(pl.col("date") == tiny_panel["date"][0])
        result = assign_quantile_groups(one_date, n_groups=5)
        groups = result["_group"].sort().to_list()
        assert groups == [0, 1, 2, 3, 4]

    def test_per_date_independent(self, tiny_panel):
        result = assign_quantile_groups(tiny_panel, n_groups=5)
        for dt in result["date"].unique():
            chunk = result.filter(pl.col("date") == dt)
            assert set(chunk["_group"].to_list()) == {0, 1, 2, 3, 4}


class TestAnnualizeReturn:
    def test_one_year(self):
        n = 252
        arr = np.full(n, 0.001)  # 0.1% per period
        dates = pl.Series([datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)])
        result = annualize_return(arr, dates)
        assert result is not None
        assert result > 0

    def test_short_range_returns_none(self):
        arr = np.array([0.01, 0.02])
        dates = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2)])
        assert annualize_return(arr, dates) is None

    def test_negative_returns(self):
        n = 100
        arr = np.full(n, -0.001)
        dates = pl.Series([datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)])
        result = annualize_return(arr, dates)
        assert result is not None
        assert result < 0
