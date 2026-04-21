"""Tests for factrix.preprocess.normalize."""

import polars as pl
import pytest
from datetime import datetime

from factrix.preprocess.normalize import cross_sectional_zscore, mad_winsorize


class TestMADWinsorize:
    def test_clips_outlier(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5,
            "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=3.0)
        assert result["factor"].max() < 100.0

    def test_noop_when_disabled(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5,
            "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=0)
        assert result["factor"].to_list() == df["factor"].to_list()

    def test_per_date(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 3 + [datetime(2024, 1, 2)] * 3,
            "factor": [1.0, 2.0, 100.0, 10.0, 20.0, 1000.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=3.0)
        d1 = result.filter(pl.col("date") == datetime(2024, 1, 1))["factor"].max()
        d2 = result.filter(pl.col("date") == datetime(2024, 1, 2))["factor"].max()
        assert d1 < 100.0
        assert d2 < 1000.0


class TestCrossSectionalZScore:
    def test_zero_mad(self):
        """All same value → MAD=0 → fill_nan(0.0)."""
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5,
            "factor": [3.0, 3.0, 3.0, 3.0, 3.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = cross_sectional_zscore(df)
        for v in result["factor_zscore"].to_list():
            assert v == 0.0

    def test_output_column(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5,
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = cross_sectional_zscore(df)
        assert "factor_zscore" in result.columns

    def test_median_near_zero(self):
        """After z-score, median should be near 0."""
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 100,
            "factor": list(range(100)),
        }).with_columns(
            pl.col("date").cast(pl.Datetime("ms")),
            pl.col("factor").cast(pl.Float64),
        )
        result = cross_sectional_zscore(df)
        median = result["factor_zscore"].median()
        assert abs(median) < 0.1
