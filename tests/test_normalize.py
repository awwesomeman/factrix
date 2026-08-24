"""Tests for factrix.preprocess.normalize."""

import math
from datetime import datetime

import polars as pl
import pytest
from factrix.preprocess.normalize import cross_sectional_zscore, mad_winsorize


class TestMADWinsorize:
    def test_clips_outlier(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=3.0)
        assert result["factor"].max() < 100.0

    def test_noop_when_disabled(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, 3.0, 4.0, 100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=0)
        assert result["factor"].to_list() == df["factor"].to_list()

    def test_per_date(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 3 + [datetime(2024, 1, 2)] * 3,
                "factor": [1.0, 2.0, 100.0, 10.0, 20.0, 1000.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = mad_winsorize(df, n_mad=3.0)
        d1 = result.filter(pl.col("date") == datetime(2024, 1, 1))["factor"].max()
        d2 = result.filter(pl.col("date") == datetime(2024, 1, 2))["factor"].max()
        assert d1 < 100.0
        assert d2 < 1000.0


class TestCrossSectionalZScore:
    def test_zero_mad(self):
        """All same value → MAD=0 → fill_nan(0.0)."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [3.0, 3.0, 3.0, 3.0, 3.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = cross_sectional_zscore(df)
        for v in result["factor_zscore"].to_list():
            assert v == 0.0

    def test_output_column(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = cross_sectional_zscore(df)
        assert "factor_zscore" in result.columns

    def test_median_near_zero(self):
        """After z-score, median should be near 0."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 100,
                "factor": list(range(100)),
            }
        ).with_columns(
            pl.col("date").cast(pl.Datetime("ms")),
            pl.col("factor").cast(pl.Float64),
        )
        result = cross_sectional_zscore(df)
        median = result["factor_zscore"].median()
        assert abs(median) < 0.1


class TestZeroMADFallback:
    """Regression: MAD == 0 (>50% ties) must not blow the scale up to inf."""

    def _bucketed(self) -> pl.DataFrame:
        # 4 ties + 1 outlier + 1 null → median 1.0, MAD 0.0, std(ddof=1) > 0.
        return pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 6,
                "factor": [1.0, 1.0, 1.0, 1.0, 2.0, None],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_zscore_is_finite_and_rank_preserving(self):
        z = cross_sectional_zscore(self._bucketed())["factor_zscore"].to_list()
        assert all(v is None or math.isfinite(v) for v in z)
        assert z[:4] == [0.0, 0.0, 0.0, 0.0]
        # std(ddof=1) of [1,1,1,1,2] = 0.4472 → (2 - 1) / 0.4472
        assert z[4] == pytest.approx(1.0 / 0.4472135954999579)

    def test_zscore_keeps_null_null(self):
        """`fill_null(0.0)` imputed missing factors to exactly the median."""
        z = cross_sectional_zscore(self._bucketed())["factor_zscore"].to_list()
        assert z[5] is None

    def test_zscore_nan_input_yields_null(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "factor": [1.0, 2.0, float("nan"), 4.0, 5.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        z = cross_sectional_zscore(df)["factor_zscore"].to_list()
        assert z[2] is None
        # The NaN must not poison the rest of the cross-section.
        assert all(math.isfinite(v) for i, v in enumerate(z) if i != 2)

    def test_zscore_constant_date_is_zero_not_nan(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 4,
                "factor": [3.0, 3.0, 3.0, None],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        z = cross_sectional_zscore(df)["factor_zscore"].to_list()
        assert z == [0.0, 0.0, 0.0, None]

    def test_winsorize_does_not_collapse_bucketed_factor(self):
        """MAD == 0 clipped every value to the median, destroying the factor."""
        out = mad_winsorize(self._bucketed(), n_mad=3.0)["factor"].to_list()
        assert out == [1.0, 1.0, 1.0, 1.0, 2.0, None]

    def test_winsorize_std_fallback_still_clips_far_outlier(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 12,
                "factor": [1.0] * 10 + [2.0, 100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        out = mad_winsorize(df, n_mad=3.0)["factor"].to_list()
        assert max(out) < 100.0
