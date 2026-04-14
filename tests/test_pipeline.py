"""Tests for factorlib.preprocessing.pipeline."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.preprocessing.pipeline import run_preprocessing


def _make_raw_data(n_dates: int = 30, n_assets: int = 10):
    import numpy as np
    rng = np.random.default_rng(42)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for i in range(n_assets):
            rows.append({
                "datetime": d,
                "ticker": f"T{i}",
                "close": 100.0 + rng.standard_normal() * 10,
                "factor": rng.standard_normal(),
            })
    return pl.DataFrame(rows).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))


class TestRunPreprocessing:
    def test_output_schema(self):
        raw = _make_raw_data()
        result = run_preprocessing(raw, forward_periods=5)
        assert result.columns == [
            "date", "asset_id", "factor_raw", "factor",
            "forward_return", "abnormal_return",
        ]

    def test_output_dtypes(self):
        raw = _make_raw_data()
        result = run_preprocessing(raw, forward_periods=5)
        assert result["date"].dtype == pl.Datetime("ms")
        assert result["asset_id"].dtype == pl.String

    def test_no_nulls(self):
        raw = _make_raw_data()
        result = run_preprocessing(raw, forward_periods=5)
        for col in ["factor", "forward_return", "abnormal_return"]:
            assert result[col].null_count() == 0

    def test_no_nans(self):
        raw = _make_raw_data()
        result = run_preprocessing(raw, forward_periods=5)
        nan_count = result["factor"].is_nan().sum()
        assert nan_count == 0

    def test_factor_raw_preserved(self):
        raw = _make_raw_data()
        result = run_preprocessing(raw, forward_periods=5)
        # factor_raw should exist and not be all zeros
        assert result["factor_raw"].std() > 0
