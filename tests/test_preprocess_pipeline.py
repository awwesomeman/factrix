"""Tests for factorlib.preprocess.pipeline."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.preprocess.pipeline import preprocess_cs_factor


def _make_raw_data(n_dates: int = 30, n_assets: int = 10):
    import numpy as np
    rng = np.random.default_rng(42)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for i in range(n_assets):
            rows.append({
                "date": d,
                "asset_id": f"T{i}",
                "price": 100.0 + rng.standard_normal() * 10,
                "factor": rng.standard_normal(),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestRunPreprocessing:
    def test_output_schema(self):
        raw = _make_raw_data()
        result = preprocess_cs_factor(raw, forward_periods=5)
        assert result.columns == [
            "date", "asset_id", "factor_raw", "factor",
            "forward_return", "abnormal_return", "price",
            "_fl_forward_periods",
        ]

    def test_forward_periods_marker_embedded(self):
        raw = _make_raw_data()
        result = preprocess_cs_factor(raw, forward_periods=7)
        assert result["_fl_forward_periods"].unique().to_list() == [7]

    def test_output_dtypes(self):
        raw = _make_raw_data()
        result = preprocess_cs_factor(raw, forward_periods=5)
        assert result["date"].dtype == raw["date"].dtype
        assert result["asset_id"].dtype == pl.String

    def test_no_nulls(self):
        raw = _make_raw_data()
        result = preprocess_cs_factor(raw, forward_periods=5)
        for col in ["factor", "forward_return", "abnormal_return"]:
            assert result[col].null_count() == 0

    def test_no_nans(self):
        raw = _make_raw_data()
        result = preprocess_cs_factor(raw, forward_periods=5)
        nan_count = result["factor"].is_nan().sum()
        assert nan_count == 0

    def test_factor_raw_preserved(self):
        raw = _make_raw_data()
        result = preprocess_cs_factor(raw, forward_periods=5)
        # factor_raw should exist and not be all zeros
        assert result["factor_raw"].std() > 0

    def test_config_overrides_kwargs(self):
        from factorlib.config import CrossSectionalConfig
        raw = _make_raw_data()
        config = CrossSectionalConfig(forward_periods=3, mad_n=5.0)
        result_cfg = preprocess_cs_factor(raw, config=config)
        result_kw = preprocess_cs_factor(raw, forward_periods=3, mad_n=5.0)
        # config path and keyword path should produce identical results
        assert result_cfg.equals(result_kw)
