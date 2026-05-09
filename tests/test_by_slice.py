"""Tests for factrix.metrics.by_slice — axis-agnostic Layer-A dispatcher."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._types import MetricOutput
from factrix.metrics import by_slice, ic
from factrix.metrics._slice import _slice_by_label


def _ic_series_with_label(
    n: int = 40, label_col: str = "regime", seed: int = 42
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    half = n // 2
    return pl.DataFrame(
        {
            "date": dates,
            "ic": rng.normal(0.05, 0.02, n),
            label_col: ["bull"] * half + ["bear"] * (n - half),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestSliceByLabel:
    def test_partitions_on_existing_column(self):
        df = _ic_series_with_label(20)
        out = _slice_by_label(df, "regime")
        assert set(out) == {"bull", "bear"}
        assert all(isinstance(v, pl.DataFrame) for v in out.values())
        assert sum(len(v) for v in out.values()) == 20

    def test_drops_label_column_from_partitions(self):
        df = _ic_series_with_label(20)
        out = _slice_by_label(df, "regime")
        for sub in out.values():
            assert "regime" not in sub.columns

    def test_missing_label_raises(self):
        df = _ic_series_with_label(10)
        with pytest.raises(ValueError, match="not found in df"):
            _slice_by_label(df, "sector")

    def test_empty_df_raises(self):
        df = _ic_series_with_label(0)
        with pytest.raises(ValueError, match="empty"):
            _slice_by_label(df, "regime")

    def test_null_label_values_raise(self):
        df = _ic_series_with_label(10).with_columns(
            pl.when(pl.int_range(0, 10) < 3)
            .then(None)
            .otherwise(pl.col("regime"))
            .alias("regime")
        )
        with pytest.raises(ValueError, match="contains nulls"):
            _slice_by_label(df, "regime")

    def test_numeric_label_stringified(self):
        df = _ic_series_with_label(20).with_columns(
            pl.Series("decile", [1, 2] * 10, dtype=pl.Int64)
        )
        out = _slice_by_label(df, "decile")
        assert set(out) == {"1", "2"}

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="polars DataFrame"):
            _slice_by_label([1, 2, 3], "regime")  # type: ignore[arg-type]


class TestBySlice:
    def test_returns_dict_per_slice(self):
        df = _ic_series_with_label(40)
        out = by_slice(ic, df, label="regime")
        assert isinstance(out, dict)
        assert set(out) == {"bull", "bear"}
        for v in out.values():
            assert isinstance(v, MetricOutput)
            assert v.name == "ic"

    def test_kwargs_forwarded(self):
        df = _ic_series_with_label(60)
        out = by_slice(ic, df, label="regime", forward_periods=3)
        for v in out.values():
            assert v.metadata.get("method", "").startswith("non-overlapping")

    def test_missing_label_raises(self):
        df = _ic_series_with_label(10)
        with pytest.raises(ValueError, match="not found in df"):
            by_slice(ic, df, label="sector")

    def test_arbitrary_label_column(self):
        """``label`` is just a column name — sector / market / anything works."""
        df = _ic_series_with_label(30, label_col="sector")
        out = by_slice(ic, df, label="sector")
        assert set(out) == {"bull", "bear"}

    def test_accepts_arbitrary_callable(self):
        def fake(df: pl.DataFrame, *, mult: float = 1.0) -> MetricOutput:
            return MetricOutput(name="fake", value=float(df["ic"].mean()) * mult)

        df = _ic_series_with_label(20)
        out = by_slice(fake, df, label="regime", mult=2.0)
        assert set(out) == {"bull", "bear"}
        assert all(o.name == "fake" for o in out.values())

    def test_cross_product_via_composite_column(self):
        """Cross-product slicing: caller composes the composite column."""
        df = _ic_series_with_label(40).with_columns(
            pl.Series("market", ["TWSE", "OTC"] * 20)
        )
        df = df.with_columns(
            pl.concat_str(["market", "regime"], separator="-").alias("uni_reg")
        )
        out = by_slice(ic, df, label="uni_reg")
        assert set(out) <= {"TWSE-bull", "TWSE-bear", "OTC-bull", "OTC-bear"}
