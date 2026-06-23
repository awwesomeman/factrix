"""Tests for factrix.datasets — synthetic panels with calibrated IC.

Trimmed to schema / shape invariants. v0.4 end-to-end tests
(``preprocess`` → ``evaluate`` round-trips with ``CrossSectionalConfig``
/ ``EventConfig``) were dropped with the v0.4 deletion sweep; v0.5
procedure tests carry their own panel synthesis.
"""

from __future__ import annotations

import polars as pl
import pytest
from factrix import datasets


class TestMakeCsPanelSchema:
    def test_canonical_columns_and_dtypes(self):
        df = datasets.make_cs_panel(n_assets=10, n_dates=60, seed=0)
        assert df.columns == ["date", "asset_id", "price", "factor"]
        assert df.schema["date"] == pl.Datetime("ms")
        assert df.schema["asset_id"] == pl.String
        assert df.schema["price"] == pl.Float64
        assert df.schema["factor"] == pl.Float64

    def test_row_count(self):
        df = datasets.make_cs_panel(n_assets=12, n_dates=40, seed=0)
        assert df.height == 12 * 40
        assert df["asset_id"].n_unique() == 12
        assert df["date"].n_unique() == 40

    def test_no_nan_or_inf(self):
        df = datasets.make_cs_panel(n_assets=10, n_dates=60, seed=0)
        for col in ("price", "factor"):
            assert df[col].is_nan().sum() == 0
            assert df[col].is_finite().all()

    def test_seed_is_deterministic(self):
        a = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=123)
        b = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=123)
        assert a.equals(b)

    def test_different_seeds_differ(self):
        a = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=1)
        b = datasets.make_cs_panel(n_assets=8, n_dates=30, seed=2)
        assert not a["factor"].equals(b["factor"])

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates must be >="):
            datasets.make_cs_panel(n_assets=5, n_dates=5, signal_horizon=5)

    def test_raises_on_singleton_cross_section(self):
        with pytest.raises(ValueError, match="n_assets"):
            datasets.make_cs_panel(n_assets=1, n_dates=60)


class TestMakeEventPanelSchema:
    def test_canonical_columns_and_dtypes(self):
        df = datasets.make_event_panel(n_assets=10, n_dates=60, seed=0)
        assert df.columns == ["date", "asset_id", "price", "factor"]
        assert df.schema["factor"] == pl.Float64

    def test_row_count(self):
        df = datasets.make_event_panel(n_assets=12, n_dates=40, seed=0)
        assert df.height == 12 * 40

    def test_factor_values_ternary(self):
        df = datasets.make_event_panel(
            n_assets=20, n_dates=120, event_rate=0.05, seed=0
        )
        assert set(df["factor"].unique().to_list()) <= {-1.0, 0.0, 1.0}

    def test_event_magnitude_scales_factor(self):
        df = datasets.make_event_panel(
            n_assets=20, n_dates=120, event_rate=0.05, event_magnitude=2.0, seed=0
        )
        assert set(df["factor"].unique().to_list()) <= {-2.0, 0.0, 2.0}

    def test_seed_is_deterministic(self):
        a = datasets.make_event_panel(n_assets=8, n_dates=30, seed=7)
        b = datasets.make_event_panel(n_assets=8, n_dates=30, seed=7)
        assert a.equals(b)

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates must be >="):
            datasets.make_event_panel(n_assets=5, n_dates=5, signal_horizon=5)


class TestMakeMultiFactorEventPanelSchema:
    def test_column_count_and_names(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=10, n_dates=60, seed=0
        )
        assert df.columns == [
            "date",
            "asset_id",
            "price",
            "factor_0000",
            "factor_0001",
            "factor_0002",
        ]

    def test_row_count(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=4, n_assets=8, n_dates=50, seed=0
        )
        assert df.height == 8 * 50

    def test_factor_values_ternary(self):
        df = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=20, n_dates=120, event_rate=0.05, seed=0
        )
        for col in ["factor_0000", "factor_0001", "factor_0002"]:
            assert set(df[col].unique().to_list()) <= {-1.0, 0.0, 1.0}

    def test_seed_is_deterministic(self):
        a = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=8, n_dates=30, seed=7
        )
        b = datasets.make_multi_factor_event_panel(
            n_factors=3, n_assets=8, n_dates=30, seed=7
        )
        assert a.equals(b)

    def test_raises_on_short_panel(self):
        with pytest.raises(ValueError, match="n_dates must be >="):
            datasets.make_multi_factor_event_panel(
                n_factors=2, n_assets=5, n_dates=5, signal_horizon=5
            )
