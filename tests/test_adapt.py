"""Tests for factorlib.adapt."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from factorlib.adapt import adapt


def _raw_panel() -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "ticker": ["A", "A"],
        "close_adj": [100.0, 101.0],
        "open_adj": [99.5, 100.5],
        "high_adj": [101.0, 102.0],
        "low_adj": [99.0, 100.0],
        "volume_adj": [1000, 1200],
        "market_cap": [1e9, 1.01e9],
    })


class TestCanonicalRenames:
    def test_only_price_renamed_by_default(self):
        out = adapt(_raw_panel(), date="trade_date", asset_id="ticker", price="close_adj")
        assert "date" in out.columns and "trade_date" not in out.columns
        assert "asset_id" in out.columns and "ticker" not in out.columns
        assert "price" in out.columns and "close_adj" not in out.columns
        # Unrelated columns pass through
        assert "open_adj" in out.columns
        assert "high_adj" in out.columns
        assert "market_cap" in out.columns

    def test_ohlcv_renames_when_supplied(self):
        out = adapt(
            _raw_panel(),
            date="trade_date", asset_id="ticker", price="close_adj",
            open="open_adj", high="high_adj", low="low_adj", volume="volume_adj",
        )
        for c in ("open", "high", "low", "volume"):
            assert c in out.columns, f"{c} canonical missing"
        for c in ("open_adj", "high_adj", "low_adj", "volume_adj"):
            assert c not in out.columns, f"{c} source should have been renamed"

    def test_partial_ohlcv_is_allowed(self):
        # User only needs 'high' for generate_52w_high_ratio
        out = adapt(
            _raw_panel(),
            date="trade_date", asset_id="ticker", price="close_adj",
            high="high_adj",
        )
        assert "high" in out.columns
        # others untouched
        assert "open_adj" in out.columns
        assert "low_adj" in out.columns

    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="high_missing"):
            adapt(
                _raw_panel(),
                date="trade_date", asset_id="ticker", price="close_adj",
                high="high_missing",
            )

    def test_collision_with_existing_canonical_raises(self):
        df = _raw_panel().rename({"open_adj": "open"})  # 'open' already exists
        with pytest.raises(ValueError, match="'open' already exists"):
            adapt(
                df,
                date="trade_date", asset_id="ticker", price="close_adj",
                open="close_adj",
            )
