"""Tests for factrix.adapt."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from factrix.adapt import adapt


def _raw_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "ticker": ["A", "A"],
            "close_adj": [100.0, 101.0],
            "open_adj": [99.5, 100.5],
            "high_adj": [101.0, 102.0],
            "low_adj": [99.0, 100.0],
            "volume_adj": [1000, 1200],
            "market_cap": [1e9, 1.01e9],
        }
    )


class TestCanonicalRenames:
    def test_only_price_renamed_by_default(self):
        out = adapt(
            _raw_panel(), date="trade_date", asset_id="ticker", price="close_adj"
        )
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
            date="trade_date",
            asset_id="ticker",
            price="close_adj",
            open="open_adj",
            high="high_adj",
            low="low_adj",
            volume="volume_adj",
        )
        for c in ("open", "high", "low", "volume"):
            assert c in out.columns, f"{c} canonical missing"
        for c in ("open_adj", "high_adj", "low_adj", "volume_adj"):
            assert c not in out.columns, f"{c} source should have been renamed"

    def test_partial_ohlcv_is_allowed(self):
        # User only needs 'high' for generate_52w_high_ratio
        out = adapt(
            _raw_panel(),
            date="trade_date",
            asset_id="ticker",
            price="close_adj",
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
                date="trade_date",
                asset_id="ticker",
                price="close_adj",
                high="high_missing",
            )

    def test_collision_with_existing_canonical_raises(self):
        df = _raw_panel().rename({"open_adj": "open"})  # 'open' already exists
        with pytest.raises(ValueError, match="'open' already exists"):
            adapt(
                df,
                date="trade_date",
                asset_id="ticker",
                price="close_adj",
                open="close_adj",
            )


class TestDateDtypePromotion:
    """`adapt()` promotes pl.Date → pl.Datetime("ms") losslessly; other
    Datetime variants pass through untouched so HF precision and
    TZ-aware panels aren't silently downcast."""

    def test_pl_date_promoted_to_datetime_ms(self):
        from datetime import date

        df = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["A", "A"],
                "close_adj": [100.0, 101.0],
            }
        )
        out = adapt(df, date="trade_date", asset_id="ticker", price="close_adj")
        assert out.schema["date"] == pl.Datetime("ms")

    def test_datetime_us_passes_through(self):
        df = _raw_panel().with_columns(pl.col("trade_date").cast(pl.Datetime("us")))
        out = adapt(df, date="trade_date", asset_id="ticker", price="close_adj")
        assert out.schema["date"] == pl.Datetime("us")

    def test_datetime_with_timezone_preserved(self):
        df = _raw_panel().with_columns(
            pl.col("trade_date").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
        )
        out = adapt(df, date="trade_date", asset_id="ticker", price="close_adj")
        assert out.schema["date"] == pl.Datetime("ms", time_zone="UTC")


class TestTypePreservation:
    """`adapt()` preserves polars input type — a LazyFrame stays lazy so
    the caller controls when to collect."""

    def test_lazyframe_input_returns_lazyframe(self):
        lf = _raw_panel().lazy()
        out = adapt(lf, date="trade_date", asset_id="ticker", price="close_adj")
        assert isinstance(out, pl.LazyFrame)
        collected = out.collect()
        assert "date" in collected.columns
        assert "asset_id" in collected.columns
        assert "price" in collected.columns

    def test_lazyframe_rename_matches_eager(self):
        df = _raw_panel()
        eager = adapt(df, date="trade_date", asset_id="ticker", price="close_adj")
        lazy = adapt(
            df.lazy(), date="trade_date", asset_id="ticker", price="close_adj"
        ).collect()
        assert eager.equals(lazy)

    def test_lazyframe_date_promotion(self):
        from datetime import date

        lf = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["A", "A"],
                "close_adj": [100.0, 101.0],
            }
        ).lazy()
        out = adapt(lf, date="trade_date", asset_id="ticker", price="close_adj")
        assert isinstance(out, pl.LazyFrame)
        assert out.collect().schema["date"] == pl.Datetime("ms")

    def test_lazyframe_fill_forward_stays_lazy(self):
        lf = _raw_panel().lazy()
        out = adapt(
            lf,
            date="trade_date",
            asset_id="ticker",
            price="close_adj",
            fill_forward=True,
        )
        assert isinstance(out, pl.LazyFrame)
        assert out.collect().height == 2

    def test_pandas_input_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        pdf = pd.DataFrame(
            {
                "trade_date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "ticker": ["A", "A"],
                "close_adj": [100.0, 101.0],
            }
        )
        out = adapt(pdf, date="trade_date", asset_id="ticker", price="close_adj")
        assert isinstance(out, pl.DataFrame)
        assert "date" in out.columns

    def test_unsupported_type_raises(self):
        with pytest.raises(
            TypeError, match=r"pl\.DataFrame, pl\.LazyFrame, or pd\.DataFrame"
        ):
            adapt([{"a": 1}], date="a", asset_id="a", price="a")
