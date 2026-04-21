"""Pipeline-entry TZ consistency checks.

factrix is TZ-agnostic but enforces **consistency** at join boundaries
— the main panel, ``regime_labels``, and each ``spanning_base_spreads``
entry must share the same date dtype (time_unit + time_zone). A mismatch
raises early with a message pointing at normalization, instead of letting
polars emit a cryptic schema error deep in a metric.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factrix as fl

from tests.conftest import _cs_panel


def _main_panel() -> pl.DataFrame:
    # 60 dates × 20 assets; signal strong enough that regime_ic / spanning
    # would do real work if TZ check didn't intercept first.
    return _cs_panel(
        n_dates=60, n_assets=20, signal_coef=0.3, seed=42,
        include_price=True,
    )


def _naive_dates(n: int = 60) -> list[datetime]:
    return [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]


class TestRegimeLabelsTZMismatch:
    def test_naive_main_utc_regime_raises(self):
        df = _main_panel()  # naive Datetime("ms")
        regime = pl.DataFrame({
            "date": _naive_dates(60),
            "regime": ["bull"] * 30 + ["bear"] * 30,
        }).with_columns(
            pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
        )
        cfg = fl.CrossSectionalConfig(regime_labels=regime)
        with pytest.raises(ValueError, match="regime_labels"):
            fl.evaluate(df, "tz_mismatch", config=cfg)

    def test_time_unit_mismatch_raises(self):
        df = _main_panel()  # Datetime("ms")
        regime = pl.DataFrame({
            "date": _naive_dates(60),
            "regime": ["bull"] * 30 + ["bear"] * 30,
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))
        cfg = fl.CrossSectionalConfig(regime_labels=regime)
        with pytest.raises(ValueError, match="dtype mismatch"):
            fl.evaluate(df, "unit_mismatch", config=cfg)

    def test_matching_tz_succeeds(self):
        # Both TZ-aware UTC → should work, not raise.
        df = _main_panel().with_columns(
            pl.col("date").dt.replace_time_zone("UTC")
        )
        regime = pl.DataFrame({
            "date": _naive_dates(60),
            "regime": ["bull"] * 30 + ["bear"] * 30,
        }).with_columns(
            pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
        )
        cfg = fl.CrossSectionalConfig(regime_labels=regime)
        p = fl.evaluate(df, "tz_match", config=cfg)
        assert p.regime_ic_min_tstat is not None


class TestSpanningBaseSpreadsTZMismatch:
    def test_base_spread_tz_mismatch_raises(self):
        df = _main_panel()  # naive Datetime("ms")
        # Fake base spread series with TZ-aware date
        base_spread = pl.DataFrame({
            "date": _naive_dates(60),
            "spread": np.random.default_rng(0).standard_normal(60),
        }).with_columns(
            pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
        )
        cfg = fl.CrossSectionalConfig(
            spanning_base_spreads={"fake_base": base_spread},
        )
        with pytest.raises(ValueError, match=r"spanning_base_spreads\['fake_base'\]"):
            fl.evaluate(df, "span_tz", config=cfg)
