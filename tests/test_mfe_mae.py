"""Tests for factorlib.metrics.mfe_mae — MFE/MAE, profit_factor, event_skewness."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.metrics.mfe_mae import (
    compute_mfe_mae,
    mfe_mae_summary,
    profit_factor,
    event_skewness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event_with_price(
    n_assets: int = 10,
    n_dates: int = 300,
    event_prob: float = 0.03,
    signal_strength: float = 0.02,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for a in assets:
        price = 100.0
        for d in dates:
            is_event = rng.random() < event_prob
            direction = rng.choice([-1.0, 1.0]) if is_event else 0.0
            daily_ret = rng.normal(0, 0.015)
            if is_event:
                daily_ret += signal_strength * direction
            price *= (1 + daily_ret)

            rows.append({
                "date": d,
                "asset_id": a,
                "factor": direction,
                "forward_return": daily_ret,
                "price": price,
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def event_data() -> pl.DataFrame:
    return _make_event_with_price()


@pytest.fixture
def no_price_data() -> pl.DataFrame:
    df = _make_event_with_price()
    return df.drop("price")


# ---------------------------------------------------------------------------
# compute_mfe_mae
# ---------------------------------------------------------------------------

class TestComputeMfeMae:
    def test_returns_expected_columns(self, event_data):
        result = compute_mfe_mae(event_data, window=10)
        assert set(result.columns) >= {"date", "asset_id", "mfe", "mae",
                                        "bars_to_mfe", "bars_to_mae"}
        assert len(result) > 0

    def test_mfe_positive_mae_negative(self, event_data):
        result = compute_mfe_mae(event_data, window=10)
        assert result["mfe"].mean() > 0
        assert result["mae"].mean() < 0

    def test_no_price_returns_empty(self, no_price_data):
        result = compute_mfe_mae(no_price_data, window=10)
        assert result.is_empty()

    def test_no_events_returns_empty(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [0.0],
            "price": [100.0],
        })
        result = compute_mfe_mae(df)
        assert result.is_empty()


# ---------------------------------------------------------------------------
# mfe_mae_summary
# ---------------------------------------------------------------------------

class TestMfeMaeSummary:
    def test_returns_metric_output(self, event_data):
        mfe_df = compute_mfe_mae(event_data, window=10)
        result = mfe_mae_summary(mfe_df)
        assert result is not None
        assert result.name == "mfe_mae"
        assert "mfe_p50" in result.metadata
        assert "mae_p75" in result.metadata

    def test_none_when_empty(self):
        empty = pl.DataFrame(
            schema={
                "date": pl.Datetime("ms"), "asset_id": pl.String,
                "mfe": pl.Float64, "mae": pl.Float64,
                "bars_to_mfe": pl.Int32, "bars_to_mae": pl.Int32,
            },
        )
        result = mfe_mae_summary(empty)
        assert result is None


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_strong_signal_above_one(self, event_data):
        result = profit_factor(event_data)
        assert result.name == "profit_factor"
        assert result.value > 0

    def test_metadata_has_gains_losses(self, event_data):
        result = profit_factor(event_data)
        assert "total_gains" in result.metadata
        assert "total_losses" in result.metadata

    def test_insufficient_events(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [1.0],
            "forward_return": [0.01],
        })
        result = profit_factor(df)
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# event_skewness
# ---------------------------------------------------------------------------

class TestEventSkewness:
    def test_returns_metric(self, event_data):
        result = event_skewness(event_data)
        assert result.name == "event_skewness"
        assert isinstance(result.value, float)

    def test_insufficient_events(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [1.0],
            "forward_return": [0.01],
        })
        result = event_skewness(df)
        assert result.value == 0.0
