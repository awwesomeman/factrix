"""Tests for factrix.metrics.mfe_mae and factrix.metrics.event_quality."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factrix.metrics.mfe_mae import (
    compute_mfe_mae,
    mfe_mae_summary,
)
from factrix.metrics.event_quality import (
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
            price *= 1 + daily_ret

            rows.append(
                {
                    "date": d,
                    "asset_id": a,
                    "factor": direction,
                    "forward_return": daily_ret,
                    "price": price,
                }
            )

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
        assert set(result.columns) >= {
            "date",
            "asset_id",
            "mfe",
            "mae",
            "bars_to_mfe",
            "bars_to_mae",
        }
        assert len(result) > 0

    def test_mfe_positive_mae_negative(self, event_data):
        result = compute_mfe_mae(event_data, window=10)
        assert result["mfe"].mean() > 0
        assert result["mae"].mean() < 0

    def test_z_score_columns_present_and_consistent(self, event_data):
        result = compute_mfe_mae(event_data, window=10, estimation_window=60)
        assert {"mfe_z", "mae_z", "est_sigma"}.issubset(result.columns)
        # Where est_sigma is finite-positive the z-scores must match
        # the raw excursion divided by σ · √window.
        finite = result.filter(
            pl.col("est_sigma").is_finite() & (pl.col("est_sigma") > 0.0),
        )
        if finite.is_empty():
            pytest.skip("No events had enough look-back for σ estimation")
        scale = finite["est_sigma"] * math.sqrt(10)
        assert finite["mfe_z"].to_numpy() == pytest.approx(
            (finite["mfe"] / scale).to_numpy(),
        )
        assert finite["mae_z"].to_numpy() == pytest.approx(
            (finite["mae"] / scale).to_numpy(),
        )

    def test_no_price_returns_empty(self, no_price_data):
        result = compute_mfe_mae(no_price_data, window=10)
        assert result.is_empty()

    def test_no_events_returns_empty(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [0.0],
                "price": [100.0],
            }
        )
        result = compute_mfe_mae(df)
        assert result.is_empty()

    def test_output_date_dtype_mirrors_input_us(self, event_data):
        df_us = event_data.with_columns(pl.col("date").cast(pl.Datetime("us")))
        result = compute_mfe_mae(df_us, window=10)
        assert result.schema["date"] == pl.Datetime("us"), (
            "us-precision input should survive to the output"
        )

    def test_output_date_dtype_mirrors_tz_aware(self, event_data):
        df_utc = event_data.with_columns(pl.col("date").dt.replace_time_zone("UTC"))
        result = compute_mfe_mae(df_utc, window=10)
        assert result.schema["date"] == pl.Datetime("ms", time_zone="UTC")

    def test_empty_output_also_mirrors_dtype(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("us")),
                "asset_id": ["A"],
                "factor": [0.0],
                "price": [100.0],
            }
        )
        result = compute_mfe_mae(df)
        assert result.is_empty()
        assert result.schema["date"] == pl.Datetime("us")

    def test_min_estimation_samples_lower_admits_more_z_scores(
        self,
        event_data,
    ):
        """Lowering the threshold (e.g. for weekly data) lets early events
        in the panel get a finite est_sigma where the BMP-default 20
        would NaN them out. Verifies the parameter actually plumbs into
        the σ̂ guard."""
        strict = compute_mfe_mae(
            event_data,
            window=10,
            estimation_window=30,
            min_estimation_samples=20,
        )
        loose = compute_mfe_mae(
            event_data,
            window=10,
            estimation_window=30,
            min_estimation_samples=5,
        )
        strict_finite = strict["est_sigma"].is_finite().sum()
        loose_finite = loose["est_sigma"].is_finite().sum()
        assert loose_finite >= strict_finite
        assert loose_finite > strict_finite, (
            "Fixture should have at least one event whose look-back "
            "fits the loose threshold but not the strict one."
        )

    def test_min_estimation_samples_below_two_raises(self, event_data):
        with pytest.raises(ValueError, match="min_estimation_samples"):
            compute_mfe_mae(event_data, window=10, min_estimation_samples=1)


# ---------------------------------------------------------------------------
# mfe_mae_summary
# ---------------------------------------------------------------------------


class TestMfeMaeSummary:
    def test_returns_metric_output(self, event_data):
        mfe_df = compute_mfe_mae(event_data, window=10)
        result = mfe_mae_summary(mfe_df)
        assert result is not None
        assert result.name == "mfe_mae_summary"
        assert "mfe_p50" in result.metadata
        assert "mae_p75" in result.metadata

    def test_short_circuit_when_empty(self):
        empty = pl.DataFrame(
            schema={
                "date": pl.Datetime("ms"),
                "asset_id": pl.String,
                "mfe": pl.Float64,
                "mae": pl.Float64,
                "bars_to_mfe": pl.Int32,
                "bars_to_mae": pl.Int32,
            },
        )
        result = mfe_mae_summary(empty)
        assert result.name == "mfe_mae_summary"
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_price_data"
        assert result.metadata["n_events"] == 0


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
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [1.0],
                "forward_return": [0.01],
            }
        )
        result = profit_factor(df)
        assert math.isnan(result.value)


# ---------------------------------------------------------------------------
# event_skewness
# ---------------------------------------------------------------------------


class TestEventSkewness:
    def test_returns_metric(self, event_data):
        result = event_skewness(event_data)
        assert result.name == "event_skewness"
        assert isinstance(result.value, float)

    def test_insufficient_events(self):
        df = pl.DataFrame(
            {
                "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
                "asset_id": ["A"],
                "factor": [1.0],
                "forward_return": [0.01],
            }
        )
        result = event_skewness(df)
        assert math.isnan(result.value)
