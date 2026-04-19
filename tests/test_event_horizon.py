"""Tests for factorlib.metrics.event_horizon and signal_density."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.metrics.event_horizon import (
    compute_event_returns,
    event_around_return,
    multi_horizon_hit_rate,
)
from factorlib.metrics.event_quality import signal_density


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event_with_price(
    n_assets: int = 20,
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
                "date": d, "asset_id": a, "factor": direction,
                "forward_return": daily_ret, "price": price,
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def event_data() -> pl.DataFrame:
    return _make_event_with_price()


@pytest.fixture
def no_price_data(event_data) -> pl.DataFrame:
    return event_data.drop("price")


# ---------------------------------------------------------------------------
# compute_event_returns
# ---------------------------------------------------------------------------

class TestComputeEventReturns:
    def test_returns_expected_columns(self, event_data):
        result = compute_event_returns(event_data, offsets=[1, 6, 12])
        assert set(result.columns) >= {"offset", "date", "asset_id", "signed_return"}
        assert len(result) > 0

    def test_multiple_offsets(self, event_data):
        result = compute_event_returns(event_data, offsets=[-3, -1, 1, 6])
        offsets_found = result["offset"].unique().sort().to_list()
        assert -3 in offsets_found
        assert 1 in offsets_found

    def test_no_price_returns_empty(self, no_price_data):
        result = compute_event_returns(no_price_data)
        assert result.is_empty()

    def test_post_event_signed(self, event_data):
        """Post-event returns are direction-adjusted (signed)."""
        # Use longer horizon for stronger signal-to-noise
        result = compute_event_returns(event_data, offsets=[12])
        assert result["signed_return"].mean() > 0

    def test_output_date_dtype_mirrors_input_us(self, event_data):
        df_us = event_data.with_columns(
            pl.col("date").cast(pl.Datetime("us"))
        )
        result = compute_event_returns(df_us, offsets=[1, 6])
        assert result.schema["date"] == pl.Datetime("us")

    def test_output_date_dtype_mirrors_tz_aware(self, event_data):
        df_utc = event_data.with_columns(
            pl.col("date").dt.replace_time_zone("UTC")
        )
        result = compute_event_returns(df_utc, offsets=[1, 6])
        assert result.schema["date"] == pl.Datetime("ms", time_zone="UTC")


# ---------------------------------------------------------------------------
# event_around_return
# ---------------------------------------------------------------------------

class TestEventAroundReturn:
    def test_returns_metric_output(self, event_data):
        result = event_around_return(event_data)
        assert result is not None
        assert result.name == "event_around_return"
        assert "per_offset" in result.metadata

    def test_per_offset_has_stats(self, event_data):
        result = event_around_return(event_data, offsets=[-3, 1, 6])
        per_offset = result.metadata["per_offset"]
        assert 1 in per_offset
        assert "mean" in per_offset[1]
        assert "hit_rate" in per_offset[1]

    def test_leakage_value_small(self, event_data):
        """Pre-event leakage should be near zero for random events."""
        result = event_around_return(event_data, offsets=[-6, -3, -1, 1])
        assert result.value < 0.01  # leakage score

    def test_none_without_price(self, no_price_data):
        result = event_around_return(no_price_data)
        assert result is None


# ---------------------------------------------------------------------------
# multi_horizon_hit_rate
# ---------------------------------------------------------------------------

class TestMultiHorizonHitRate:
    def test_returns_metric_output(self, event_data):
        result = multi_horizon_hit_rate(event_data, horizons=[1, 6, 12])
        assert result is not None
        assert result.name == "multi_horizon_hit_rate"
        assert "per_horizon" in result.metadata

    def test_per_horizon_has_stats(self, event_data):
        result = multi_horizon_hit_rate(event_data, horizons=[1, 6])
        ph = result.metadata["per_horizon"]
        assert 1 in ph
        assert "hit_rate" in ph[1]
        assert "z_stat" in ph[1]

    def test_none_without_price(self, no_price_data):
        result = multi_horizon_hit_rate(no_price_data)
        assert result is None


# ---------------------------------------------------------------------------
# signal_density
# ---------------------------------------------------------------------------

class TestSignalDensity:
    def test_returns_metric_output(self, event_data):
        result = signal_density(event_data)
        assert result.name == "signal_density"
        assert result.value > 0

    def test_sparse_events_large_gap(self):
        """Low event_prob → large gap between events."""
        df = _make_event_with_price(event_prob=0.005, seed=99)
        result = signal_density(df)
        assert result.value > 50  # ~200 bars between events

    def test_dense_events_small_gap(self):
        """High event_prob → small gap."""
        df = _make_event_with_price(event_prob=0.10, seed=88)
        result = signal_density(df)
        assert result.value < 20

    def test_metadata_has_counts(self, event_data):
        result = signal_density(event_data)
        assert "n_events_total" in result.metadata
        assert "mean_events_per_asset" in result.metadata


# ---------------------------------------------------------------------------
# Standalone import
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_importable(self):
        from factorlib.metrics import (
            compute_event_returns, event_around_return,
            multi_horizon_hit_rate, signal_density,
        )
        assert all(callable(f) for f in [
            compute_event_returns, event_around_return,
            multi_horizon_hit_rate, signal_density,
        ])
