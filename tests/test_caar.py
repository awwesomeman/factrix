"""Tests for factorlib.metrics.caar — CAAR, BMP, event_hit_rate, event_ic."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.metrics.caar import (
    compute_caar,
    caar,
    bmp_test,
)
from factorlib.metrics.event_quality import event_hit_rate, event_ic
from factorlib._types import MIN_EVENTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event_signal(
    n_assets: int = 50,
    n_dates: int = 500,
    event_prob: float = 0.02,
    signal_strength: float = 0.01,
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic event signal data.

    Each day, each asset has ``event_prob`` chance of triggering an event
    (factor = +1 or -1). Post-event forward_return = signal_strength *
    sign(factor) + noise.
    """
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        for a in assets:
            is_event = rng.random() < event_prob
            if is_event:
                direction = rng.choice([-1.0, 1.0])
                ret = signal_strength * direction + rng.normal(0, 0.02)
            else:
                direction = 0.0
                ret = rng.normal(0, 0.02)

            rows.append({
                "date": d,
                "asset_id": a,
                "factor": direction,
                "forward_return": ret,
                "price": 100 + rng.normal(0, 5),
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


@pytest.fixture
def strong_signal() -> pl.DataFrame:
    return _make_event_signal(signal_strength=0.03)


@pytest.fixture
def noise_signal() -> pl.DataFrame:
    return _make_event_signal(signal_strength=0.0, seed=99)


@pytest.fixture
def single_asset_signal() -> pl.DataFrame:
    return _make_event_signal(n_assets=1, n_dates=1000, event_prob=0.05,
                              signal_strength=0.03, seed=77)


# ---------------------------------------------------------------------------
# compute_caar
# ---------------------------------------------------------------------------

class TestComputeCaar:
    def test_returns_date_caar_columns(self, strong_signal):
        result = compute_caar(strong_signal)
        assert "date" in result.columns
        assert "caar" in result.columns
        assert len(result) > 0

    def test_filters_non_events(self, strong_signal):
        result = compute_caar(strong_signal)
        n_event_dates = (
            strong_signal.filter(pl.col("factor") != 0)["date"].n_unique()
        )
        assert len(result) == n_event_dates

    def test_strong_signal_positive_mean(self, strong_signal):
        result = compute_caar(strong_signal)
        assert result["caar"].mean() > 0

    def test_noise_mean_near_zero(self, noise_signal):
        result = compute_caar(noise_signal)
        assert abs(result["caar"].mean()) < 0.01

    def test_empty_events(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [0.0],
            "forward_return": [0.01],
        })
        result = compute_caar(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# caar (significance test)
# ---------------------------------------------------------------------------

class TestCaar:
    def test_strong_signal_significant(self, strong_signal):
        caar_df = compute_caar(strong_signal)
        result = caar(caar_df)
        assert result.name == "caar"
        assert result.value > 0
        assert abs(result.stat) > 2.0
        assert result.significance in ("***", "**")

    def test_noise_not_significant(self, noise_signal):
        caar_df = compute_caar(noise_signal)
        result = caar(caar_df)
        assert abs(result.stat) < 2.0

    def test_insufficient_data(self):
        df = pl.DataFrame({
            "date": pl.Series([], dtype=pl.Datetime("ms")),
            "caar": pl.Series([], dtype=pl.Float64),
        })
        result = caar(df)
        assert result.value == 0.0
        assert result.stat == 0.0


# ---------------------------------------------------------------------------
# bmp_test
# ---------------------------------------------------------------------------

class TestBmpTest:
    def test_strong_signal_significant(self, strong_signal):
        result = bmp_test(strong_signal)
        assert result.name == "bmp_test"
        assert abs(result.stat) > 2.0
        assert result.metadata["stat_type"] == "z"

    def test_noise_not_significant(self, noise_signal):
        result = bmp_test(noise_signal)
        assert abs(result.stat) < 2.0

    def test_no_events(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [0.0],
            "forward_return": [0.01],
            "price": [100.0],
        })
        result = bmp_test(df)
        assert result.value == 0.0

    def test_uses_price_for_vol(self, strong_signal):
        result = bmp_test(strong_signal)
        assert result.metadata.get("n_events", 0) > 0


# ---------------------------------------------------------------------------
# event_hit_rate
# ---------------------------------------------------------------------------

class TestEventHitRate:
    def test_strong_signal_high_hit_rate(self, strong_signal):
        result = event_hit_rate(strong_signal)
        assert result.name == "event_hit_rate"
        assert result.value > 0.5
        assert abs(result.stat) > 2.0

    def test_noise_hit_rate_near_half(self, noise_signal):
        result = event_hit_rate(noise_signal)
        assert abs(result.value - 0.5) < 0.1

    def test_single_asset(self, single_asset_signal):
        result = event_hit_rate(single_asset_signal)
        assert result.metadata["n_events"] > 0
        assert 0.0 <= result.value <= 1.0

    def test_no_events(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [0.0],
            "forward_return": [0.01],
        })
        result = event_hit_rate(df)
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# event_ic
# ---------------------------------------------------------------------------

def _make_continuous_signal(
    n_assets: int = 50,
    n_dates: int = 500,
    event_prob: float = 0.02,
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic continuous event signal: stronger |signal| → larger return."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"asset_{i}" for i in range(n_assets)]

    rows = []
    for d in dates:
        for a in assets:
            is_event = rng.random() < event_prob
            if is_event:
                magnitude = rng.uniform(0.5, 5.0)
                direction = rng.choice([-1.0, 1.0])
                signal = direction * magnitude
                # Stronger signal → larger directional return
                ret = 0.005 * magnitude * direction + rng.normal(0, 0.02)
            else:
                signal = 0.0
                ret = rng.normal(0, 0.02)

            rows.append({
                "date": d, "asset_id": a,
                "factor": signal, "forward_return": ret,
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


class TestEventIc:
    def test_continuous_signal_positive_ic(self):
        df = _make_continuous_signal()
        result = event_ic(df)
        assert result.name == "event_ic"
        assert result.value > 0
        assert result.metadata["method"] == "Spearman rank correlation (|signal| vs signed_car)"

    def test_discrete_signal_skipped(self, strong_signal):
        """All ±1 values → |factor| constant → IC = 0 (no variance)."""
        result = event_ic(strong_signal)
        assert result.value == 0.0
        assert result.stat == 0.0

    def test_insufficient_events(self):
        df = pl.DataFrame({
            "date": pl.Series([datetime(2020, 1, 1)], dtype=pl.Datetime("ms")),
            "asset_id": ["A"],
            "factor": [2.5],
            "forward_return": [0.01],
        })
        result = event_ic(df)
        assert result.value == 0.0

    def test_standalone_import(self):
        from factorlib.metrics import event_ic as eic
        assert callable(eic)
