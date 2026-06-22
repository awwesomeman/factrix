"""Tests for factrix.preprocess.returns."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from factrix.preprocess.returns import (
    compute_abnormal_return,
    compute_forward_return,
    winsorize_forward_return,
)


def _make_price_data():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    return pl.DataFrame(
        {
            "date": dates * 2,
            "asset_id": ["A"] * 5 + ["B"] * 5,
            "price": [
                *[100.0, 110.0, 121.0, 133.1, 146.41],
                *[200.0, 190.0, 180.5, 171.475, 162.9],
            ],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestComputeForwardReturn:
    def test_basic(self):
        df = _make_price_data()
        result = compute_forward_return(df, forward_periods=1)
        a_returns = result.filter(pl.col("asset_id") == "A")["forward_return"].to_list()
        assert a_returns[0] == pytest.approx(0.10)

    def test_drops_nulls(self):
        df = _make_price_data()
        result = compute_forward_return(df, forward_periods=1)
        assert result["forward_return"].null_count() == 0
        # 5 dates per asset, forward_periods=1, t+1 entry:
        # need price[t+1] (entry) and price[t+2] (exit) → 3 valid rows per asset
        assert len(result) == 6

    def test_per_asset_independent(self):
        df = _make_price_data()
        result = compute_forward_return(df, forward_periods=1)
        a_ret = result.filter(pl.col("asset_id") == "A")["forward_return"][0]
        b_ret = result.filter(pl.col("asset_id") == "B")["forward_return"][0]
        assert a_ret > 0
        assert b_ret < 0

    def test_divided_by_periods(self):
        """forward_return = (price[t+1+N]/price[t+1] - 1) / N."""
        df = _make_price_data()
        # Asset A: price 100 → 121 over 2 periods → raw return 0.21
        # Per-period: 0.21 / 2 = 0.105
        result = compute_forward_return(df, forward_periods=2)
        a_ret = result.filter(pl.col("asset_id") == "A")["forward_return"][0]
        assert a_ret == pytest.approx(0.21 / 2)

    def test_raises_when_forward_return_already_present(self):
        from factrix._errors import UserInputError

        once = compute_forward_return(_make_price_data(), forward_periods=1)
        with pytest.raises(UserInputError, match="not idempotent"):
            compute_forward_return(once, forward_periods=1)

    def test_overwrite_recomputes_in_place(self):
        once = compute_forward_return(_make_price_data(), forward_periods=1)
        # overwrite drops the old column and recomputes; not idempotent — the
        # already-truncated tail is dropped again, so the row count shrinks.
        twice = compute_forward_return(once, forward_periods=1, overwrite=True)
        assert "forward_return" in twice.columns
        assert twice["forward_return"].null_count() == 0
        assert twice.height < once.height

    def test_overwrite_changes_horizon(self):
        raw = _make_price_data()
        once = compute_forward_return(raw, forward_periods=1)
        changed = compute_forward_return(once, forward_periods=2, overwrite=True)
        assert "forward_return" in changed.columns
        assert changed["forward_return"].null_count() == 0
        assert changed["forward_return"].to_list() != once["forward_return"].to_list()

    def test_stamps_overlap_horizon(self):
        from factrix._data_input import (
            _FORWARD_PERIODS_COL,
            _read_forward_periods_stamp,
        )

        result = compute_forward_return(_make_price_data(), forward_periods=3)
        assert _FORWARD_PERIODS_COL in result.columns
        assert _read_forward_periods_stamp(result) == 3
        # one constant value across all rows
        assert result[_FORWARD_PERIODS_COL].n_unique() == 1

    def test_overwrite_restamps_new_horizon(self):
        import factrix as fx
        from factrix._data_input import _read_forward_periods_stamp

        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=60, seed=0)
        once = compute_forward_return(raw, forward_periods=1)
        changed = compute_forward_return(once, forward_periods=2, overwrite=True)
        assert changed.height > 0
        assert _read_forward_periods_stamp(changed) == 2


class TestWinsorizeForwardReturn:
    def test_noop(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = winsorize_forward_return(df, lower=0.0, upper=1.0)
        assert result["forward_return"].to_list() == df["forward_return"].to_list()

    def test_clips_extreme(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 10,
                "forward_return": [0.01] * 8 + [0.50, -0.50],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = winsorize_forward_return(df, lower=0.1, upper=0.9)
        vals = result["forward_return"].to_list()
        assert max(vals) < 0.50
        assert min(vals) > -0.50


class TestComputeAbnormalReturn:
    def test_zero_mean_per_date(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_abnormal_return(df)
        mean_abnormal = result["abnormal_return"].mean()
        assert abs(mean_abnormal) < 1e-10

    def test_known_values(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 3,
                "forward_return": [0.01, 0.02, 0.03],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_abnormal_return(df)
        expected = [-0.01, 0.0, 0.01]
        actual = result["abnormal_return"].to_list()
        for a, e in zip(actual, expected, strict=False):
            assert a == pytest.approx(e)
