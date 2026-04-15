"""Tests for factorlib.tools.panel.ic."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.tools.panel.ic import compute_ic, ic, ic_ir


class TestComputeIC:
    def test_perfect_rank(self, tiny_panel):
        result = compute_ic(tiny_panel)
        # factor and return have identical ranking → IC = 1.0
        for ic_val in result["ic"].to_list():
            assert ic_val == pytest.approx(1.0)

    def test_inverse_rank(self, tiny_panel):
        # Reverse the returns
        inverted = tiny_panel.with_columns(
            (0.06 - pl.col("forward_return")).alias("forward_return")
        )
        result = compute_ic(inverted)
        for ic_val in result["ic"].to_list():
            assert ic_val == pytest.approx(-1.0)

    def test_drops_small_dates(self):
        """Dates with < MIN_IC_PERIODS assets should be excluded."""
        # 3 assets < MIN_IC_PERIODS=10
        dates = [datetime(2024, 1, 1)]
        rows = [
            {"date": dates[0], "asset_id": f"A{i}", "factor": float(i), "forward_return": float(i) * 0.01}
            for i in range(3)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)
        assert len(result) == 0

    def test_output_schema(self, noisy_panel):
        result = compute_ic(noisy_panel)
        assert result.columns == ["date", "ic"]
        assert result["date"].dtype == pl.Datetime("ms")


class TestIC:
    def test_positive_ic(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)
        result = ic(ic_df, forward_periods=1)
        assert result.name == "ic"
        assert result.value > 0  # noisy_panel has positive IC
        assert result.t_stat > 0
        assert result.significance != ""

    def test_insufficient_periods(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
            "ic": [0.05, 0.03, 0.04],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic(df, forward_periods=1)
        assert result.value == 0.0


class TestICIR:
    def test_positive_ir(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)
        result = ic_ir(ic_df)
        assert result.value > 0
        assert result.name == "ic_ir"
        assert result.t_stat is None
        assert "mean_ic" in result.metadata

    def test_insufficient_periods(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
            "ic": [0.05, 0.03, 0.04],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic_ir(df)
        assert result.value == 0.0
