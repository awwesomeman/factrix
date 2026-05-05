"""Tests for factrix.metrics.ic."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.ic import compute_ic, ic, ic_ir, regime_ic


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
        """Dates with < MIN_ASSETS_PER_DATE_IC assets should be excluded."""
        # 3 assets < MIN_ASSETS_PER_DATE_IC=10
        dates = [datetime(2024, 1, 1)]
        rows = [
            {
                "date": dates[0],
                "asset_id": f"A{i}",
                "factor": float(i),
                "forward_return": float(i) * 0.01,
            }
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
        assert result.stat > 0
        assert result.significance != ""

    def test_insufficient_periods(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "ic": [0.05, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic(df, forward_periods=1)
        assert math.isnan(result.value)


class TestICIR:
    def test_positive_ir(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)
        result = ic_ir(ic_df)
        assert result.value > 0
        assert result.name == "ic_ir"
        assert result.stat is None
        assert "mean_ic" in result.metadata

    def test_insufficient_periods(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "ic": [0.05, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic_ir(df)
        assert math.isnan(result.value)


class TestRegimeIC:
    def _make_ic_series(self, n: int = 30, mean: float = 0.05):
        rng = np.random.default_rng(42)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        return pl.DataFrame(
            {
                "date": dates,
                "ic": rng.normal(mean, 0.02, n),
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_time_bisection_fallback(self):
        ic_df = self._make_ic_series(30)
        result = regime_ic(ic_df)
        assert result.name == "regime_ic"
        assert "per_regime" in result.metadata
        assert "first_half" in result.metadata["per_regime"]
        assert "second_half" in result.metadata["per_regime"]

    def test_with_regime_labels(self):
        ic_df = self._make_ic_series(30)
        dates = ic_df["date"].to_list()
        labels = pl.DataFrame(
            {
                "date": dates,
                "regime": ["bull"] * 15 + ["bear"] * 15,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = regime_ic(ic_df, regime_labels=labels)
        assert "bull" in result.metadata["per_regime"]
        assert "bear" in result.metadata["per_regime"]

    def test_direction_consistency(self):
        # All positive IC → consistent
        ic_df = self._make_ic_series(30, mean=0.05)
        result = regime_ic(ic_df)
        assert result.metadata["direction_consistent"] is True

    def test_summary_uses_mean_value_min_stat(self):
        ic_df = self._make_ic_series(30, mean=0.05)
        result = regime_ic(ic_df)
        assert result.metadata["aggregation"] == "mean_value_min_stat"
        assert result.stat is not None
        assert result.significance != ""

    def test_insufficient_data(self):
        ic_df = self._make_ic_series(5)
        result = regime_ic(ic_df)
        assert math.isnan(result.value)

    def test_bhy_adjusted_p_per_regime(self):
        """Each regime gets a raw and a BHY-adjusted p; aggregate exposed."""
        ic_df = self._make_ic_series(40, mean=0.05)
        result = regime_ic(ic_df)
        for bucket in result.metadata["per_regime"].values():
            # Adjusted p must be >= raw p (BHY shrinks rejection power).
            assert bucket["p_adjusted_bhy"] >= bucket["p_value"] - 1e-12
        assert "p_value_bhy_adjusted" in result.metadata
        assert result.metadata["n_regimes"] == 2
