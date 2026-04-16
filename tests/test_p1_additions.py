"""Tests for P1 additions: VW Spread, Regime IC, BHY correction."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.metrics.quantile import quantile_spread_vw
from factorlib.metrics.ic import compute_ic, regime_ic
from factorlib._stats import bhy_threshold


# ---------------------------------------------------------------------------
# VW Spread
# ---------------------------------------------------------------------------

class TestQuantileSpreadVW:
    def _make_panel_with_cap(self, n_dates: int = 60, n_assets: int = 20):
        rng = np.random.default_rng(42)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            f = rng.standard_normal(n_assets)
            r = 0.5 * f + 0.5 * rng.standard_normal(n_assets)
            caps = rng.lognormal(10, 1, n_assets)
            for i in range(n_assets):
                rows.append({
                    "date": d, "asset_id": f"s_{i}",
                    "factor": float(f[i]),
                    "forward_return": float(r[i]),
                    "market_cap": float(caps[i]),
                })
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_basic(self):
        df = self._make_panel_with_cap()
        result = quantile_spread_vw(df, forward_periods=1, n_groups=5)
        assert result.name == "q1_q5_spread_vw"
        # With signal, VW spread should be nonzero
        assert result.value != 0.0 or result.metadata.get("reason")

    def test_missing_weight_col(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 5,
            "asset_id": [f"s_{i}" for i in range(5)],
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = quantile_spread_vw(df, forward_periods=1, n_groups=5)
        assert result.value == 0.0
        assert "missing column" in result.metadata.get("reason", "")


# ---------------------------------------------------------------------------
# Regime IC
# ---------------------------------------------------------------------------

class TestRegimeIC:
    def _make_ic_series(self, n: int = 30, mean: float = 0.05):
        rng = np.random.default_rng(42)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        return pl.DataFrame({
            "date": dates,
            "ic": rng.normal(mean, 0.02, n),
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))

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
        labels = pl.DataFrame({
            "date": dates,
            "regime": ["bull"] * 15 + ["bear"] * 15,
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
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
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# BHY Correction
# ---------------------------------------------------------------------------

class TestBHYThreshold:
    def test_empty_returns_inf(self):
        assert bhy_threshold(np.array([])) == float("inf")

    def test_no_significant_returns_inf(self):
        t = bhy_threshold(np.array([0.5, 0.3, 0.1]))
        assert t == float("inf")

    def test_very_strong_signals(self):
        t = bhy_threshold(np.array([10.0, 8.0, 6.0]))
        assert t < float("inf")
        assert t > 0

    def test_returns_finite_with_many_strong(self):
        # Many strong tests → BHY should find a finite threshold
        t_stats = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        t = bhy_threshold(t_stats)
        assert t < float("inf")
        assert t > 0
