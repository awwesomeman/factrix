"""Tests for factrix.metrics.quantile."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.quantile import (
    compute_spread_series,
    quantile_spread,
    quantile_spread_vw,
)


class TestQuantileSpreadSeries:
    def test_perfect_panel(self, tiny_panel):
        series = compute_spread_series(tiny_panel, forward_periods=1, n_groups=5)
        assert "spread" in series.columns
        assert "top_return" in series.columns
        assert "bottom_return" in series.columns
        # factor=[1..5], return=[0.01..0.05], 5 groups → q1=0.05, q5=0.01
        for row in series.iter_rows(named=True):
            assert row["top_return"] == pytest.approx(0.05)
            assert row["bottom_return"] == pytest.approx(0.01)
            assert row["spread"] == pytest.approx(0.04)


class TestQuantileSpread:
    def test_noisy_panel(self, noisy_panel):
        series = compute_spread_series(noisy_panel, forward_periods=1, n_groups=5)
        assert len(series) >= 5
        assert series["spread"].null_count() == 0

    def test_insufficient_periods(self):
        from datetime import datetime

        import polars as pl

        # 2 dates < MIN_PORTFOLIO_PERIODS=5
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5 + [datetime(2024, 1, 2)] * 5,
                "asset_id": ["A", "B", "C", "D", "E"] * 2,
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05] * 2,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = quantile_spread(df, forward_periods=1, n_groups=5)
        assert math.isnan(result.value)

    def test_decomposition_in_metadata(self, tiny_panel):
        """spread = long_alpha + short_alpha (per-period)."""
        series = compute_spread_series(tiny_panel, forward_periods=1, n_groups=5)
        for row in series.iter_rows(named=True):
            long = row["top_return"] - row["universe_return"]
            short = row["universe_return"] - row["bottom_return"]
            assert long + short == pytest.approx(row["spread"])

    def test_metadata_has_long_short(self, noisy_panel):
        result = quantile_spread(noisy_panel, forward_periods=1, n_groups=5)
        if result.value != 0.0:
            assert "long_alpha" in result.metadata
            assert "short_alpha" in result.metadata
            assert "long_stat" in result.metadata
            assert "short_stat" in result.metadata
            assert "p_value" in result.metadata


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
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"s_{i}",
                        "factor": float(f[i]),
                        "forward_return": float(r[i]),
                        "market_cap": float(caps[i]),
                    }
                )
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_basic(self):
        df = self._make_panel_with_cap()
        result = quantile_spread_vw(df, forward_periods=1, n_groups=5)
        assert result.name == "quantile_spread_vw"
        # With signal, VW spread should be nonzero
        assert result.value != 0.0 or result.metadata.get("reason")

    def test_lag_weights_flag_recorded(self):
        df = self._make_panel_with_cap()
        default = quantile_spread_vw(df, forward_periods=1, n_groups=5)
        explicit_off = quantile_spread_vw(
            df,
            forward_periods=1,
            n_groups=5,
            lag_weights=False,
        )
        assert default.metadata["weights_lagged"] is True
        assert explicit_off.metadata["weights_lagged"] is False
        # Default lag drops exactly the first sampled row per asset on
        # the balanced panel — strict shrinkage of the effective window.
        assert default.metadata["n_periods"] < explicit_off.metadata["n_periods"]

    def test_missing_weight_col(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "asset_id": [f"s_{i}" for i in range(5)],
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = quantile_spread_vw(df, forward_periods=1, n_groups=5)
        assert math.isnan(result.value)
        assert result.metadata.get("reason") == "no_weight_column"
        assert result.metadata.get("missing_column") == "market_cap"
