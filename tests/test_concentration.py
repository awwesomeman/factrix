"""Tests for factrix.metrics.concentration."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factrix.metrics.concentration import top_concentration


class TestQ1Concentration:
    def test_uniform_factor(self):
        """All Q1 stocks have same |factor| → HHI = 1/n_top → eff_n = n_top."""
        n_dates, n_assets = 10, 20
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for i in range(n_assets):
                rows.append({
                    "date": d,
                    "asset_id": f"A{i}",
                    "factor": float(i + 1),  # ranks 1..20
                    "forward_return": 0.01,
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, forward_periods=1, q_top=0.2)
        # Top 20% = 4 stocks, all with similar |factor| → eff_n near 4
        assert result.value > 2.0  # reasonably diversified

    def test_single_dominant(self):
        """One stock has extreme factor → eff_n near 1."""
        n_dates = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for i in range(10):
                f = 100.0 if i == 9 else 1.0  # asset_9 dominates
                rows.append({
                    "date": d, "asset_id": f"A{i}",
                    "factor": f, "forward_return": 0.01,
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, forward_periods=1, q_top=0.2)
        assert result.value < 2.0  # highly concentrated
