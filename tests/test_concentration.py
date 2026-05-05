"""Tests for factrix.metrics.concentration."""

import polars as pl
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
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{i}",
                        "factor": float(i + 1),  # ranks 1..20
                        "forward_return": 0.01,
                    }
                )
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
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{i}",
                        "factor": f,
                        "forward_return": 0.01,
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, forward_periods=1, q_top=0.2)
        assert result.value < 2.0  # highly concentrated

    def test_alpha_contribution_sees_return_concentration(self):
        """Uniform factor + one outlier return → alpha-weighted HHI flags it."""
        n_dates = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            # Top bucket = assets 8, 9 (top 20% of 10 with uniform factor
            # values — ranks are broken by tie handling).
            for i in range(10):
                ret = 0.10 if i == 9 else 0.001
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{i}",
                        "factor": float(i + 1),
                        "forward_return": ret,
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        signal = top_concentration(
            df,
            forward_periods=1,
            q_top=0.2,
            weight_by="abs_factor",
        )
        alpha = top_concentration(
            df,
            forward_periods=1,
            q_top=0.2,
            weight_by="alpha_contribution",
        )
        # Signal says top bucket is balanced (two near-equal |factor|).
        # Alpha says bucket is driven by one outlier → far more concentrated.
        assert alpha.value < signal.value
        assert alpha.metadata["weight_by"] == "alpha_contribution"
        assert signal.metadata["weight_by"] == "abs_factor"

    def test_alpha_contribution_missing_return_column(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "asset_id": [f"A{i}" for i in range(5)],
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = top_concentration(df, weight_by="alpha_contribution")
        assert result.metadata.get("reason") == "no_return_column"
