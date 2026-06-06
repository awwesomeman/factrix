"""Tests for factrix.metrics.monotonicity."""

import math

import pytest
from factrix.metrics.monotonicity import monotonicity


class TestComputeMonotonicity:
    def test_perfect_monotonic(self, noisy_panel):
        # WHY: tiny_panel only has 3 dates, < MIN_MONOTONICITY_PERIODS after sampling
        # Use noisy_panel (20 dates × 30 assets) with perfect factor-return alignment
        import polars as pl

        perfect = noisy_panel.with_columns(
            pl.col("factor").rank(method="average").over("date").alias("forward_return")
        )
        result = monotonicity(perfect, forward_periods=1, n_groups=5)["factor"]
        assert result.value == pytest.approx(1.0)

    def test_inverse_monotonic(self, noisy_panel):
        import polars as pl

        inverted = noisy_panel.with_columns(
            (-pl.col("factor").rank(method="average").over("date")).alias(
                "forward_return"
            )
        )
        result = monotonicity(inverted, forward_periods=1, n_groups=5)["factor"]
        assert result.value == pytest.approx(1.0)
        assert result.metadata["mean_signed"] == pytest.approx(-1.0)

    def test_insufficient_periods(self):
        from datetime import datetime

        import polars as pl

        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5,
                "asset_id": ["A", "B", "C", "D", "E"],
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = monotonicity(df, forward_periods=1, n_groups=5)["factor"]
        # Only 1 date < MIN_MONOTONICITY_PERIODS=5
        assert math.isnan(result.value)
        assert result.p_value is None or result.p_value >= 0.10


class TestMonotonicityBatch:
    """Multi-factor path of ``monotonicity``."""

    def test_multi_factor_matches_list_of_one(self):
        from datetime import datetime, timedelta

        import numpy as np
        import polars as pl

        rng = np.random.default_rng(31)
        n_assets, n_dates = 80, 60
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for date in dates:
            returns = rng.standard_normal(n_assets) * 0.02
            f1 = returns + rng.standard_normal(n_assets) * 0.05
            f2 = -returns + rng.standard_normal(n_assets) * 0.05
            for asset_id in range(n_assets):
                rows.append(
                    {
                        "date": date,
                        "asset_id": asset_id,
                        "f1": float(f1[asset_id]),
                        "f2": float(f2[asset_id]),
                        "forward_return": float(returns[asset_id]),
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        batch = monotonicity(
            panel, forward_periods=1, n_groups=5, factor_cols=["f1", "f2"]
        )
        for col in ("f1", "f2"):
            single = monotonicity(
                panel, forward_periods=1, n_groups=5, factor_cols=[col]
            )[col]
            assert batch[col].value == pytest.approx(single.value)
            assert batch[col].stat == pytest.approx(single.stat)

    def test_empty_factor_list_rejected(self, noisy_panel):
        with pytest.raises(ValueError, match="non-empty"):
            monotonicity(noisy_panel, forward_periods=1, n_groups=5, factor_cols=[])
