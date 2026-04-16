"""Tests for factorlib.metrics.tradability."""

import polars as pl
import pytest
from datetime import datetime, timedelta

from factorlib.metrics.tradability import breakeven_cost, turnover, net_spread


class TestComputeTurnover:
    def test_static_factor(self):
        """Same factor values every date → rank_autocorr=1.0 → turnover=0."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        rows = []
        for d in dates:
            for i, a in enumerate(["A", "B", "C"]):
                rows.append({"date": d, "asset_id": a, "factor": float(i + 1)})
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = turnover(df)
        assert result.value == pytest.approx(0.0, abs=0.01)

    def test_single_date(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 3,
            "asset_id": ["A", "B", "C"],
            "factor": [1.0, 2.0, 3.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = turnover(df)
        assert result.value == 0.0


class TestBreakevenCost:
    def test_basic(self):
        # gross=0.10, turnover=0.5 → 0.10/(2*0.5)*10000 = 1000 bps
        result = breakeven_cost(0.10, 0.5)
        assert result.value == pytest.approx(1000.0)

    def test_zero_turnover(self):
        result = breakeven_cost(0.10, 0.0)
        assert result.value == float("inf")


class TestNetSpread:
    def test_basic(self):
        # gross=0.10, turnover=0.5, cost=30bps
        # net = 0.10 - 2*(30/10000)*0.5 = 0.10 - 0.003 = 0.097
        result = net_spread(0.10, 0.5, estimated_cost_bps=30)
        assert result.value == pytest.approx(0.097)

    def test_cost_exceeds_alpha(self):
        result = net_spread(0.001, 0.5, estimated_cost_bps=100)
        assert result.value < 0
