"""Tests for factrix.metrics.tradability."""

import math
import polars as pl
import pytest
from datetime import datetime, timedelta

from factrix.metrics.tradability import breakeven_cost, turnover, net_spread


def _panel(n_dates: int, assets: list[str], factor_of) -> pl.DataFrame:
    """Build a ``date, asset_id, factor`` panel from a per-(date_idx, asset) fn."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = [
        {"date": d, "asset_id": a, "factor": float(factor_of(t, a))}
        for t, d in enumerate(dates)
        for a in assets
    ]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestComputeTurnover:
    def test_static_factor(self):
        """Same factor values every date → rank_autocorr=1.0 → turnover=0."""
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a) - ord("A") + 1)
        result = turnover(df)
        assert result.value == pytest.approx(0.0, abs=0.01)
        assert result.metadata["n_pairs"] == 4
        assert result.metadata["forward_periods"] == 1
        assert result.metadata["quantile"] is None

    def test_single_date(self):
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 1)] * 3,
            "asset_id": ["A", "B", "C"],
            "factor": [1.0, 2.0, 3.0],
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = turnover(df)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_dates"

    def test_insufficient_dates_for_horizon(self):
        """2·h + 1 dates is the minimum for SE to be defined."""
        df = _panel(4, ["A", "B", "C"], lambda t, a: ord(a) + t)
        result = turnover(df, forward_periods=2)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_dates"
        assert result.metadata["min_required"] == 5

    def test_non_overlapping_skips_intermediate_noise(self):
        """Rank flips inside the holding window must not count as turnover.

        At even ``t`` ranks are (A=1,B=2,C=3); at odd ``t`` they are the
        reverse. With ``forward_periods=2`` we sample only even dates, so
        every pair's rank-AC is +1 → turnover=0.
        """
        def factor(t, a):
            base = ord(a) - ord("A") + 1
            return base if t % 2 == 0 else (4 - base)
        df = _panel(7, ["A", "B", "C"], factor)
        result = turnover(df, forward_periods=2)
        assert result.value == pytest.approx(0.0, abs=0.01)
        assert result.metadata["n_pairs"] == 3

    def test_quantile_filter_restricts_to_tails(self):
        """Quantile filter must actually select tail names and only tail names.

        Ten assets with monotone factor + tiny time drift → tails are
        stable: bottom-2={A,B}, top-2={I,J} on every date. Union of tails
        across either endpoint therefore contains exactly 4 names per
        pair — compared to all 10 in the unfiltered case.
        """
        assets = [chr(ord("A") + i) for i in range(10)]
        df = _panel(6, assets, lambda t, a: ord(a) + 0.1 * t)
        q = 0.2

        filtered = turnover(df, forward_periods=1, quantile=q)
        unfiltered = turnover(df, forward_periods=1)

        assert filtered.metadata["quantile"] == q
        assert filtered.metadata["n_cross_section_mean"] == pytest.approx(4.0)
        assert unfiltered.metadata["n_cross_section_mean"] == pytest.approx(10.0)
        assert filtered.value == pytest.approx(0.0, abs=0.01)

    def test_quantile_validation(self):
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a))
        with pytest.raises(ValueError, match="quantile"):
            turnover(df, quantile=0.6)

    def test_forward_periods_validation(self):
        df = _panel(5, ["A", "B", "C"], lambda t, a: ord(a))
        with pytest.raises(ValueError, match="forward_periods"):
            turnover(df, forward_periods=0)


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
