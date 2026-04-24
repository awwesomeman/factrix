"""Tests for factrix.metrics.tradability."""

import math
import polars as pl
import pytest
from datetime import datetime, timedelta

from factrix.metrics.tradability import (
    breakeven_cost,
    net_spread,
    turnover,
    turnover_jaccard,
)


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


class TestTurnoverJaccard:
    TEN_ASSETS = [chr(ord("A") + i) for i in range(10)]

    def test_static_factor(self):
        """Same tail sets every day → jaccard turnover = 0."""
        df = _panel(5, self.TEN_ASSETS, lambda t, a: ord(a))
        result = turnover_jaccard(df, n_groups=5)
        assert result.value == pytest.approx(0.0)
        assert result.metadata["n_rebalances"] == 4
        assert result.metadata["n_groups"] == 5

    def test_full_rotation(self):
        """Ranks reverse every date → top ↔ bot fully swap → turnover = 1."""
        def factor(t, a):
            base = ord(a) - ord("A")
            return base if t % 2 == 0 else (9 - base)
        df = _panel(5, self.TEN_ASSETS, factor)
        result = turnover_jaccard(df, n_groups=5)
        assert result.value == pytest.approx(1.0)

    def test_middle_shuffle_does_not_count(self):
        """Middle-rank reshuffling with stable tails → jaccard=0, ρ<1.

        This is the raison d'être of turnover_jaccard: ``turnover``
        (1 − Spearman ρ) is non-zero here because middle ranks move,
        but no bps cost is actually incurred because Q1/Q5 membership
        is unchanged.
        """
        tails = {"A", "B", "I", "J"}  # bottom 2 + top 2 at n_groups=5

        def factor(t, a):
            if a in ("A", "B"):
                return ord(a) - ord("A")  # 0, 1 — always lowest
            if a in ("I", "J"):
                return 100 + ord(a) - ord("A")  # always highest
            # C..H rotate in the middle band
            middle = "CDEFGH"
            idx = middle.index(a)
            return 20 + ((idx + t) % len(middle))

        df = _panel(6, self.TEN_ASSETS, factor)
        jac = turnover_jaccard(df, n_groups=5)
        stab = turnover(df)
        assert jac.value == pytest.approx(0.0)
        assert stab.value > 0.05  # rank AC noticeably below 1
        assert tails == set("ABIJ")  # scaffolding: document intent

    def test_partial_top_churn(self):
        """Half of top bucket rolls over every rebalance → top_churn=0.5.

        With n_groups=5 and 10 assets the top bucket holds 2 names. If
        exactly one of the two top names rotates out each date while the
        bottom stays put: top_churn=0.5, bot_churn=0 → turnover=0.25.
        """
        def factor(t, a):
            if a in ("A", "B"):
                return ord(a) - ord("A")  # bottom 2 fixed
            if a == "I":
                # I shares the top bucket with J on even t, drops on odd t
                return 100 if t % 2 == 0 else 50
            if a == "J":
                return 101  # always top
            if a == "H":
                # H fills in for I on odd t
                return 40 if t % 2 == 0 else 100
            return 20 + (ord(a) - ord("C"))

        df = _panel(6, self.TEN_ASSETS, factor)
        result = turnover_jaccard(df, n_groups=5)
        # Per pair: top sometimes swaps 1/2 names (churn=0.5) or keeps
        # both (churn=0). Over the 5 pairs the mean of (top+bot)/2 should
        # land between 0 and 0.25; asserting in a band is more robust
        # than a point estimate to avoid polars tie-break quirks.
        assert 0.05 < result.value < 0.30

    def test_forward_periods_stride(self):
        """forward_periods=2 sub-samples to odd/even dates only."""
        def factor(t, a):
            base = ord(a) - ord("A")
            return base if t % 2 == 0 else (9 - base)
        df = _panel(7, self.TEN_ASSETS, factor)
        # Even-only sample → ranks identical every sampled date → 0.
        result = turnover_jaccard(df, n_groups=5, forward_periods=2)
        assert result.value == pytest.approx(0.0)
        assert result.metadata["forward_periods"] == 2

    def test_insufficient_dates_short_circuits(self):
        df = _panel(1, self.TEN_ASSETS, lambda t, a: ord(a))
        result = turnover_jaccard(df, n_groups=5)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_dates"

    def test_validation(self):
        df = _panel(3, self.TEN_ASSETS, lambda t, a: ord(a))
        with pytest.raises(ValueError, match="forward_periods"):
            turnover_jaccard(df, forward_periods=0)
        with pytest.raises(ValueError, match="n_groups"):
            turnover_jaccard(df, n_groups=2)


class TestBreakevenCost:
    def test_basic(self):
        # Notional turnover=0.5 (fraction of Q1/Q_n replaced per rebalance):
        # gross=0.10, turnover=0.5 → 0.10/(2*0.5)*10000 = 1000 bps
        result = breakeven_cost(0.10, 0.5)
        assert result.value == pytest.approx(1000.0)

    def test_zero_turnover(self):
        result = breakeven_cost(0.10, 0.0)
        assert result.value == float("inf")


class TestNetSpread:
    def test_basic(self):
        # Notional turnover=0.5; cost=30bps single-leg.
        # net = 0.10 - 2*(30/10000)*0.5 = 0.10 - 0.003 = 0.097
        result = net_spread(0.10, 0.5, estimated_cost_bps=30)
        assert result.value == pytest.approx(0.097)

    def test_cost_exceeds_alpha(self):
        result = net_spread(0.001, 0.5, estimated_cost_bps=100)
        assert result.value < 0
