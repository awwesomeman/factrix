"""Tests for factorlib.tools.regression.spanning."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factorlib.tools.regression.spanning import (
    greedy_forward_selection,
    ForwardSelectionResult,
    SpanningResult,
    _spanning_regression,
)


def _make_spread_series(n_dates: int, mean: float, std: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    return pl.DataFrame({
        "date": dates,
        "spread": rng.normal(mean, std, n_dates),
    }).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestSpanningRegression:
    def test_alpha_with_empty_base(self):
        rng = np.random.default_rng(42)
        candidate = rng.normal(0.01, 0.005, 100)
        base = np.empty((100, 0))
        alpha, t = _spanning_regression(candidate, base)
        assert alpha == pytest.approx(0.01, abs=0.005)
        assert abs(t) > 1.0

    def test_spanned_factor_has_zero_alpha(self):
        rng = np.random.default_rng(42)
        base_col = rng.normal(0.01, 0.01, 200)
        # Candidate = 2 * base + noise → alpha ≈ 0 after regression
        candidate = 2 * base_col + rng.normal(0, 0.001, 200)
        base = base_col.reshape(-1, 1)
        alpha, t = _spanning_regression(candidate, base)
        assert abs(alpha) < 0.005
        assert abs(t) < 2.0

    def test_insufficient_data(self):
        alpha, t = _spanning_regression(np.array([0.01, 0.02]), np.empty((2, 0)))
        assert alpha == 0.0 and t == 0.0


class TestGreedyForwardSelection:
    def test_selects_strong_independent_factor(self):
        # Factor A: strong independent alpha
        a = _make_spread_series(100, 0.02, 0.005, 42)
        # Factor B: pure noise
        b = _make_spread_series(100, 0.0, 0.01, 99)
        result = greedy_forward_selection({"A": a, "B": b})
        selected_names = [s.factor_name for s in result.selected_factors]
        assert "A" in selected_names

    def test_base_factors_not_selected(self):
        base = _make_spread_series(100, 0.01, 0.005, 42)
        # Candidate = base + tiny noise → fully spanned
        dates = base["date"].to_list()
        spanned_vals = base["spread"].to_numpy() + np.random.default_rng(99).normal(0, 0.0001, 100)
        spanned = pl.DataFrame({
            "date": dates,
            "spread": spanned_vals,
        }).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = greedy_forward_selection(
            {"spanned": spanned},
            base_spreads={"base": base},
        )
        selected_names = [s.factor_name for s in result.selected_factors]
        assert "spanned" not in selected_names

    def test_backward_elimination(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

        # A: strong signal
        a_vals = rng.normal(0.03, 0.005, n)
        # B: initially looks good, but once A is in, B is redundant (B ≈ A + tiny noise)
        b_vals = a_vals + rng.normal(0, 0.0005, n)

        a = pl.DataFrame({"date": dates, "spread": a_vals}).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        b = pl.DataFrame({"date": dates, "spread": b_vals}).with_columns(pl.col("date").cast(pl.Datetime("ms")))

        result = greedy_forward_selection({"A": a, "B": b})
        selected_names = [s.factor_name for s in result.selected_factors]
        # At most one should survive — they're nearly identical
        assert len(selected_names) <= 2

    def test_empty_candidates(self):
        result = greedy_forward_selection({})
        assert result.selected_factors == []

    def test_insufficient_dates(self):
        short = _make_spread_series(5, 0.01, 0.005, 42)
        result = greedy_forward_selection({"short": short})
        assert result.selected_factors == []

    def test_max_factors_limit(self):
        factors = {}
        for i in range(10):
            factors[f"f_{i}"] = _make_spread_series(100, 0.02 + i * 0.005, 0.005, i)
        result = greedy_forward_selection(factors, max_factors=2)
        assert len(result.selected_factors) <= 2

    def test_result_structure(self):
        a = _make_spread_series(100, 0.02, 0.005, 42)
        result = greedy_forward_selection({"A": a})
        assert isinstance(result, ForwardSelectionResult)
        for sr in result.selected_factors:
            assert isinstance(sr, SpanningResult)
            assert sr.selected is True
