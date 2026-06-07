"""Tests for factrix.metrics.k_spread (fixed-K long-short spread)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.k_spread import k_spread
from factrix.metrics.quantile import quantile_spread


def _panel_from_matrix(factor: np.ndarray, returns: np.ndarray) -> pl.DataFrame:
    """Panel from a fixed per-asset ``factor`` and a ``[n_dates, n_assets]`` returns."""
    n_dates, n_assets = returns.shape
    rows = []
    for d in range(n_dates):
        day = date(2021, 1, 1) + timedelta(days=d)
        for a in range(n_assets):
            rows.append(
                {
                    "date": day,
                    "asset_id": f"A{a}",
                    "factor": float(factor[a]),
                    "forward_return": float(returns[d, a]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _expected_spread(factor: np.ndarray, returns: np.ndarray, k: int) -> float:
    top_idx = np.argsort(factor)[-k:]
    bot_idx = np.argsort(factor)[:k]
    per_date = returns[:, top_idx].mean(axis=1) - returns[:, bot_idx].mean(axis=1)
    return float(per_date.mean())


class TestSpreadComputation:
    def test_fixed_k_selection_matches_reference(self):
        rng = np.random.default_rng(0)
        factor = np.arange(8, dtype=float)  # distinct → unambiguous ranks
        returns = rng.normal(0.0, 0.02, size=(6, 8))
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=2)

        assert result.value == pytest.approx(_expected_spread(factor, returns, k=2))
        assert result.metadata["k"] == 2
        assert result.n_obs == 6

    def test_reports_cross_sectional_dispersion(self):
        rng = np.random.default_rng(1)
        factor = np.arange(10, dtype=float)
        returns = rng.normal(0.0, 0.03, size=(8, 10))
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=3)

        expected_disp = float(np.mean(returns.std(axis=1, ddof=1)))
        assert result.metadata["cross_sectional_dispersion"] == pytest.approx(
            expected_disp
        )
        assert "top_return" in result.metadata
        assert "bottom_return" in result.metadata


class TestSmallNSignificanceSwitch:
    def test_large_cross_section_uses_t_test(self):
        rng = np.random.default_rng(2)
        factor = np.arange(40, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(30, 40))
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=5)
        assert result.metadata["method"] == "non-overlapping t-test"
        assert "p_value_t" not in result.metadata

    def test_small_cross_section_uses_block_bootstrap(self):
        rng = np.random.default_rng(3)
        factor = np.arange(20, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(40, 20))
        panel = _panel_from_matrix(factor, returns)
        result = k_spread(panel, forward_periods=1, k=5)

        assert result.metadata["method"] == "block-bootstrap CI"
        assert "p_value_t" in result.metadata  # parametric p retained for reference
        assert result.metadata["bootstrap_seed"] == 0
        # the method switch surfaces as a cross-section warning, not silently
        assert "borderline_cross_section_n" in result.warning_codes
        # reproducible run-to-run under the fixed seed
        again = k_spread(panel, forward_periods=1, k=5)
        assert result.p_value == again.p_value

    def test_large_cross_section_emits_no_switch_warning(self):
        rng = np.random.default_rng(8)
        factor = np.arange(40, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(30, 40))
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=5)
        assert result.warning_codes == ()


class TestShortCircuits:
    def test_k_too_large_for_universe(self):
        rng = np.random.default_rng(4)
        factor = np.arange(8, dtype=float)
        returns = rng.normal(0.0, 0.02, size=(10, 8))
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=5)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_assets_for_k_legs"
        assert result.metadata["max_assets_per_date"] == 8

    def test_insufficient_periods(self):
        rng = np.random.default_rng(5)
        factor = np.arange(8, dtype=float)
        returns = rng.normal(0.0, 0.02, size=(2, 8))  # 2 dates < MIN_PORTFOLIO_PERIODS
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=2)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "insufficient_portfolio_periods"

    def test_missing_return_column(self):
        factor = np.arange(8, dtype=float)
        returns = np.zeros((5, 8))
        df = _panel_from_matrix(factor, returns).drop("forward_return")
        result = k_spread(df, forward_periods=1, k=2)
        assert math.isnan(result.value)
        assert result.metadata["reason"] == "no_return_column"

    def test_invalid_k_raises(self):
        factor = np.arange(8, dtype=float)
        returns = np.zeros((5, 8))
        with pytest.raises(ValueError, match="k must be"):
            k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=0)


class TestUnderfilledDatesDropped:
    def test_dates_with_fewer_than_2k_assets_excluded(self):
        # Date 0 has 6 assets (≥ 2k=4 → kept); date 1 has 3 (< 4 → dropped).
        rows = []
        for a in range(6):
            rows.append(
                {
                    "date": date(2021, 1, 1),
                    "asset_id": f"A{a}",
                    "factor": float(a),
                    "forward_return": 0.01 * a,
                }
            )
        for a in range(3):
            rows.append(
                {
                    "date": date(2021, 1, 2),
                    "asset_id": f"A{a}",
                    "factor": float(a),
                    "forward_return": 0.5,
                }
            )
        for a in range(6):  # third qualifying date so n_periods ≥ 3
            rows.append(
                {
                    "date": date(2021, 1, 3),
                    "asset_id": f"A{a}",
                    "factor": float(a),
                    "forward_return": 0.02 * a,
                }
            )
        for a in range(6):
            rows.append(
                {
                    "date": date(2021, 1, 4),
                    "asset_id": f"A{a}",
                    "factor": float(a),
                    "forward_return": 0.03 * a,
                }
            )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = k_spread(df, forward_periods=1, k=2)
        assert result.n_obs == 3  # only the three 6-asset dates

    def test_null_factor_rows_excluded_from_leg_count(self):
        # Null factor/return rows must not inflate the per-date count: a date
        # with 5 valid names (≥ 2k=4) still qualifies, and ranks stay
        # contiguous so the bottom leg is not silently emptied.
        rows = []
        for d in range(4):
            day = date(2021, 1, 1) + timedelta(days=d)
            for a in range(8):
                f = None if (d == 1 and a in (0, 1, 2)) else float(a)
                rows.append(
                    {
                        "date": day,
                        "asset_id": f"A{a}",
                        "factor": f,
                        "forward_return": 0.01 * a + 0.001 * d,
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = k_spread(df, forward_periods=1, k=2)
        assert result.n_obs == 4  # all four dates qualify (date 1 has 5 valid)


class TestQuantileSpreadSharesPolicy:
    """The small-N bootstrap switch is shared with quantile_spread."""

    def test_quantile_spread_switches_on_small_cross_section(self):
        rng = np.random.default_rng(6)
        factor = np.arange(20, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(40, 20))
        out = quantile_spread(
            _panel_from_matrix(factor, returns), forward_periods=1, n_groups=5
        )["factor"]
        assert out.metadata["method"] == "block-bootstrap CI"
        assert "p_value_t" in out.metadata

    def test_quantile_spread_keeps_t_test_on_large_cross_section(self):
        rng = np.random.default_rng(7)
        factor = np.arange(40, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(30, 40))
        out = quantile_spread(
            _panel_from_matrix(factor, returns), forward_periods=1, n_groups=5
        )["factor"]
        assert out.metadata["method"] == "non-overlapping t-test"


class TestDispatch:
    def test_runs_via_evaluate(self):
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=40, n_dates=120)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        results = fx.evaluate(
            panel,
            metrics={"tks": k_spread(k=5)},
            factor_cols=["factor"],
            forward_periods=5,
        )
        er = results["factor"]
        assert er.metrics["tks"].name == "tks"
        assert not math.isnan(er.metrics["tks"].value)
