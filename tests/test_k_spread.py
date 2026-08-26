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
    per_period = returns[:, top_idx].mean(axis=1) - returns[:, bot_idx].mean(axis=1)
    return float(per_period.mean())


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

    def test_small_cross_section_keeps_the_t_test_and_warns(self):
        """No automatic bootstrap switch: the t stays, FEW_ASSETS is attached.

        The switch was removed after measurement — the bootstrap p rejected
        8–20% at a nominal 5% on thin cross-sections against the t's 7–9%,
        and its heavy-tail rationale had the size direction backwards.
        """
        rng = np.random.default_rng(3)
        factor = np.arange(20, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(40, 20))
        panel = _panel_from_matrix(factor, returns)
        result = k_spread(panel, forward_periods=1, k=5)

        assert result.metadata["method"] == "non-overlapping t-test"
        assert "p_value_t" not in result.metadata
        # the thin cross-section surfaces as a warning, not as a different test
        assert "few_assets" in result.warning_codes
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

    def test_constant_factor_returns_explicit_no_signal(self):
        rng = np.random.default_rng(9)
        factor = np.ones(8, dtype=float)
        returns = rng.normal(0.0, 0.02, size=(6, 8))
        result = k_spread(_panel_from_matrix(factor, returns), forward_periods=1, k=2)

        assert result.value == 0.0
        assert result.stat == 0.0
        assert result.p_value == 1.0
        assert result.is_applicable is True
        assert result.metadata["signal_status"] == "no_signal_zero_variance_factor"


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
        # Null factor/return rows must not inflate the per-period count: a date
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
    """The small-N policy (warn, keep the t) is shared with quantile_spread."""

    def test_quantile_spread_warns_without_switching_on_small_cross_section(self):
        from factrix._codes import WarningCode

        rng = np.random.default_rng(6)
        factor = np.arange(20, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(40, 20))
        out = quantile_spread(
            _panel_from_matrix(factor, returns), forward_periods=1, n_groups=5
        )["factor"]
        assert out.metadata["method"] == "non-overlapping t-test"
        assert WarningCode.FEW_ASSETS.value in out.warning_codes

    def test_quantile_spread_keeps_t_test_on_large_cross_section(self):
        rng = np.random.default_rng(7)
        factor = np.arange(40, dtype=float)
        returns = rng.normal(0.001, 0.02, size=(30, 40))
        out = quantile_spread(
            _panel_from_matrix(factor, returns), forward_periods=1, n_groups=5
        )["factor"]
        assert out.metadata["method"] == "non-overlapping t-test"


class TestInference:
    """The ``inference=`` knob: bit-for-bit default, HAC opt-in, bootstrap guard."""

    @staticmethod
    def _ample_panel():
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=60, n_dates=400, seed=3)
        return fx.preprocess.compute_forward_return(raw, forward_periods=5)

    def test_explicit_non_overlapping_is_bit_for_bit_default(self):
        import factrix as fx

        panel = self._ample_panel()
        default = k_spread(panel, forward_periods=5, k=5)
        explicit = k_spread(
            panel, forward_periods=5, k=5, inference=fx.inference.NON_OVERLAPPING
        )
        assert explicit.value == default.value
        assert explicit.p_value == default.p_value
        assert explicit.stat == default.stat
        assert explicit.metadata["method"] == "non-overlapping t-test"

    def test_newey_west_runs_hac_on_full_series(self):
        import factrix as fx

        panel = self._ample_panel()
        nw = k_spread(panel, forward_periods=5, k=5, inference=fx.inference.NEWEY_WEST)
        assert nw.metadata["method"] == "Newey-West HAC t-test"
        assert "nw_lags" in nw.metadata
        # HAC keeps every date; the full series is longer than the strided one,
        # and n_obs / n_periods must describe the sample the test ran on.
        assert nw.metadata["n_periods_full"] > nw.metadata["n_periods_strided"]
        assert nw.metadata["n_periods"] == nw.metadata["n_periods_full"]
        assert nw.n_obs == nw.metadata["n_periods_full"]

    def test_small_cross_section_keeps_requested_hac_and_warns(self):
        import factrix as fx
        from factrix._codes import WarningCode

        raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=400, seed=4)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        nw = k_spread(panel, forward_periods=5, k=3, inference=fx.inference.NEWEY_WEST)
        assert nw.metadata["method"] == "Newey-West HAC t-test"
        assert "inference_overridden" not in nw.metadata
        assert WarningCode.FEW_ASSETS.value in nw.warning_codes

    def test_unapplicable_inference_raises_not_silent_fallback(self):
        import factrix as fx

        panel = self._ample_panel()
        with pytest.raises(fx.IncompatibleInferenceError) as exc:
            k_spread(
                panel, forward_periods=5, k=5, inference=fx.inference.HANSEN_HODRICK
            )
        assert exc.value.func_name == "k_spread"
        assert exc.value.applicable == ("NeweyWest", "NonOverlapping")

    def test_non_inference_object_raises(self):
        import factrix as fx

        panel = self._ample_panel()
        with pytest.raises(fx.IncompatibleInferenceError):
            k_spread(panel, forward_periods=5, k=5, inference="newey")


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


class TestNonFiniteFactors:
    """``rank(descending=True)`` sorts NaN as the largest value."""

    @staticmethod
    def _panel(bad_factor):
        rows = []
        for d in range(60):
            day = date(2021, 1, 1) + timedelta(days=d)
            for a in range(10):
                # Factor and return are perfectly aligned: the top-2 names
                # (A8, A9) earn +0.10, the bottom-2 (A0, A1) earn -0.10.
                rows.append(
                    {
                        "date": day,
                        "asset_id": f"A{a}",
                        "factor": bad_factor if a == 5 else float(a),
                        "forward_return": 0.10 if a >= 8 else (-0.10 if a < 2 else 0.0),
                    }
                )
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_nan_factor_does_not_lead_the_long_leg(self):
        """A NaN factor used to take rank 1 and drag a 0.0-return name into
        the long leg, halving the measured spread."""
        clean = k_spread(self._panel(5.0), forward_periods=1, k=2)
        with_nan = k_spread(self._panel(float("nan")), forward_periods=1, k=2)
        assert clean.value == pytest.approx(0.20)
        assert with_nan.value == pytest.approx(0.20)
        assert with_nan.metadata["top_return"] == pytest.approx(0.10)

    def test_null_and_nan_factors_are_treated_alike(self):
        nan_result = k_spread(self._panel(float("nan")), forward_periods=1, k=2)
        null_result = k_spread(self._panel(None), forward_periods=1, k=2)
        assert nan_result.value == pytest.approx(null_result.value)
        assert nan_result.metadata["median_cross_section"] == 9

    def test_nan_return_does_not_poison_the_leg_mean(self):
        rows = []
        for d in range(60):
            day = date(2021, 1, 1) + timedelta(days=d)
            for a in range(10):
                ret = 0.10 if a >= 8 else (-0.10 if a < 2 else 0.0)
                rows.append(
                    {
                        "date": day,
                        "asset_id": f"A{a}",
                        "factor": float(a),
                        # A middle name (never in a leg) has a NaN return; it
                        # would still NaN out xs_dispersion.
                        "forward_return": float("nan") if a == 5 else ret,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = k_spread(panel, forward_periods=1, k=2)
        assert result.value == pytest.approx(0.20)
        assert math.isfinite(result.metadata["cross_sectional_dispersion"])


class TestSmallCrossSectionKeying:
    def test_rotating_universe_still_counts_as_thin(self):
        from factrix._codes import WarningCode

        """12 names per period, 720 distinct asset_ids over the sample.

        The advisory used to read ``asset_id.n_unique()`` over the whole panel
        (720 -> "wide", no warning); the thin-cross-section rationale is per
        date, where only 12 names back each leg.
        """
        rng = np.random.default_rng(7)
        rows = []
        for d in range(60):
            day = date(2021, 1, 1) + timedelta(days=d)
            for a in range(12):
                rows.append(
                    {
                        "date": day,
                        "asset_id": f"A{d * 12 + a:05d}",
                        "factor": float(rng.normal()),
                        "forward_return": float(rng.normal()) * 0.02,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = k_spread(panel, forward_periods=1, k=3)
        assert result.metadata["median_cross_section"] == 12
        assert WarningCode.FEW_ASSETS.value in result.warning_codes


class TestKSpreadTiePolicy:
    """A discrete signal must not be split into legs by row order alone."""

    @staticmethod
    def _ternary_panel(n_dates=60, n_assets=30, seed=0):
        from datetime import datetime, timedelta

        import numpy as np

        rng = np.random.default_rng(seed)
        rows = n_dates * n_assets
        return pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1) + timedelta(days=d)
                    for d in range(n_dates)
                    for _ in range(n_assets)
                ],
                "asset_id": [f"A{i}" for _ in range(n_dates) for i in range(n_assets)],
                # Three levels: every leg of size k=5 is filled from a tied block.
                "factor": rng.integers(-1, 2, rows).astype(float),
                "forward_return": rng.standard_normal(rows) * 0.01,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    def test_tie_ratio_is_reported(self):
        panel = self._ternary_panel()
        with pytest.warns(UserWarning, match="tie_ratio"):
            out = k_spread(panel, forward_periods=1, k=5)
        assert out.metadata["tie_ratio"] > 0.8
        assert out.metadata["tie_policy"] == "ordinal"

    def test_average_policy_refuses_to_invent_a_split(self):
        """A 10-way tie cannot yield a 5-name leg without an arbitrary rule.

        Under ``"average"`` every name in a tied block shares one rank, so a
        block wider than ``k`` puts no name inside the leg cutoff and the date
        drops out — surfaced by the existing drop-rate advisory. That is the
        honest answer; ``"ordinal"`` returns a number for the same date by
        filling the leg in row order.
        """
        import warnings

        panel = self._ternary_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ordinal = k_spread(panel, forward_periods=1, k=5)
            average = k_spread(panel, forward_periods=1, k=5, tie_policy="average")
        assert average.metadata["tie_policy"] == "average"
        assert ordinal.value != average.value
        assert average.n_obs < ordinal.n_obs

    def test_continuous_factor_does_not_warn(self, noisy_panel):
        out = k_spread(noisy_panel, forward_periods=1, k=5)
        assert out.metadata["tie_ratio"] == pytest.approx(0.0)
        assert out.metadata["tie_policy"] == "ordinal"
