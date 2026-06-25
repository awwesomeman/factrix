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
        series = compute_spread_series(tiny_panel, forward_periods=1, n_groups=5)[
            "factor"
        ]
        assert "spread" in series.columns
        assert "top_return" in series.columns
        assert "bottom_return" in series.columns
        # factor=[1..5], return=[0.01..0.05], 5 groups → q1=0.05, q5=0.01
        for row in series.iter_rows(named=True):
            assert row["top_return"] == pytest.approx(0.05)
            assert row["bottom_return"] == pytest.approx(0.01)
            assert row["spread"] == pytest.approx(0.04)


class TestComputeSpreadSeriesBatch:
    """Multi-factor path of ``compute_spread_series``."""

    def test_multi_factor_columns_match_list_of_one(self):
        rng = np.random.default_rng(11)
        n_assets, n_dates = 100, 60
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
        batch = compute_spread_series(
            panel, forward_periods=1, n_groups=5, factor_cols=["f1", "f2"]
        )
        for col in ("f1", "f2"):
            single = compute_spread_series(
                panel, forward_periods=1, n_groups=5, factor_cols=[col]
            )[col]
            assert batch[col].equals(single), col

    def test_empty_factor_list_rejected(self, tiny_panel):
        with pytest.raises(ValueError, match="non-empty"):
            compute_spread_series(
                tiny_panel, forward_periods=1, n_groups=5, factor_cols=[]
            )

    def test_two_group_small_cross_section_warning_is_actionable(self, tiny_panel):
        with pytest.warns(UserWarning, match="coarsest long-short split") as caught:
            compute_spread_series(tiny_panel, forward_periods=1, n_groups=2)

        assert "Consider reducing n_groups" not in str(caught[0].message)


class TestQuantileSpread:
    def test_noisy_panel(self, noisy_panel):
        series = compute_spread_series(noisy_panel, forward_periods=1, n_groups=5)[
            "factor"
        ]
        assert len(series) >= 5
        assert series["spread"].null_count() == 0

    def test_insufficient_periods(self):
        from datetime import datetime

        import polars as pl

        # 2 dates < MIN_PORTFOLIO_PERIODS_HARD=3
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)] * 5 + [datetime(2024, 1, 2)] * 5,
                "asset_id": ["A", "B", "C", "D", "E"] * 2,
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
                "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05] * 2,
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = quantile_spread(df, forward_periods=1, n_groups=5)["factor"]
        assert math.isnan(result.value)

    def test_decomposition_in_metadata(self, tiny_panel):
        """spread = long_alpha + short_alpha (per-period)."""
        series = compute_spread_series(tiny_panel, forward_periods=1, n_groups=5)[
            "factor"
        ]
        for row in series.iter_rows(named=True):
            long = row["top_return"] - row["universe_return"]
            short = row["universe_return"] - row["bottom_return"]
            assert long + short == pytest.approx(row["spread"])

    def test_metadata_has_long_short(self, noisy_panel):
        result = quantile_spread(noisy_panel, forward_periods=1, n_groups=5)["factor"]
        if result.value != 0.0:
            assert "long_alpha" in result.metadata
            assert "short_alpha" in result.metadata
            assert "long_stat" in result.metadata
            assert "short_stat" in result.metadata
            assert result.p_value is not None


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
        # With density, VW spread should be nonzero
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


class TestQuantileSpreadInference:
    """The ``inference=`` knob mirrors k_spread: default bit-for-bit, HAC opt-in."""

    @staticmethod
    def _ample_panel():
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=80, n_dates=400, seed=5)
        return fx.preprocess.compute_forward_return(raw, forward_periods=5)

    def test_explicit_non_overlapping_is_bit_for_bit_default(self):
        import factrix as fx

        panel = self._ample_panel()
        default = quantile_spread(panel, forward_periods=5, n_groups=5)["factor"]
        explicit = quantile_spread(
            panel, forward_periods=5, n_groups=5, inference=fx.inference.NON_OVERLAPPING
        )["factor"]
        assert explicit.value == default.value
        assert explicit.p_value == default.p_value
        assert explicit.stat == default.stat
        assert explicit.metadata["method"] == "non-overlapping t-test"

    def test_newey_west_runs_hac_on_full_series(self):
        import factrix as fx

        panel = self._ample_panel()
        nw = quantile_spread(
            panel, forward_periods=5, n_groups=5, inference=fx.inference.NEWEY_WEST
        )["factor"]
        assert nw.metadata["method"] == "Newey-West HAC t-test"
        assert "nw_lags" in nw.metadata
        assert nw.metadata["n_periods_full"] > nw.metadata["n_periods"]

    def test_small_cross_section_bootstrap_overrides_requested_hac(self):
        import factrix as fx

        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=400, seed=6)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        nw = quantile_spread(
            panel, forward_periods=5, n_groups=5, inference=fx.inference.NEWEY_WEST
        )["factor"]
        assert nw.metadata["method"] == "block-bootstrap CI"
        assert nw.metadata["inference_overridden"] is True
        assert nw.metadata["inference_requested"] == "Newey-West HAC t-test"

    def test_unapplicable_inference_raises_not_silent_fallback(self):
        import factrix as fx

        panel = self._ample_panel()
        # HansenHodrick used to be silently swallowed -> non-overlap t-test.
        with pytest.raises(fx.IncompatibleInferenceError) as exc:
            quantile_spread(
                panel,
                forward_periods=5,
                n_groups=5,
                inference=fx.inference.HANSEN_HODRICK,
            )
        assert exc.value.func_name == "quantile_spread"
        assert exc.value.applicable == ("NeweyWest", "NonOverlapping")

    def test_non_inference_object_raises(self):
        import factrix as fx

        panel = self._ample_panel()
        with pytest.raises(fx.IncompatibleInferenceError):
            quantile_spread(panel, forward_periods=5, n_groups=5, inference="newey")
