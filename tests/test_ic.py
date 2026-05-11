"""Tests for factrix.metrics.ic."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.ic import compute_ic, ic, ic_ir


class TestComputeIC:
    def test_perfect_rank(self, tiny_panel):
        result = compute_ic(tiny_panel)
        # factor and return have identical ranking → IC = 1.0
        for ic_val in result["ic"].to_list():
            assert ic_val == pytest.approx(1.0)

    def test_inverse_rank(self, tiny_panel):
        # Reverse the returns
        inverted = tiny_panel.with_columns(
            (0.06 - pl.col("forward_return")).alias("forward_return")
        )
        result = compute_ic(inverted)
        for ic_val in result["ic"].to_list():
            assert ic_val == pytest.approx(-1.0)

    def test_drops_small_dates(self):
        """Dates with < MIN_ASSETS_PER_DATE_IC assets should be excluded."""
        # 3 assets < MIN_ASSETS_PER_DATE_IC=10
        dates = [datetime(2024, 1, 1)]
        rows = [
            {
                "date": dates[0],
                "asset_id": f"A{i}",
                "factor": float(i),
                "forward_return": float(i) * 0.01,
            }
            for i in range(3)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)
        assert len(result) == 0

    def test_output_schema(self, noisy_panel):
        result = compute_ic(noisy_panel)
        assert result.columns == ["date", "ic", "tie_ratio"]
        assert result["date"].dtype == pl.Datetime("ms")

    def test_tie_ratio_zero_on_unique_factor(self, noisy_panel):
        result = compute_ic(noisy_panel)
        # noisy_panel factor is continuous noise — no per-date ties expected.
        assert result["tie_ratio"].max() == pytest.approx(0.0)

    def test_tie_ratio_detects_bucketed_factor(self):
        """Bucketed factor → tie_ratio surfaces non-trivially per date."""
        n_assets = 12
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(5)]
        rows = [
            {
                "date": dt,
                "asset_id": f"A{i}",
                "factor": float(i % 3),  # 3 buckets → ties
                "forward_return": float(i) * 0.01,
            }
            for dt in dates
            for i in range(n_assets)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = compute_ic(df)
        # 12 obs, 3 unique → tie_ratio = 1 - 3/12 = 0.75
        assert result["tie_ratio"].max() == pytest.approx(0.75)
        assert result["tie_ratio"].min() == pytest.approx(0.75)

    def test_tie_ratio_propagated_to_metadata(self, noisy_panel):
        from factrix.metrics.ic import ic, ic_ir, ic_newey_west

        ic_df = compute_ic(noisy_panel)
        for out in (
            ic(ic_df, forward_periods=1),
            ic_newey_west(ic_df, forward_periods=1),
            ic_ir(ic_df),
        ):
            assert "tie_ratio" in out.metadata
            assert 0.0 <= out.metadata["tie_ratio"] <= 1.0

    def test_high_tie_ratio_emits_warning(self):
        """ic / ic_newey_west / ic_ir warn when median tie_ratio > threshold."""
        import warnings

        from factrix.metrics.ic import ic, ic_ir, ic_newey_west

        # 12 assets bucketed into 2 buckets per date → tie_ratio = 1 - 2/12
        # ≈ 0.83 (well above the 0.3 threshold).
        n_assets = 12
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(40)]
        rows = [
            {
                "date": dt,
                "asset_id": f"A{i}",
                "factor": float(i % 2),
                "forward_return": float(i) * 0.01,
            }
            for dt in dates
            for i in range(n_assets)
        ]
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        ic_df = compute_ic(df)
        for fn in (
            lambda d: ic(d, forward_periods=1),
            lambda d: ic_newey_west(d, forward_periods=1),
            ic_ir,
        ):
            with pytest.warns(UserWarning, match="tie_ratio"):
                fn(ic_df)

        # Low-tie panel must not trigger the warning.
        rng = np.random.default_rng(0)
        clean_rows = [
            {
                "date": dt,
                "asset_id": f"A{i}",
                "factor": float(rng.standard_normal()),
                "forward_return": float(rng.standard_normal()) * 0.01,
            }
            for dt in dates
            for i in range(n_assets)
        ]
        clean = pl.DataFrame(clean_rows).with_columns(
            pl.col("date").cast(pl.Datetime("ms"))
        )
        clean_ic = compute_ic(clean)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ic(clean_ic, forward_periods=1)
            ic_newey_west(clean_ic, forward_periods=1)
            ic_ir(clean_ic)


class TestIC:
    def test_positive_ic(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)
        result = ic(ic_df, forward_periods=1)
        assert result.name == "ic"
        assert result.value > 0  # noisy_panel has positive IC
        assert result.stat > 0
        assert result.significance != ""

    def test_insufficient_periods(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "ic": [0.05, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic(df, forward_periods=1)
        assert math.isnan(result.value)


class TestICIR:
    def test_positive_ir(self, noisy_panel):
        ic_df = compute_ic(noisy_panel)
        result = ic_ir(ic_df)
        assert result.value > 0
        assert result.name == "ic_ir"
        assert result.stat is None
        assert "mean_ic" in result.metadata

    def test_insufficient_periods(self):
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "ic": [0.05, 0.03, 0.04],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = ic_ir(df)
        assert math.isnan(result.value)
