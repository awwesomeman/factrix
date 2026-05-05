"""Tests for factrix.preprocess.orthogonalize."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.preprocess.orthogonalize import orthogonalize_factor


def _make_ortho_data(n_dates: int = 10, n_assets: int = 20, seed: int = 42):
    """Create factor_df and base_factors for testing."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]

    rows_factor = []
    rows_base = []
    for d in dates:
        size = rng.standard_normal(n_assets)
        value = rng.standard_normal(n_assets)
        # factor = 0.5*size + 0.3*value + noise
        factor = 0.5 * size + 0.3 * value + rng.standard_normal(n_assets) * 0.2
        for i in range(n_assets):
            aid = f"A{i}"
            rows_factor.append({"date": d, "asset_id": aid, "factor": float(factor[i])})
            rows_base.append(
                {
                    "date": d,
                    "asset_id": aid,
                    "size": float(size[i]),
                    "value": float(value[i]),
                }
            )

    factor_df = pl.DataFrame(rows_factor).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    base_df = pl.DataFrame(rows_base).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    return factor_df, base_df


class TestOrthogonalizeFactor:
    def test_residual_mean_near_zero(self):
        factor_df, base_df = _make_ortho_data()
        ortho = orthogonalize_factor(factor_df, base_df)
        for dt in ortho.df["date"].unique():
            residuals = ortho.df.filter(pl.col("date") == dt)["factor"].to_numpy()
            assert abs(np.mean(residuals)) < 1e-10

    def test_residual_uncorrelated_with_base(self):
        factor_df, base_df = _make_ortho_data()
        ortho = orthogonalize_factor(factor_df, base_df)
        merged = ortho.df.join(base_df, on=["date", "asset_id"])
        for dt in merged["date"].unique():
            chunk = merged.filter(pl.col("date") == dt)
            residual = chunk["factor"].to_numpy()
            for col in ["size", "value"]:
                base_vals = chunk[col].to_numpy()
                corr = np.corrcoef(residual, base_vals)[0, 1]
                assert abs(corr) < 1e-8

    def test_preserves_original(self):
        factor_df, base_df = _make_ortho_data()
        ortho = orthogonalize_factor(factor_df, base_df)
        assert "factor_pre_ortho" in ortho.df.columns
        orig = factor_df.sort(["date", "asset_id"])["factor"].to_numpy()
        pre_ortho = ortho.df.sort(["date", "asset_id"])["factor_pre_ortho"].to_numpy()
        np.testing.assert_array_almost_equal(orig, pre_ortho)

    def test_no_base_cols_unchanged(self):
        factor_df, _ = _make_ortho_data()
        empty_base = factor_df.select("date", "asset_id")
        ortho = orthogonalize_factor(factor_df, empty_base)
        orig = factor_df.sort(["date", "asset_id"])["factor"].to_list()
        after = ortho.df.sort(["date", "asset_id"])["factor"].to_list()
        assert orig == after

    def test_attribution_betas(self):
        """factor = 0.5*size + 0.3*value + noise → betas ≈ [0.5, 0.3]."""
        factor_df, base_df = _make_ortho_data()
        ortho = orthogonalize_factor(factor_df, base_df)
        assert ortho.mean_betas["size"] == pytest.approx(0.5, abs=0.1)
        assert ortho.mean_betas["value"] == pytest.approx(0.3, abs=0.1)
        assert ortho.mean_r_squared > 0.5
