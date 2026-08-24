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
        for dt in ortho.data["date"].unique():
            residuals = ortho.data.filter(pl.col("date") == dt)["factor"].to_numpy()
            assert abs(np.mean(residuals)) < 1e-10

    def test_residual_uncorrelated_with_base(self):
        factor_df, base_df = _make_ortho_data()
        ortho = orthogonalize_factor(factor_df, base_df)
        merged = ortho.data.join(base_df, on=["date", "asset_id"])
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
        assert "factor_pre_ortho" in ortho.data.columns
        orig = factor_df.sort(["date", "asset_id"])["factor"].to_numpy()
        pre_ortho = ortho.data.sort(["date", "asset_id"])["factor_pre_ortho"].to_numpy()
        np.testing.assert_array_almost_equal(orig, pre_ortho)

    def test_no_base_cols_unchanged(self):
        factor_df, _ = _make_ortho_data()
        empty_base = factor_df.select("date", "asset_id")
        ortho = orthogonalize_factor(factor_df, empty_base)
        orig = factor_df.sort(["date", "asset_id"])["factor"].to_list()
        after = ortho.data.sort(["date", "asset_id"])["factor"].to_list()
        assert orig == after

    def test_attribution_betas(self):
        """factor = 0.5*size + 0.3*value + noise → betas ≈ [0.5, 0.3]."""
        factor_df, base_df = _make_ortho_data()
        ortho = orthogonalize_factor(factor_df, base_df)
        assert ortho.mean_betas["size"] == pytest.approx(0.5, abs=0.1)
        assert ortho.mean_betas["value"] == pytest.approx(0.3, abs=0.1)
        assert ortho.mean_r_squared > 0.5


class TestDuplicateBaseKeys:
    def test_duplicate_keys_raise(self):
        """A duplicated (date, asset_id) fans the inner join out silently."""
        factor_df, base_df = _make_ortho_data(n_dates=3, n_assets=5)
        dupe_base = pl.concat([base_df, base_df.head(1)])
        with pytest.raises(ValueError, match=r"duplicate \(date, asset_id\) keys"):
            orthogonalize_factor(factor_df, dupe_base)

    def test_unique_keys_keep_height_and_coverage(self):
        factor_df, base_df = _make_ortho_data(n_dates=3, n_assets=5)
        ortho = orthogonalize_factor(factor_df, base_df)
        assert ortho.data.height == factor_df.height
        assert ortho.coverage == pytest.approx(1.0)


class TestNonFiniteRows:
    def _with_nan(self, factor_df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Blank one cell on the first date to NaN."""
        first_date = factor_df["date"][0]
        return factor_df.with_columns(
            pl.when((pl.col("date") == first_date) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col(col))
            .alias(col)
        )

    def test_one_nan_does_not_null_the_whole_date(self):
        """lstsq propagated a single NaN into every beta → all-NaN residuals."""
        factor_df, base_df = _make_ortho_data(n_dates=4, n_assets=20)
        dirty = self._with_nan(factor_df, "factor")
        ortho = orthogonalize_factor(dirty, base_df)

        first_date = factor_df["date"][0]
        chunk = ortho.data.filter(pl.col("date") == first_date).sort("asset_id")
        bad = chunk.filter(pl.col("asset_id") == "A0")["factor"].to_list()
        good = chunk.filter(pl.col("asset_id") != "A0")["factor"].to_list()

        assert bad == [None]
        assert all(v is not None and np.isfinite(v) for v in good)

    def test_non_finite_base_row_is_also_excluded(self):
        factor_df, base_df = _make_ortho_data(n_dates=4, n_assets=20)
        dirty_base = self._with_nan(base_df, "size")
        ortho = orthogonalize_factor(factor_df, dirty_base)
        assert ortho.n_rows_non_finite == 1

    def test_reports_non_finite_row_count(self):
        factor_df, base_df = _make_ortho_data(n_dates=4, n_assets=20)
        dirty = self._with_nan(factor_df, "factor")
        ortho = orthogonalize_factor(dirty, base_df)
        assert ortho.n_rows_non_finite == 1
        assert ortho.n_dates_skipped == 0
        # The undefined row is excluded from coverage, not counted as done.
        assert ortho.coverage == pytest.approx(1 - 1 / factor_df.height)

    def test_null_factor_row_behaves_like_nan(self):
        factor_df, base_df = _make_ortho_data(n_dates=4, n_assets=20)
        first_date = factor_df["date"][0]
        dirty = factor_df.with_columns(
            pl.when((pl.col("date") == first_date) & (pl.col("asset_id") == "A0"))
            .then(None)
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        ortho = orthogonalize_factor(dirty, base_df)
        assert ortho.n_rows_non_finite == 1
        row = ortho.data.filter(
            (pl.col("date") == first_date) & (pl.col("asset_id") == "A0")
        )
        assert row["factor"][0] is None

    def test_too_few_finite_rows_skips_the_date(self):
        """Below len(base_cols) + 2 finite rows the date keeps its raw values."""
        # 2 base cols + intercept → 4 finite rows required. The first date is
        # one short once A0 goes non-finite; the second date clears the bar.
        factor_df, base_df = _make_ortho_data(n_dates=2, n_assets=4)
        first_date = factor_df["date"][0]
        dirty = factor_df.with_columns(
            pl.when((pl.col("date") == first_date) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        ortho = orthogonalize_factor(dirty, base_df)
        assert ortho.n_dates_skipped == 1

        kept = ortho.data.filter(pl.col("date") == first_date).sort("asset_id")
        original = dirty.filter(pl.col("date") == first_date).sort("asset_id")
        np.testing.assert_array_equal(
            kept["factor"].to_numpy(), original["factor"].to_numpy()
        )
