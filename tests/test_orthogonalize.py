"""Tests for factrix.preprocess.orthogonalize."""

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
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
        # min_residual_df=1 keeps this a coverage test: 5 names on 2 base
        # columns is far below the default df floor.
        ortho = orthogonalize_factor(factor_df, base_df, min_residual_df=1)
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
        """Below the residual-df floor the date keeps its raw values."""
        # 2 base cols + intercept → 4 finite rows required at
        # min_residual_df=1. The first date is one short once A0 goes
        # non-finite; the second date clears the bar.
        factor_df, base_df = _make_ortho_data(n_dates=2, n_assets=4)
        first_date = factor_df["date"][0]
        dirty = factor_df.with_columns(
            pl.when((pl.col("date") == first_date) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        ortho = orthogonalize_factor(dirty, base_df, min_residual_df=1)
        assert ortho.n_dates_skipped == 1

        kept = ortho.data.filter(pl.col("date") == first_date).sort("asset_id")
        original = dirty.filter(pl.col("date") == first_date).sort("asset_id")
        np.testing.assert_array_equal(
            kept["factor"].to_numpy(), original["factor"].to_numpy()
        )


class TestResidualDegreesOfFreedom:
    """A df floor, not a bare row count (finding: R2 0.79 at a true R2 of 0)."""

    @staticmethod
    def _independent_panel(n_dates: int, n_assets: int, n_base: int, seed: int = 0):
        """factor drawn independently of the base set — true R2 is exactly 0."""
        rng = np.random.default_rng(seed)
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(n_dates)]
        rows = n_dates * n_assets
        base_names = [f"b{i}" for i in range(n_base)]
        keys = {
            "date": [d for d in dates for _ in range(n_assets)],
            "asset_id": [f"A{i}" for _ in dates for i in range(n_assets)],
        }
        factor_df = pl.DataFrame({**keys, "factor": rng.standard_normal(rows)})
        base_df = pl.DataFrame(
            {**keys, **{c: rng.standard_normal(rows) for c in base_names}}
        )
        return factor_df, base_df, base_names

    def test_thin_cross_section_is_skipped_not_fitted(self):
        factor_df, base_df, base_names = self._independent_panel(
            n_dates=200, n_assets=6, n_base=4
        )
        with pytest.warns(UserWarning, match="insufficient_regression_df"):
            ortho = orthogonalize_factor(factor_df, base_df, base_cols=base_names)
        assert ortho.n_dates_insufficient_df == 200
        assert ortho.n_dates_skipped == 200
        assert ortho.coverage == 0.0
        assert WarningCode.INSUFFICIENT_REGRESSION_DF.value in ortho.warning_codes
        # The factor survives intact rather than being residualised into noise.
        np.testing.assert_allclose(
            ortho.data.sort(["date", "asset_id"])["factor"].to_numpy(),
            factor_df.sort(["date", "asset_id"])["factor"].to_numpy(),
        )

    def test_old_floor_reported_noise_as_explanatory_power(self):
        """min_residual_df=1 reproduces the defect the floor exists to stop."""
        factor_df, base_df, base_names = self._independent_panel(
            n_dates=200, n_assets=6, n_base=4
        )
        ortho = orthogonalize_factor(
            factor_df, base_df, base_cols=base_names, min_residual_df=1
        )
        # Raw R2 ~ K/(N-1) = 4/5 even though the true R2 is 0 ...
        assert ortho.mean_r_squared > 0.7
        # ... and the adjusted figure says so.
        assert abs(ortho.mean_adj_r_squared) < 0.2
        assert ortho.mean_adj_r_squared < ortho.mean_r_squared

    def test_wide_cross_section_still_fits(self):
        factor_df, base_df, base_names = self._independent_panel(
            n_dates=30, n_assets=40, n_base=3
        )
        ortho = orthogonalize_factor(factor_df, base_df, base_cols=base_names)
        assert ortho.n_dates_insufficient_df == 0
        assert ortho.coverage == pytest.approx(1.0)
        assert ortho.warning_codes == ()

    def test_floor_is_residual_df_not_row_count(self):
        """n_assets = n_base + 1 + min_residual_df is the exact boundary."""
        for n_assets, expect_skip in ((13, True), (14, False)):
            factor_df, base_df, base_names = self._independent_panel(
                n_dates=5, n_assets=n_assets, n_base=3
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ortho = orthogonalize_factor(
                    factor_df, base_df, base_cols=base_names, min_residual_df=10
                )
            assert (ortho.n_dates_insufficient_df == 5) is expect_skip


class TestRankDeficiency:
    """lstsq does not raise on a singular design; it returns min-norm betas."""

    @staticmethod
    def _dummy_panel(*, drop_reference: bool, n_dates: int = 50, n_assets: int = 20):
        rng = np.random.default_rng(3)
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(n_dates)]
        keys = {
            "date": [d for d in dates for _ in range(n_assets)],
            "asset_id": [f"A{i}" for _ in dates for i in range(n_assets)],
        }
        rows = n_dates * n_assets
        industry = np.tile(np.arange(n_assets) % 4, n_dates)
        cols = {f"ind{k}": (industry == k).astype(float) for k in range(4)}
        base_df = pl.DataFrame({**keys, **cols})
        factor_df = pl.DataFrame({**keys, "factor": rng.standard_normal(rows)})
        names = sorted(cols)
        return factor_df, base_df, names[1:] if drop_reference else names

    def test_full_dummy_set_is_detected(self):
        factor_df, base_df, base_names = self._dummy_panel(drop_reference=False)
        with pytest.warns(UserWarning, match="rank_deficient_design"):
            ortho = orthogonalize_factor(factor_df, base_df, base_cols=base_names)
        assert ortho.n_dates_rank_deficient == 50
        assert WarningCode.RANK_DEFICIENT_DESIGN.value in ortho.warning_codes
        # Unidentified betas are withheld rather than reported as attribution.
        assert ortho.mean_betas == {}

    def test_duplicated_column_is_detected(self):
        factor_df, base_df, base_names = self._dummy_panel(drop_reference=True)
        base_df = base_df.with_columns(pl.col(base_names[0]).alias("copy"))
        with pytest.warns(UserWarning, match="rank_deficient_design"):
            ortho = orthogonalize_factor(
                factor_df, base_df, base_cols=[*base_names, "copy"]
            )
        assert ortho.n_dates_rank_deficient == 50
        assert ortho.mean_betas == {}

    def test_reference_level_dropped_gives_identified_betas(self):
        factor_df, base_df, base_names = self._dummy_panel(drop_reference=True)
        ortho = orthogonalize_factor(factor_df, base_df, base_cols=base_names)
        assert ortho.n_dates_rank_deficient == 0
        assert set(ortho.mean_betas) == set(base_names)
        assert ortho.warning_codes == ()

    def test_residuals_are_identical_either_way(self):
        """The projection is unique; only the betas were arbitrary."""
        factor_df, base_df, full = self._dummy_panel(drop_reference=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deficient = orthogonalize_factor(factor_df, base_df, base_cols=full)
        reduced = orthogonalize_factor(factor_df, base_df, base_cols=full[1:])
        np.testing.assert_allclose(
            deficient.data.sort(["date", "asset_id"])["factor"].to_numpy(),
            reduced.data.sort(["date", "asset_id"])["factor"].to_numpy(),
            atol=1e-10,
        )


class TestRestandardize:
    def test_default_leaves_the_residual_on_its_own_scale(self):
        rng = np.random.default_rng(11)
        n_dates, n_assets = 40, 40
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(n_dates)]
        keys = {
            "date": [d for d in dates for _ in range(n_assets)],
            "asset_id": [f"A{i}" for _ in dates for i in range(n_assets)],
        }
        rows = n_dates * n_assets
        size = rng.standard_normal(rows)
        factor = size + rng.standard_normal(rows)
        factor_df = pl.DataFrame({**keys, "factor": factor})
        base_df = pl.DataFrame({**keys, "size": size})

        raw = orthogonalize_factor(factor_df, base_df, base_cols=["size"])
        pre = float(factor_df["factor"].std(ddof=1))
        post = float(raw.data["factor"].std(ddof=1))
        # sd(post) ~ sd(pre) * sqrt(1 - R2)
        assert post == pytest.approx(pre * (1 - raw.mean_r_squared) ** 0.5, rel=0.1)
        assert raw.restandardized is False

    def test_restandardize_restores_the_per_date_input_dispersion(self):
        rng = np.random.default_rng(12)
        n_dates, n_assets = 40, 40
        dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(n_dates)]
        keys = {
            "date": [d for d in dates for _ in range(n_assets)],
            "asset_id": [f"A{i}" for _ in dates for i in range(n_assets)],
        }
        rows = n_dates * n_assets
        size = rng.standard_normal(rows)
        factor_df = pl.DataFrame({**keys, "factor": size + rng.standard_normal(rows)})
        base_df = pl.DataFrame({**keys, "size": size})

        out = orthogonalize_factor(
            factor_df, base_df, base_cols=["size"], restandardize=True
        )
        assert out.restandardized is True
        per_date = out.data.group_by("date").agg(
            pl.col("factor").std(ddof=1).alias("post"),
            pl.col("factor_pre_ortho").std(ddof=1).alias("pre"),
        )
        np.testing.assert_allclose(
            per_date["post"].to_numpy(), per_date["pre"].to_numpy(), rtol=1e-9
        )
