"""Tests for ``compute_fm_betas`` — vectorized per-date cross-sectional OLS slope.

Mirrors ``test_ic.py``: single-factor behaviour (schema, closed-form value,
small-date / zero-variance drops) plus the multi-factor batch contract (each
batch element equals the corresponding list-of-one call).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._codes import WarningCode
from factrix.metrics.fm_beta import (
    compute_fm_betas,
    fm_beta,
    fm_beta_sign_consistency,
    pooled_beta,
)


def _lstsq_betas(df: pl.DataFrame, factor_col: str = "factor") -> dict:
    """Reference per-date slope via the pre-vectorization lstsq path."""
    out = {}
    for dt in df["date"].unique().sort():
        chunk = df.filter(pl.col("date") == dt)
        y = chunk["forward_return"].to_numpy().astype(np.float64)
        x = chunk[factor_col].to_numpy().astype(np.float64)
        if len(y) < 3:
            continue
        beta, _, _, _ = np.linalg.lstsq(
            np.column_stack([np.ones(len(x)), x]), y, rcond=None
        )
        out[dt] = float(beta[1])
    return out


class TestComputeFMBetas:
    def test_returns_dict_keyed_by_factor(self, tiny_panel):
        result = compute_fm_betas(tiny_panel)
        assert isinstance(result, dict)
        assert set(result) == {"factor"}

    def test_output_schema(self, noisy_panel):
        df = compute_fm_betas(noisy_panel)["factor"]
        # ``_drop_stats`` is an internal diagnostic struct column appended by
        # the primitive; the public series columns are ``date, beta, n_assets``.
        assert df.columns == ["date", "beta", "n_assets", "_drop_stats"]
        assert df["date"].is_sorted()

    def test_closed_form_value(self, tiny_panel):
        # forward_return = 0.01 * factor exactly → per-date slope == 0.01.
        df = compute_fm_betas(tiny_panel)["factor"]
        assert df.height == 3
        assert df["beta"].to_numpy() == pytest.approx(0.01, abs=1e-12)

    def test_matches_lstsq_reference(self, noisy_panel):
        df = compute_fm_betas(noisy_panel)["factor"]
        ref = _lstsq_betas(noisy_panel)
        got = dict(zip(df["date"].to_list(), df["beta"].to_list(), strict=True))
        assert got.keys() == ref.keys()
        for dt, beta in got.items():
            assert beta == pytest.approx(ref[dt], abs=1e-12)

    def test_drops_dates_below_min_obs(self):
        # date d0 has 2 assets (below MIN_FM_ASSETS_HARD=3), d1 has 4.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "A",
                "factor": 1.0,
                "forward_return": 0.1,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "B",
                "factor": 2.0,
                "forward_return": 0.2,
            },
        ] + [
            {
                "date": datetime(2024, 1, 2),
                "asset_id": a,
                "factor": f,
                "forward_return": 0.05 * f,
            }
            for a, f in zip("ABCD", (1.0, 2.0, 3.0, 4.0), strict=True)
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df["date"].to_list() == [datetime(2024, 1, 2)]
        assert df["n_assets"].to_list() == [4]

    def test_drops_zero_variance_dates(self):
        # date d0: factor constant → no identifiable slope, dropped.
        # date d1: ordinary spread → kept.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": a,
                "factor": 5.0,
                "forward_return": r,
            }
            for a, r in zip("ABC", (0.1, 0.2, 0.3), strict=True)
        ] + [
            {
                "date": datetime(2024, 1, 2),
                "asset_id": a,
                "factor": f,
                "forward_return": 0.05 * f,
            }
            for a, f in zip("ABC", (1.0, 2.0, 3.0), strict=True)
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df["date"].to_list() == [datetime(2024, 1, 2)]
        assert df["beta"].is_finite().all()

    def test_null_return_uses_pairwise_complete_slope(self):
        # One asset has a null return: cov drops the pair, and var(factor)
        # must drop it too so the slope is the OLS fit on complete pairs
        # (forward_return = 0.1 * factor over the 3 complete rows → 0.1),
        # not a numerator/denominator-mismatched value.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "A",
                "factor": 1.0,
                "forward_return": 0.1,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "B",
                "factor": 2.0,
                "forward_return": 0.2,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "C",
                "factor": 3.0,
                "forward_return": 0.3,
            },
            {
                "date": datetime(2024, 1, 1),
                "asset_id": "D",
                "factor": 4.0,
                "forward_return": None,
            },
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df["beta"].to_numpy() == pytest.approx(0.1, abs=1e-12)

    def test_tiny_scale_variance_is_not_dropped(self):
        # Variance of a 1e-5-scale factor is ~1e-10; a fixed absolute epsilon
        # would wrongly discard this legitimately dispersed date. The slope is
        # scale-free: forward_return = 1e4 * factor.
        rows = [
            {
                "date": datetime(2024, 1, 1),
                "asset_id": a,
                "factor": f,
                "forward_return": 1e4 * f,
            }
            for a, f in zip("ABC", (1e-5, 2e-5, 3e-5), strict=True)
        ]
        df = compute_fm_betas(pl.DataFrame(rows))["factor"]
        assert df.height == 1
        assert df["beta"][0] == pytest.approx(1e4, rel=1e-9)


class TestFMBetaConsumers:
    @staticmethod
    def _thin_beta_df(n_dates: int = 35, n_assets: int = 8) -> pl.DataFrame:
        rows = []
        for d in range(n_dates):
            date = datetime(2024, 1, 1) + timedelta(days=d)
            for i in range(n_assets):
                factor = float(i + 1)
                rows.append(
                    {
                        "date": date,
                        "asset_id": f"A{i}",
                        "factor": factor,
                        "forward_return": 0.01 * factor,
                    }
                )
        panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        return compute_fm_betas(panel)["factor"]

    def test_few_assets_warns_without_blocking(self):
        beta_df = self._thin_beta_df()
        with pytest.warns(UserWarning, match="MIN_FM_ASSETS_WARN"):
            result = fm_beta(beta_df)
        assert not np.isnan(result.value)
        assert WarningCode.FEW_ASSETS.value in result.warning_codes
        assert result.metadata["min_assets_per_period"] == 8
        assert result.metadata["warn_assets_per_period"] == 10

    def test_sign_consistency_surfaces_few_assets(self):
        beta_df = self._thin_beta_df()
        with pytest.warns(UserWarning, match="MIN_FM_ASSETS_WARN"):
            result = fm_beta_sign_consistency(beta_df)
        assert not np.isnan(result.value)
        assert WarningCode.FEW_ASSETS.value in result.warning_codes


class TestComputeFMBetasBatch:
    """Multi-factor path — each batch element equals the list-of-one call."""

    def test_multi_factor_matches_list_of_one(self, noisy_panel):
        rng = np.random.default_rng(7)
        panel = noisy_panel.with_columns(
            (pl.col("factor") * 1.5).alias("f1"),
            pl.Series("f2", rng.standard_normal(noisy_panel.height)),
        )
        cols = ["f1", "f2"]
        batch = compute_fm_betas(panel, factor_cols=cols)
        for col in cols:
            single = compute_fm_betas(panel, factor_cols=[col])[col]
            assert batch[col].equals(single), col

    def test_empty_factor_list_rejected(self, tiny_panel):
        with pytest.raises(ValueError, match="factor_cols must be non-empty"):
            compute_fm_betas(tiny_panel, factor_cols=[])


def _nan_panel(n_dates: int = 30, n_assets: int = 8, seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for a in range(n_assets):
            f = float(rng.normal())
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{a}",
                    "factor": f,
                    "forward_return": 0.5 * f + float(rng.normal(0, 0.1)),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


class TestNonFinitePassThrough:
    """REGRESSION: polars var/cov/drop_nulls do not skip float NaN."""

    def test_single_nan_return_does_not_emit_nan_beta(self):
        panel = _nan_panel()
        target = panel["date"][0]
        poisoned = panel.with_columns(
            pl.when((pl.col("date") == target) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        out = compute_fm_betas(poisoned)["factor"]
        betas = out["beta"].to_numpy()
        # Old code: var(x) > 0 stayed True while cov went NaN -> NaN beta that
        # survived drop_nulls("beta").
        assert np.isfinite(betas).all()
        # The NaN row is excluded from the effective n for that date.
        row = out.filter(pl.col("date") == target)
        assert row.height == 1
        assert row["n_assets"][0] == 7

    def test_single_nan_factor_does_not_emit_nan_beta(self):
        panel = _nan_panel()
        poisoned = panel.with_columns(
            pl.when(pl.int_range(pl.len()) == 3)
            .then(float("nan"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        out = compute_fm_betas(poisoned)["factor"]
        assert np.isfinite(out["beta"].to_numpy()).all()

    def test_date_below_hard_floor_after_nan_drop_is_removed(self):
        panel = _nan_panel(n_dates=12, n_assets=3)
        target = panel["date"][0]
        poisoned = panel.with_columns(
            pl.when((pl.col("date") == target) & (pl.col("asset_id") == "A0"))
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        out = compute_fm_betas(poisoned)["factor"]
        # 3 assets - 1 non-finite = 2 finite pairs < MIN_FM_ASSETS_HARD.
        assert target not in out["date"].to_list()
        stats = out["_drop_stats"][0]
        assert stats["dropped_periods"] >= 1

    def test_pooled_beta_survives_a_nan_cell(self):
        panel = _nan_panel()
        poisoned = panel.with_columns(
            pl.when(pl.int_range(pl.len()) == 11)
            .then(float("nan"))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        # Old code: drop_nulls kept the NaN -> lstsq slope NaN -> MetricResult
        # rejected the non-finite value with a ValueError.
        result = pooled_beta(poisoned)
        assert np.isfinite(result.value)
        assert result.metadata["dropped_pairs"] == 1
        assert result.n_obs == poisoned.height - 1

    def test_sign_consistency_ignores_nan_betas(self):
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        beta_df = pl.DataFrame(
            {
                "date": dates,
                "beta": [1.0, 2.0, float("nan"), 3.0, 4.0],
                "n_assets": [10, 10, 10, 10, 10],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        result = fm_beta_sign_consistency(beta_df, expected_sign=1)
        # Old code: NaN > 0 is False -> counted as a wrong-sign period AND in
        # n_obs, giving 4/5 = 0.8 instead of 4/4 = 1.0.
        assert result.value == pytest.approx(1.0)
        assert result.n_obs == 4
