"""Tests for ``compute_ts_betas`` — vectorized per-asset time-series OLS.

Mirrors ``test_fm_betas.py``: single-factor behaviour (schema, column-by-column
parity with the lstsq reference, floor / degeneracy drops, pairwise-complete
null handling) plus the multi-factor batch contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix._types import EPSILON
from factrix.metrics._primitives._ts_betas import MIN_TS_OBS, compute_ts_betas

_OUT_COLS = ["asset_id", "beta", "alpha", "t_stat", "r_squared", "n_obs"]


def _common_factor_panel(n_assets: int, n_dates: int, seed: int) -> pl.DataFrame:
    """Panel with a COMMON factor F_t shared across assets and per-asset returns."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    f = rng.standard_normal(n_dates)  # common time series
    rows = []
    for a in range(n_assets):
        beta_a = rng.uniform(-2, 2)
        noise = rng.standard_normal(n_dates)
        r = beta_a * f + 0.5 * noise
        for t, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{a:03d}",
                    "factor": float(f[t]),
                    "forward_return": float(r[t]),
                }
            )
    return pl.DataFrame(rows)


def _lstsq_reference(
    df: pl.DataFrame, fc: str = "factor", rc: str = "forward_return"
) -> pl.DataFrame:
    """Per-asset slope/SE via the pre-vectorization lstsq path."""
    rows = []
    for a in df["asset_id"].unique().sort():
        c = df.filter(pl.col("asset_id") == a).sort("date")
        y = c[rc].drop_nulls().to_numpy().astype(np.float64)
        x = c[fc].drop_nulls().to_numpy().astype(np.float64)
        n = min(len(y), len(x))
        if n < MIN_TS_OBS:
            continue
        y, x = y[:n], x[:n]
        X = np.column_stack([np.ones(n), x])
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        ssr = float(resid @ resid)
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ssr / sst if sst > EPSILON else 0.0
        dof, t = n - 2, 0.0
        if dof > 0 and ssr / dof > EPSILON:
            se = float(np.sqrt((ssr / dof) * np.linalg.inv(X.T @ X)[1, 1]))
            t = b[1] / se if se > EPSILON else 0.0
        rows.append(
            {
                "asset_id": a,
                "beta": float(b[1]),
                "alpha": float(b[0]),
                "t_stat": t,
                "r_squared": r2,
                "n_obs": n,
            }
        )
    return pl.DataFrame(rows).sort("asset_id")


class TestComputeTSBetas:
    def test_returns_dict_keyed_by_factor(self):
        panel = _common_factor_panel(5, 40, seed=0)
        result = compute_ts_betas(panel)
        assert isinstance(result, dict)
        assert set(result) == {"factor"}

    def test_output_schema(self):
        df = compute_ts_betas(_common_factor_panel(6, 50, seed=1))["factor"]
        assert df.columns == _OUT_COLS
        assert df["asset_id"].is_sorted()
        assert df.schema["n_obs"] == pl.Int64

    def test_matches_lstsq_reference_columnwise(self):
        panel = _common_factor_panel(40, 120, seed=2)
        got = compute_ts_betas(panel)["factor"]
        ref = _lstsq_reference(panel)
        assert got["asset_id"].to_list() == ref["asset_id"].to_list()
        j = ref.join(got, on="asset_id", suffix="_g")
        for col in ("beta", "alpha", "t_stat", "r_squared"):
            assert j[col].to_numpy() == pytest.approx(
                j[f"{col}_g"].to_numpy(), abs=1e-10
            ), col
        assert (j["n_obs"] == j["n_obs_g"]).all()

    def test_drops_assets_below_min_obs(self):
        # GOOD has MIN_TS_OBS rows; SHORT has fewer → dropped.
        panel = _common_factor_panel(1, MIN_TS_OBS, seed=3).with_columns(
            pl.lit("GOOD").alias("asset_id")
        )
        short = panel.head(MIN_TS_OBS - 1).with_columns(
            pl.lit("SHORT").alias("asset_id")
        )
        df = compute_ts_betas(pl.concat([panel, short]))["factor"]
        assert df["asset_id"].to_list() == ["GOOD"]

    def test_drops_zero_variance_asset_without_nan(self):
        # FLAT's factor is constant over time → no identifiable slope, dropped.
        good = _common_factor_panel(1, 25, seed=4).with_columns(
            pl.lit("GOOD").alias("asset_id")
        )
        flat = good.with_columns(
            pl.lit("FLAT").alias("asset_id"), pl.lit(5.0).alias("factor")
        )
        df = compute_ts_betas(pl.concat([good, flat]))["factor"]
        assert df["asset_id"].to_list() == ["GOOD"]
        assert not df["beta"].is_nan().any()

    def test_pairwise_complete_null_handling(self):
        # A mid-series null return must drop that whole (factor, return) row
        # from both cov and var. The old per-asset path dropped nulls in x and
        # y *independently* then zipped by position — misaligning pairs after
        # the hole. The vectorized version matches a properly aligned OLS.
        panel = _common_factor_panel(1, 30, seed=5).with_columns(
            pl.lit("A").alias("asset_id")
        )
        holed = panel.with_columns(
            pl.when(pl.int_range(pl.len()) == 3)
            .then(None)
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )
        got = compute_ts_betas(holed)["factor"]

        complete = holed.drop_nulls(["factor", "forward_return"]).sort("date")
        x = complete["factor"].to_numpy()
        y = complete["forward_return"].to_numpy()
        beta_aligned = np.polyfit(x, y, 1)[0]

        assert got["n_obs"][0] == 29
        assert got["beta"][0] == pytest.approx(beta_aligned, abs=1e-10)


class TestComputeTSBetasBatch:
    def test_multi_factor_matches_list_of_one(self):
        panel = _common_factor_panel(20, 80, seed=6)
        rng = np.random.default_rng(7)
        panel = panel.with_columns(
            (pl.col("factor") * 0.5).alias("f1"),
            pl.Series("f2", rng.standard_normal(panel.height)),
        )
        cols = ["f1", "f2"]
        batch = compute_ts_betas(panel, factor_cols=cols)
        for col in cols:
            assert batch[col].equals(compute_ts_betas(panel, factor_cols=[col])[col]), (
                col
            )

    def test_empty_factor_list_rejected(self):
        with pytest.raises(ValueError, match="factor_cols must be non-empty"):
            compute_ts_betas(_common_factor_panel(3, 30, seed=8), factor_cols=[])
