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
from factrix.metrics._primitives._ts_betas import MIN_TS_PERIODS, compute_ts_betas
from factrix.metrics.ts_beta import compute_rolling_mean_beta

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
        if n < MIN_TS_PERIODS:
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
        # Analysis columns, plus the broadcast assets-axis drop-stat carrier.
        assert df.columns == [*_OUT_COLS, "_drop_stats"]
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
        # GOOD has MIN_TS_PERIODS rows; SHORT has fewer → dropped.
        panel = _common_factor_panel(1, MIN_TS_PERIODS, seed=3).with_columns(
            pl.lit("GOOD").alias("asset_id")
        )
        short = panel.head(MIN_TS_PERIODS - 1).with_columns(
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


def _rolling_mean_beta_reference(
    df: pl.DataFrame, window: int, fc: str = "factor", rc: str = "forward_return"
) -> pl.DataFrame:
    """Brute-force rolling mean beta: trailing-window date membership, per-asset
    OLS on the complete (factor, return) pairs, cross-asset mean. Independent of
    the optimized searchsorted/partition implementation.
    """
    dates = df["date"].unique().sort()
    rows: list[dict] = []
    for i in range(window, len(dates)):
        window_dates = dates[i - window : i]
        chunk = df.filter(pl.col("date").is_in(window_dates.implode()))
        betas: list[float] = []
        for a in chunk["asset_id"].unique():
            c = chunk.filter(
                (pl.col("asset_id") == a)
                & pl.col(fc).is_not_null()
                & pl.col(rc).is_not_null()
            )
            x = c[fc].to_numpy().astype(np.float64)
            y = c[rc].to_numpy().astype(np.float64)
            if len(x) < 10:
                continue
            X = np.column_stack([np.ones(len(x)), x])
            b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            betas.append(float(b[1]))
        if betas:
            rows.append({"date": dates[i], "value": float(np.mean(betas))})
    return pl.DataFrame(rows, schema={"date": dates.dtype, "value": pl.Float64}).sort(
        "date"
    )


class TestRollingMeanBeta:
    def test_matches_bruteforce_reference_on_sparse_panel(self):
        # Sparse panel (assets miss random dates) — the optimized window slicing
        # must agree with the trailing-date-membership reference.
        rng = np.random.default_rng(11)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
        rows = []
        for d in dates:
            for a in range(6):
                if rng.random() < 0.1:  # drop ~10% of rows
                    continue
                f = rng.standard_normal()
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{a}",
                        "factor": float(f),
                        "forward_return": float(2.0 * f + 0.5 * rng.standard_normal()),
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        got = compute_rolling_mean_beta(df, window=30).sort("date")
        ref = _rolling_mean_beta_reference(df, window=30)
        assert got["date"].to_list() == ref["date"].to_list()
        assert np.allclose(got["value"].to_numpy(), ref["value"].to_numpy(), atol=1e-9)

    def test_null_return_pairs_do_not_poison_beta(self):
        # A null factor/return must be dropped from the per-asset OLS, not fed in
        # as NaN (which would poison the slope and the cross-asset mean).
        rng = np.random.default_rng(3)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        rows = []
        for di, d in enumerate(dates):
            for a in range(5):
                f = rng.standard_normal()
                r = 2.0 * f + 0.5 * rng.standard_normal()
                # scatter a few null returns through every asset's window
                if di % 17 == a:
                    r = None
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{a}",
                        "factor": float(f),
                        "forward_return": None if r is None else float(r),
                    }
                )
        df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))
        got = compute_rolling_mean_beta(df, window=30)
        vals = got["value"].to_numpy()
        assert got.height > 0
        assert not np.any(np.isnan(vals))
        assert np.allclose(
            vals,
            _rolling_mean_beta_reference(df, window=30)["value"].to_numpy(),
            atol=1e-9,
        )
