"""Tests for ``compute_fm_betas`` — vectorized per-date cross-sectional OLS slope.

Mirrors ``test_ic.py``: single-factor behaviour (schema, closed-form value,
small-date / zero-variance drops) plus the multi-factor batch contract (each
batch element equals the corresponding list-of-one call).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from factrix.metrics.fm_beta import compute_fm_betas


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
        # the primitive; the public series columns are ``date, beta``.
        assert df.columns == ["date", "beta", "_drop_stats"]
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
