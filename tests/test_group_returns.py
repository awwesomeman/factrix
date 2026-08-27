"""Tests for the ``compute_group_returns`` primitive."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
from factrix.metrics.quantile import compute_group_returns


def _panel(rows) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def _rows(n_dates=10, n_assets=10, factor_of=None, return_of=None):
    rows = []
    for d in range(n_dates):
        day = datetime(2024, 1, 1) + timedelta(days=d)
        for a in range(n_assets):
            rows.append(
                {
                    "date": day,
                    "asset_id": f"A{a}",
                    "factor": float(a) if factor_of is None else factor_of(a),
                    "forward_return": (0.01 * a if return_of is None else return_of(a)),
                }
            )
    return rows


class TestUnbucketedNamesExcluded:
    def test_no_null_group_row(self):
        """Null-factor names have no quantile group and must not surface as
        an extra ``group=None`` bucket."""
        out = compute_group_returns(
            _panel(_rows(factor_of=lambda a: float(a) if a < 5 else None)),
            overlap_periods=1,
            n_groups=5,
        )
        assert out["group"].null_count() == 0
        assert sorted(out["group"].to_list()) == [0, 1, 2, 3, 4]

    def test_nan_factor_is_unbucketed_too(self):
        out = compute_group_returns(
            _panel(_rows(factor_of=lambda a: float("nan") if a == 9 else float(a))),
            overlap_periods=1,
            n_groups=5,
        )
        assert out["group"].null_count() == 0
        assert out.height == 5
        # A9 (NaN) never lands in the top bucket; A8 does.
        top = out.filter(pl.col("group") == 4)["mean_return"].item()
        assert top == pytest.approx(0.08)

    def test_all_factors_null_yields_no_rows(self):
        out = compute_group_returns(
            _panel(_rows(factor_of=lambda _a: None)).with_columns(
                pl.col("factor").cast(pl.Float64)
            ),
            overlap_periods=1,
            n_groups=5,
        )
        assert out.height == 0


class TestNonFiniteReturns:
    def test_nan_return_does_not_poison_the_bucket_mean(self):
        """polars ``mean`` propagates NaN, so one bad print used to NaN out
        the whole bucket."""
        out = compute_group_returns(
            _panel(_rows(return_of=lambda a: float("nan") if a == 9 else 0.01 * a)),
            overlap_periods=1,
            n_groups=5,
        )
        assert all(math.isfinite(v) for v in out["mean_return"].to_list())
        # Top bucket holds A8 and A9; only A8's return is observed.
        assert out.filter(pl.col("group") == 4)["mean_return"].item() == pytest.approx(
            0.08
        )
