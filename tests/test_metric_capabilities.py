"""Resolver + per-metric ``per_date_series`` contract checks."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from factrix.metrics import fm_beta, hit_rate, ic, monotonicity
from factrix.metrics._metric_capabilities import (
    resolve_min_assets_per_group,
    resolve_per_date_series,
)


def _dates(n: int) -> list[dt.date]:
    return [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def test_resolve_per_date_series_ic_shape() -> None:
    ic_df = pl.DataFrame({"date": _dates(5), "ic": [0.1, 0.2, None, 0.3, 0.0]})
    out = resolve_per_date_series(ic)(ic_df)
    assert out.columns == ["date", "value"]
    assert out.height == 4


def test_resolve_per_date_series_fama_macbeth_shape() -> None:
    beta_df = pl.DataFrame({"date": _dates(3), "beta": [0.01, -0.02, 0.0]})
    out = resolve_per_date_series(fm_beta)(beta_df)
    assert out.columns == ["date", "value"]
    assert out.height == 3


def test_resolve_per_date_series_hit_rate_binary_cast() -> None:
    series = pl.DataFrame({"date": _dates(4), "value": [0.5, -0.1, 0.0, 1.2]})
    out = resolve_per_date_series(hit_rate)(series)
    assert out.columns == ["date", "value"]
    assert out["value"].to_list() == [1.0, 0.0, 0.0, 1.0]


def test_resolve_per_date_series_raises_for_ineligible_metric() -> None:
    def fake_metric(df: pl.DataFrame) -> None:
        return None

    with pytest.raises(TypeError, match="slice-test-eligible"):
        resolve_per_date_series(fake_metric)


def test_resolve_min_assets_per_group_passthrough() -> None:
    assert resolve_min_assets_per_group(ic) is None
    assert resolve_min_assets_per_group(fm_beta) is None
    assert resolve_min_assets_per_group(hit_rate) is None
    assert resolve_min_assets_per_group(monotonicity) == 50


def test_resolve_min_assets_per_group_missing_attr_returns_none() -> None:
    def fake_metric(df: pl.DataFrame) -> None:
        return None

    assert resolve_min_assets_per_group(fake_metric) is None
