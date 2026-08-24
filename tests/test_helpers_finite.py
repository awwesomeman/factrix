"""Shared helpers: NaN is a missing value, never a rankable one."""

from __future__ import annotations

import polars as pl
from factrix.metrics._helpers import (
    _aggregate_to_per_date,
    _assign_quantile_groups,
    _assign_quantile_groups_batch,
    _compute_tie_ratio,
)


def _panel():
    return pl.DataFrame(
        {
            "date": [1] * 6,
            "asset_id": list("abcdef"),
            "factor": [1.0, 2.0, 3.0, float("nan"), None, 6.0],
            "forward_return": [0.1, 0.2, float("nan"), 0.4, 0.5, 0.6],
        }
    )


def test_nan_factor_gets_null_group_not_top_bucket():
    out = _assign_quantile_groups(_panel(), n_groups=2)
    groups = dict(zip(out["asset_id"], out["_group"], strict=True))
    assert groups["d"] is None and groups["e"] is None
    assert groups["f"] == 1 and groups["a"] == 0


def test_batch_matches_single():
    single = _assign_quantile_groups(_panel(), n_groups=2)["_group"].to_list()
    batch = _assign_quantile_groups_batch(_panel(), ["factor"], 2)[
        "_group__factor"
    ].to_list()
    assert single == batch


def test_tie_ratio_ignores_nulls():
    df = pl.DataFrame({"date": [1] * 4, "factor": [1.0, 2.0, None, None]})
    assert _compute_tie_ratio(df) == 0.0
    df2 = pl.DataFrame({"date": [1] * 4, "factor": [1.0, 1.0, None, None]})
    assert _compute_tie_ratio(df2) == 0.5


def test_aggregate_to_per_date_skips_nan():
    df = pl.DataFrame(
        {
            "date": [1, 1, 2, 2],
            "factor": [1.0, 3.0, 1.0, 1.0],
            "forward_return": [0.1, float("nan"), float("nan"), float("nan")],
        }
    )
    out = _aggregate_to_per_date(df)
    assert out.height == 1
    assert out["_f"][0] == 2.0 and out["_r"][0] == 0.1
