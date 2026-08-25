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


def _nan_vs_null_panels(n_periods: int = 30, n_assets: int = 20):
    """Two identical panels; one marks half the factor cells NaN, the other null."""
    import random

    rng = random.Random(0)
    dates, assets, factor, ret = [], [], [], []
    for d in range(n_periods):
        for a in range(n_assets):
            dates.append(d)
            assets.append(f"a{a}")
            factor.append(None if a % 2 == 0 else rng.gauss(0.0, 1.0))
            ret.append(rng.gauss(0.0, 0.02))
    null_panel = pl.DataFrame(
        {"date": dates, "asset_id": assets, "factor": factor, "forward_return": ret}
    )
    nan_panel = null_panel.with_columns(pl.col("factor").fill_null(float("nan")))
    return nan_panel, null_panel


def test_compute_ic_treats_nan_factor_cells_as_missing():
    from factrix.metrics._primitives._ic import compute_ic

    nan_panel, null_panel = _nan_vs_null_panels()
    nan_ic = compute_ic(nan_panel)["factor"].drop("_drop_stats")
    null_ic = compute_ic(null_panel)["factor"].drop("_drop_stats")
    assert nan_ic.equals(null_ic)
    assert nan_ic["n_assets"].to_list() == [10] * nan_ic.height


def test_coverage_counters_treat_nan_factor_cells_as_missing():
    import factrix as fx

    nan_panel, null_panel = _nan_vs_null_panels()
    nan_props = fx.inspect_data(nan_panel).properties
    null_props = fx.inspect_data(null_panel).properties
    assert nan_props.n_pairs == null_props.n_pairs
    assert nan_props.n_pairs == 300

    nan_res = fx.evaluate(nan_panel, factor_cols=["factor"], metrics={"ic": fx.metrics.ic()}, forward_periods=1)
    null_res = fx.evaluate(null_panel, factor_cols=["factor"], metrics={"ic": fx.metrics.ic()}, forward_periods=1)
    assert nan_res["factor"].metrics["ic"].value == null_res["factor"].metrics["ic"].value
    assert nan_res["factor"].metrics["ic"].n_obs == null_res["factor"].metrics["ic"].n_obs
