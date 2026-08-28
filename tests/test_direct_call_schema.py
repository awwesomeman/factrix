"""Direct raw-panel metric calls validate the key columns up front (#882).

``evaluate`` gates the baseline schema once and projects a view per factor;
a standalone ``metric(data, ...)`` call skipped that gate, so a mis-named
``asset_id`` surfaced as a polars ``ColumnNotFoundError`` from inside a
quantile join — after ``_group`` had been attached, naming an internal
column the caller never wrote. The check now lives in ``MetricBase.__call__``
for every metric that consumes the raw panel (``input_shape=PANEL`` with no
``requires``), so each direct-call path fails the same way ``evaluate`` does,
with the same error type and docs pointer, and no metric carries its own copy.
Consumers of a producer's derived frame (``ic`` on ``compute_ic`` output,
``common_beta`` on per-asset betas) are not gated: that schema is the
producer's contract, not the panel's.
"""

from __future__ import annotations

import factrix as fx
import polars as pl
import pytest
from factrix._errors import UserInputError
from factrix.metrics import (
    breakeven_cost,
    compute_ic,
    notional_turnover,
    positive_rate,
    quantile_spread,
    rank_turnover,
)
from factrix.preprocess import compute_forward_return


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=8, n_dates=60, seed=0)
    return compute_forward_return(raw, forward_periods=5)


_PANEL_METRICS = [
    ("notional_turnover", lambda p: notional_turnover(p, n_groups=2, rebalance_lag=1)),
    ("rank_turnover", lambda p: rank_turnover(p)),
    ("quantile_spread", lambda p: quantile_spread(p, n_groups=2)),
    ("compute_ic", lambda p: compute_ic(p)),
]


@pytest.mark.parametrize(
    ("label", "run"), _PANEL_METRICS, ids=[m[0] for m in _PANEL_METRICS]
)
def test_missing_asset_id_is_a_user_input_error_before_any_polars_work(
    panel, label, run
):
    renamed = panel.rename({"asset_id": "asset"})
    with pytest.raises(UserInputError, match=r"asset_id") as info:
        run(renamed)
    message = str(info.value)
    assert label in message  # named after the metric the caller invoked
    assert "_group" not in message  # no internal column leaks into the message
    assert "api/data-schema" in message


def test_missing_date_is_reported_the_same_way(panel):
    with pytest.raises(UserInputError, match=r"'date'"):
        notional_turnover(panel.drop("date"), n_groups=2)


def test_derived_frame_consumers_are_not_gated(panel):
    series = compute_ic(panel)["factor"]
    assert "asset_id" not in series.columns  # a producer's frame has no asset axis
    assert positive_rate(series, value_col="ic").n_obs > 0


def test_scalar_helpers_are_untouched():
    assert breakeven_cost(0.001, turnover=0.2, holding_periods=5).value > 0


def test_correct_schema_is_unchanged(panel):
    assert (
        notional_turnover(panel, n_groups=2, rebalance_lag=1).metadata["n_groups"] == 2
    )
    assert quantile_spread(panel, n_groups=2)["factor"].n_obs > 0


def test_evaluate_and_direct_call_agree_on_the_error_contract(panel):
    renamed = panel.rename({"asset_id": "asset"})
    with pytest.raises(UserInputError, match=r"asset_id") as via_evaluate:
        fx.evaluate(
            renamed,
            metrics={"nt": notional_turnover(n_groups=2)},
            factor_cols=["factor"],
        )
    with pytest.raises(UserInputError, match=r"asset_id") as direct:
        notional_turnover(renamed, n_groups=2)
    assert via_evaluate.value.field == direct.value.field == "data"
