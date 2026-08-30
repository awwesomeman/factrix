"""Direct raw-panel metric calls validate their columns up front.

``evaluate`` gates the baseline schema once and projects a view per factor;
a standalone ``metric(data, ...)`` call skipped that gate, so a mis-named
``asset_id`` surfaced as a polars ``ColumnNotFoundError`` from inside a
quantile join — after ``_group`` had been attached, naming an internal
column the caller never wrote. The check now lives in ``MetricBase.__call__``
for every metric that consumes the raw panel (``input_shape=PANEL`` with no
``requires``), so each direct-call path fails the same way ``evaluate`` does,
with the same error type and docs pointer, and no metric carries its own copy.

The gate covers two kinds of column. The panel's *key* columns (``date`` /
``asset_id``) are a fixed contract. On top of them, every column this call
*named* — any ``*_col`` / ``*_cols`` parameter set away from its documented
default — must exist on the frame: a mis-typed ``factor_col`` is the same
mistake as a mis-typed ``asset_id`` and must fail the same way, rather than
reaching a polars expression or being answered with a NaN "insufficient
data" envelope that reads as a property of the data. A parameter left at its
default names a column the metric documents (``factor``, ``forward_return``,
``price``, ``market_cap``); its absence is a fact about the data, and the
metric bodies keep reporting it as a short-circuit verdict.

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
    raw = fx.datasets.make_cs_panel(n_assets=8, n_dates=60, rng=0)
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


def _panel_metrics_with_named_columns():
    """Every public raw-panel metric paired with each column it lets you name."""
    import factrix.metrics as metrics_module
    from factrix._axis import InputShape

    cases = []
    for name in sorted(dir(metrics_module)):
        cls = getattr(metrics_module, name)
        if not isinstance(cls, type):
            continue
        if getattr(cls, "input_shape", None) is not InputShape.PANEL:
            continue
        if getattr(cls, "requires", None):
            continue
        for param in getattr(cls, "_param_names", ()):
            if param.endswith(("_col", "_cols")):
                cases.append((name, cls, param))
    return cases


_NAMED_COLUMN_CASES = _panel_metrics_with_named_columns()


def test_every_raw_panel_metric_exposes_a_named_column():
    """Guard the sweep below against silently enumerating nothing."""
    assert len({case[0] for case in _NAMED_COLUMN_CASES}) >= 15


@pytest.mark.parametrize(
    ("label", "cls", "param"),
    _NAMED_COLUMN_CASES,
    ids=[f"{label}-{param}" for label, _, param in _NAMED_COLUMN_CASES],
)
def test_a_mis_named_column_is_a_user_input_error(panel, label, cls, param):
    typo = ["nope"] if param.endswith("_cols") else "nope"
    with pytest.raises(UserInputError) as info:
        cls(panel, **{param: typo})
    err = info.value
    assert err.func_name == label
    assert err.field == param
    assert err.value == "nope"
    assert "factor" in err.candidates  # the frame's own columns are the candidates
    assert "_overlap_periods" not in err.candidates  # reserved stamps are not
    assert err.docs_url.rstrip("/").split("/factrix/")[-1].startswith("api/metrics")


def test_the_suggestion_points_at_the_column_the_caller_meant(panel):
    from factrix.metrics import k_spread

    with pytest.raises(UserInputError) as info:
        k_spread(panel, k=1, return_col="forward_returns")
    assert info.value.suggestions == ("forward_return",)


def test_a_column_left_at_its_default_stays_a_data_verdict(panel):
    """An absent *documented* column is a fact about the data, not a typo.

    ``market_cap`` is optional schema: a panel without it is an ordinary
    configuration ``evaluate`` reports as a short-circuit, so the gate must
    not convert it into a caller error.
    """
    from factrix.metrics import quantile_spread_vw

    assert "market_cap" not in panel.columns
    out = quantile_spread_vw(panel, overlap_periods=5, n_groups=2)
    assert out.metadata["reason"] == "no_weight_column"


def test_evaluate_and_direct_call_agree_on_a_mis_named_factor_column(panel):
    with pytest.raises(UserInputError) as via_evaluate:
        fx.evaluate(
            panel,
            metrics={"qs": quantile_spread(n_groups=2)},
            factor_cols=["nope"],
        )
    with pytest.raises(UserInputError) as direct:
        quantile_spread(panel, n_groups=2, factor_cols=["nope"])
    assert via_evaluate.value.func_name == "evaluate"
    assert direct.value.func_name == "quantile_spread"
    assert "nope" in str(via_evaluate.value)
    assert "nope" in str(direct.value)
