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

import pathlib

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


# --------------------------------------------------------------------------
# Knob parity: the same mistake raises the same exception whether the metric
# was reached through ``evaluate`` or called directly.
#
# ``evaluate`` type/range-checks ``overlap_periods`` at the data boundary and
# every metric body checks its ``inference=`` against its own allowlist. A
# direct call used to reach neither: ``ic(ic_df, overlap_periods=-1)`` raised
# an ``OverflowError`` from a stride computation, ``overlap_periods=0`` a
# polars ``InvalidOperationError``, and a stray ``inference=`` string an
# ``AttributeError`` from the sample-floor pre-flight. Each now fails at the
# boundary, with the library's own exception, from either entry point.
# --------------------------------------------------------------------------

_BAD_OVERLAPS = [-1, 0, 2.5, True]


def _overlap_metrics():
    from factrix.metrics import ic, k_spread, positive_rate, top_concentration

    return [
        ("k_spread", lambda data, **kw: k_spread(data, k=1, **kw), None),
        ("top_concentration", top_concentration, None),
        ("positive_rate", positive_rate, None),
        ("ic", ic, "ic_df"),
    ]


@pytest.mark.parametrize("bad", _BAD_OVERLAPS, ids=[repr(b) for b in _BAD_OVERLAPS])
@pytest.mark.parametrize(
    ("label", "cls", "consumes"),
    _overlap_metrics(),
    ids=[case[0] for case in _overlap_metrics()],
)
def test_direct_call_rejects_a_nonsensical_overlap_periods(
    panel, label, cls, consumes, bad
):
    data = compute_ic(panel)["factor"] if consumes else panel
    with pytest.raises(UserInputError) as info:
        cls(data, overlap_periods=bad)
    assert info.value.func_name == label
    assert info.value.field == "overlap_periods"
    assert info.value.value == bad


@pytest.mark.parametrize("bad", _BAD_OVERLAPS, ids=[repr(b) for b in _BAD_OVERLAPS])
def test_evaluate_rejects_the_same_overlap_periods_the_same_way(panel, bad):
    from factrix.metrics import ic

    with pytest.raises(UserInputError) as info:
        fx.evaluate(
            panel,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            overlap_periods=bad,
        )
    assert info.value.func_name == "evaluate"
    assert info.value.field == "overlap_periods"


def test_a_valid_overlap_periods_still_reaches_the_metric(panel):
    from factrix.metrics import ic

    assert ic(compute_ic(panel)["factor"], overlap_periods=5).name == ""


def test_ic_rejects_an_unvetted_inference_from_either_entry_point(panel):
    """The sample-floor pre-flight dereferences ``inference`` before the body.

    ``evaluate`` resolves the dynamic floor first, so an unvetted value used
    to surface there as an ``AttributeError`` while the direct call reached
    the body and raised ``IncompatibleInferenceError``. Both now raise at the
    same chokepoint.
    """
    from factrix._errors import IncompatibleInferenceError
    from factrix.metrics import ic

    with pytest.raises(IncompatibleInferenceError) as direct:
        ic(compute_ic(panel)["factor"], inference="newey_west")
    with pytest.raises(IncompatibleInferenceError) as via_evaluate:
        fx.evaluate(
            panel, metrics={"ic": ic(inference="newey_west")}, factor_cols=["factor"]
        )
    assert direct.value.func_name == via_evaluate.value.func_name == "ic"
    assert str(direct.value) == str(via_evaluate.value)


def _bucketed_metrics():
    from factrix.metrics import (
        common_quantile_spread,
        compute_group_returns,
        compute_spread_series,
        monotonicity,
        notional_turnover,
        quantile_spread,
        quantile_spread_vw,
    )

    return [
        ("quantile_spread", quantile_spread),
        ("quantile_spread_vw", quantile_spread_vw),
        ("monotonicity", monotonicity),
        ("common_quantile_spread", common_quantile_spread),
        ("notional_turnover", notional_turnover),
        ("compute_spread_series", compute_spread_series),
        ("compute_group_returns", compute_group_returns),
    ]


@pytest.mark.parametrize("bad", [1, 0, -3], ids=["one", "zero", "negative"])
@pytest.mark.parametrize(
    ("label", "cls"), _bucketed_metrics(), ids=[case[0] for case in _bucketed_metrics()]
)
def test_a_split_below_the_bucketing_floor_is_a_user_input_error(
    panel, label, cls, bad
):
    """``n_groups`` below the floor fails before any data work, everywhere.

    It used to raise a bare ``ValueError`` from the group-assignment kernel,
    i.e. only once a metric had already sampled and scanned the panel.
    """
    with pytest.raises(UserInputError) as info:
        cls(panel, n_groups=bad)
    assert info.value.func_name == label
    assert info.value.field == "n_groups"
    assert info.value.value == bad


def test_evaluate_reports_a_bad_split_identically(panel):
    with pytest.raises(UserInputError) as info:
        fx.evaluate(
            panel, metrics={"qs": quantile_spread(n_groups=1)}, factor_cols=["factor"]
        )
    assert info.value.func_name == "quantile_spread"
    assert info.value.field == "n_groups"


# --------------------------------------------------------------------------
# The same contract outside the metrics: the generators and the slice tests
# are documented entry points, so their argument guards are user-input
# failures too, not bare builtins.
# --------------------------------------------------------------------------

_DATASET_CASES = [
    ("make_cs_panel", {"n_assets": 1, "n_dates": 40}, "n_assets"),
    (
        "make_cs_panel",
        {"n_assets": 8, "n_dates": 40, "factor_persistence": 1.0},
        "factor_persistence",
    ),
    ("make_cs_panel", {"n_assets": 8, "n_dates": 3}, "n_dates"),
    ("make_event_panel", {"n_assets": 0, "n_dates": 40}, "n_assets"),
    ("make_event_panel", {"n_assets": 8, "n_dates": 3}, "n_dates"),
    (
        "make_event_panel",
        {"n_assets": 8, "n_dates": 40, "event_rate": 1.5},
        "event_rate",
    ),
    (
        "make_event_panel",
        {"n_assets": 8, "n_dates": 40, "event_magnitude_jitter": -1.0},
        "event_magnitude_jitter",
    ),
    (
        "make_multi_factor_panel",
        {"n_factors": 0, "n_assets": 8, "n_dates": 40},
        "n_factors",
    ),
    (
        "make_multi_factor_panel",
        {"n_factors": 2, "n_assets": 1, "n_dates": 40},
        "n_assets",
    ),
    (
        "make_multi_factor_panel",
        {"n_factors": 2, "n_assets": 8, "n_dates": 3},
        "n_dates",
    ),
    (
        "make_multi_factor_panel",
        {"n_factors": 2, "n_assets": 8, "n_dates": 40, "factor_correlation": 1.0},
        "factor_correlation",
    ),
    (
        "make_multi_factor_event_panel",
        {"n_factors": 0, "n_assets": 8, "n_dates": 40},
        "n_factors",
    ),
    (
        "make_multi_factor_event_panel",
        {"n_factors": 2, "n_assets": 0, "n_dates": 40},
        "n_assets",
    ),
    (
        "make_multi_factor_event_panel",
        {"n_factors": 2, "n_assets": 8, "n_dates": 3},
        "n_dates",
    ),
    (
        "make_multi_factor_event_panel",
        {"n_factors": 2, "n_assets": 8, "n_dates": 40, "event_rate": -0.1},
        "event_rate",
    ),
    (
        "make_multi_factor_event_panel",
        {"n_factors": 2, "n_assets": 8, "n_dates": 40, "event_magnitude_jitter": -0.5},
        "event_magnitude_jitter",
    ),
]


@pytest.mark.parametrize(
    ("builder", "kwargs", "field"),
    _DATASET_CASES,
    ids=[f"{builder}-{field}" for builder, _, field in _DATASET_CASES],
)
def test_a_generator_guard_names_the_argument_it_rejected(builder, kwargs, field):
    with pytest.raises(UserInputError) as info:
        getattr(fx.datasets, builder)(**kwargs)
    assert info.value.func_name == builder
    assert info.value.field == field
    assert "api/datasets" in info.value.docs_url


def test_every_generator_guard_is_covered():
    """Guard the sweep against a builder growing an unswept guard."""
    source = pathlib.Path("factrix/datasets.py").read_text(encoding="utf-8")
    assert source.count("raise _bad_arg(") == len(_DATASET_CASES)


def test_a_date_disjoint_partition_is_a_user_input_error():
    """The cross-sectional pair's refusal keeps its message, gains its type."""
    from factrix.metrics import ic
    from factrix.slicing import slice_joint_test

    raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=80, rng=0)
    panel = compute_forward_return(raw, forward_periods=5)
    dates = panel["date"].unique().sort()
    labelled = panel.with_columns(
        pl.when(pl.col("date") <= dates[len(dates) // 2])
        .then(pl.lit("early"))
        .otherwise(pl.lit("late"))
        .alias("regime")
    )
    with pytest.raises(UserInputError) as info:
        slice_joint_test(
            labelled, ic(), by="regime", factor_col="factor", overlap_periods=5
        )
    assert info.value.field == "by"
    assert "aligned dates" in str(info.value)


@pytest.mark.parametrize(
    "module",
    [
        "factrix/datasets.py",
        "factrix/slicing/inference.py",
        "factrix/slicing/period_inference.py",
        "factrix/slicing/_primitive.py",
    ],
)
def test_the_user_facing_modules_raise_no_bare_value_error(module):
    """Keep ``docs/api/errors.md``'s ``FactrixError`` contract true.

    Every ``raise ValueError`` left in these modules would be a library
    failure ``except fx.FactrixError`` does not catch. ``UserInputError``
    multi-inherits ``ValueError``, so nothing downstream loses a handler.
    """
    source = pathlib.Path(module).read_text(encoding="utf-8")
    assert "raise ValueError" not in source
