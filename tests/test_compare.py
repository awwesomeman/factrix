"""``fx.compare`` multi-metric wide leaderboard."""

from __future__ import annotations

import pytest
from factrix._compare import compare
from factrix._errors import UserInputError
from factrix._results import MetricResult

from .conftest import make_result, make_spec


def _with_extra(factor: str, ic_value: float, sharpe_value: float):
    return make_result(
        factor=factor,
        p=0.1,
        metric="ic",
        value=ic_value,
        extra_outputs={
            "sharpe": MetricResult(
                value=sharpe_value,
                p_value=0.2,
                alternative="two-sided",
                n_obs=100,
                name="sharpe",
                metadata={"p_value": 0.2},
            )
        },
    )


def test_multi_metric_wide_layout():
    make_spec("ic")
    make_spec("sharpe")
    results = [
        _with_extra("a", ic_value=0.05, sharpe_value=1.2),
        _with_extra("b", ic_value=0.02, sharpe_value=0.8),
    ]
    df = compare(results, metrics=["ic", "sharpe"])
    assert df.columns == [
        "factor",
        "forward_periods",
        "ic",
        "ic_p_value",
        "sharpe",
        "sharpe_p_value",
    ]
    assert df["factor"].to_list() == ["a", "b"]
    assert df["ic"].to_list() == [pytest.approx(0.05), pytest.approx(0.02)]
    assert df["sharpe"].to_list() == [pytest.approx(1.2), pytest.approx(0.8)]


def test_p_value_column_uses_requested_metric_label():
    make_spec("custom_ic")
    results = [make_result(factor="a", p=0.03, metric="custom_ic", value=0.07)]

    df = compare(results, metrics=["custom_ic"], sort_by="custom_ic_p_value")

    assert df.columns == [
        "factor",
        "forward_periods",
        "custom_ic",
        "custom_ic_p_value",
        "rank",
    ]
    assert df["custom_ic_p_value"].to_list() == [pytest.approx(0.03)]


def test_sort_by_descending_true_adds_rank():
    make_spec("ic")
    results = [
        make_result(factor="lo", p=0.5, metric="ic", value=0.01),
        make_result(factor="hi", p=0.01, metric="ic", value=0.10),
        make_result(factor="mid", p=0.1, metric="ic", value=0.05),
    ]
    df = compare(results, metrics=["ic"], sort_by="ic", descending=True)
    assert df["factor"].to_list() == ["hi", "mid", "lo"]
    assert df["rank"].to_list() == [1, 2, 3]


def test_sort_by_descending_false_ranks_low_first():
    make_spec("rank_turnover")
    results = [
        make_result(factor="high_to", p=0.5, metric="rank_turnover", value=0.80),
        make_result(factor="low_to", p=0.5, metric="rank_turnover", value=0.10),
        make_result(factor="mid_to", p=0.5, metric="rank_turnover", value=0.45),
    ]
    df = compare(
        results, metrics=["rank_turnover"], sort_by="rank_turnover", descending=False
    )
    assert df["factor"].to_list() == ["low_to", "mid_to", "high_to"]
    assert df["rank"].to_list() == [1, 2, 3]


def test_sort_by_p_value_column():
    make_spec("ic")
    results = [
        make_result(factor="weak", p=0.5, metric="ic", value=0.10),
        make_result(factor="strong", p=0.01, metric="ic", value=0.01),
        make_result(factor="mid", p=0.1, metric="ic", value=0.05),
    ]
    df = compare(results, metrics=["ic"], sort_by="ic_p_value", descending=False)
    assert df["factor"].to_list() == ["strong", "mid", "weak"]
    assert df["rank"].to_list() == [1, 2, 3]


def test_sort_by_identity_and_param_columns():
    make_spec("ic")
    results = [
        make_result(factor="b", p=0.1, metric="ic", params={"region": "US"}),
        make_result(factor="a", p=0.1, metric="ic", params={"region": "EU"}),
    ]

    by_factor = compare(results, metrics=["ic"], sort_by="factor", descending=False)
    assert by_factor["factor"].to_list() == ["a", "b"]

    by_params = compare(results, metrics=["ic"], sort_by="region", descending=False)
    assert by_params["region"].to_list() == ["EU", "US"]


def test_no_sort_keeps_input_order_omits_rank():
    make_spec("ic")
    results = [
        make_result(factor="c", p=0.1, metric="ic", value=0.01),
        make_result(factor="a", p=0.1, metric="ic", value=0.10),
        make_result(factor="b", p=0.1, metric="ic", value=0.05),
    ]
    df = compare(results, metrics=["ic"])
    assert df["factor"].to_list() == ["c", "a", "b"]
    assert "rank" not in df.columns


def test_param_keys_propagate_with_null_fill():
    make_spec("ic")
    results = [
        make_result(factor="a", p=0.1, metric="ic", params={"region": "US"}),
        make_result(factor="b", p=0.1, metric="ic", params={"sector": "tech"}),
    ]
    df = compare(results, metrics=["ic"])
    assert set(df.columns) >= {"region", "sector"}
    a_row = df.filter(df["factor"] == "a").to_dicts()[0]
    b_row = df.filter(df["factor"] == "b").to_dicts()[0]
    assert a_row["region"] == "US" and a_row["sector"] is None
    assert b_row["sector"] == "tech" and b_row["region"] is None


def test_p_column_populated_when_metadata_present():
    make_spec("ic")
    results = [make_result(factor="f1", p=0.042, metric="ic", value=0.05)]
    df = compare(results, metrics=["ic"])
    assert df["ic_p_value"].to_list() == [pytest.approx(0.042)]


def test_empty_results_raises():
    make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[EvaluationResult\\]"):
        compare([], metrics=["ic"])


def test_metrics_must_be_list():
    ic = make_spec("ic")
    with pytest.raises(UserInputError, match="always a list"):
        compare(
            [make_result(factor="f", p=0.01, metric="ic")],
            metrics=ic,  # type: ignore[arg-type]
        )


def test_metrics_must_be_non_empty():
    make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[str\\]"):
        compare([make_result(factor="f", p=0.01, metric="ic")], metrics=[])


def test_metrics_element_must_be_str():
    with pytest.raises(UserInputError, match="str metric label"):
        compare([make_result(factor="f", p=0.01, metric="ic")], metrics=[123])  # type: ignore[list-item]


def test_missing_metric_raises():
    make_spec("ic")
    make_spec("alpha")
    results = [make_result(factor="f", p=0.01, metric="ic")]
    with pytest.raises(UserInputError, match="other"):
        compare(results, metrics=["other"])


def test_sort_by_unknown_output_column_raises():
    make_spec("ic")
    make_spec("sharpe")
    results = [make_result(factor="f", p=0.01, metric="ic")]
    with pytest.raises(UserInputError, match="unknown sort_by") as excinfo:
        compare(results, metrics=["ic"], sort_by="sharpe")
    assert "ic_p_value" in str(excinfo.value)
