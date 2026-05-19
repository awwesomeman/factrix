"""``fx.compare`` on the EvaluationResult contract."""

from __future__ import annotations

import pytest
from factrix._compare import compare
from factrix._errors import UserInputError

from .conftest import make_result, make_spec


def test_descending_true_ranks_high_value_first():
    ic = make_spec("ic")
    results = [
        make_result(factor="lo", p=0.5, primary=ic, value=0.01),
        make_result(factor="hi", p=0.01, primary=ic, value=0.10),
        make_result(factor="mid", p=0.1, primary=ic, value=0.05),
    ]
    df = compare(results, metric=ic, descending=True)
    assert df["factor"].to_list() == ["hi", "mid", "lo"]
    assert df["rank"].to_list() == [1, 2, 3]


def test_descending_false_ranks_low_value_first():
    turnover = make_spec("turnover")
    results = [
        make_result(factor="high_to", p=0.5, primary=turnover, value=0.80),
        make_result(factor="low_to", p=0.5, primary=turnover, value=0.10),
        make_result(factor="mid_to", p=0.5, primary=turnover, value=0.45),
    ]
    df = compare(results, metric=turnover, descending=False)
    assert df["factor"].to_list() == ["low_to", "mid_to", "high_to"]
    assert df["rank"].to_list() == [1, 2, 3]


def test_columns_include_factor_forward_periods_value_p_rank():
    ic = make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.1, primary=ic) for i in range(2)]
    df = compare(results, metric=ic)
    for col in ("factor", "forward_periods", "value", "p", "rank"):
        assert col in df.columns


def test_context_keys_propagate_with_null_fill():
    ic = make_spec("ic")
    results = [
        make_result(factor="a", p=0.1, primary=ic, context={"region": "US"}),
        make_result(factor="b", p=0.1, primary=ic, context={"sector": "tech"}),
    ]
    df = compare(results, metric=ic)
    assert set(df.columns) >= {"region", "sector"}
    a_row = df.filter(df["factor"] == "a").to_dicts()[0]
    b_row = df.filter(df["factor"] == "b").to_dicts()[0]
    assert a_row["region"] == "US" and a_row["sector"] is None
    assert b_row["sector"] == "tech" and b_row["region"] is None


def test_empty_input_raises():
    ic = make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty"):
        compare([], metric=ic)


def test_metric_must_be_metricspec():
    with pytest.raises(UserInputError, match="MetricSpec"):
        compare(
            [make_result(factor="f", p=0.01, primary=make_spec("ic"))],
            metric="ic",  # type: ignore[arg-type]
        )


def test_missing_metric_raises():
    ic = make_spec("ic")
    other = make_spec("alpha")
    results = [make_result(factor="f", p=0.01, primary=ic)]
    with pytest.raises(UserInputError, match="alpha"):
        compare(results, metric=other)


def test_p_column_populated_when_metadata_present():
    ic = make_spec("ic")
    results = [make_result(factor="f1", p=0.042, primary=ic, value=0.05)]
    df = compare(results, metric=ic)
    assert df["p"].to_list() == [pytest.approx(0.042)]
