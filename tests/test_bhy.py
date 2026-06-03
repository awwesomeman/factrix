"""``fx.multi_factor.bhy`` on the EvaluationResult contract."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import BhyResult, bhy
from factrix._results import MetricResult

from .conftest import make_result, make_spec


def test_returns_dict_keyed_by_primary_name_even_for_single_primary():
    ic = make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.001, primary=ic) for i in range(5)]
    out = bhy(results, primary=[ic], q=0.05)
    assert isinstance(out, dict)
    assert set(out) == {"ic"}
    assert isinstance(out["ic"], BhyResult)


def test_multi_primary_runs_independent_screens():
    ic = make_spec("ic")
    ir = make_spec("ic_ir")
    results = [
        make_result(
            factor=f"f{i}",
            p=0.0001,
            primary=ic,
            extra_outputs={
                "ic_ir": MetricResult(
                    value=0.4,
                    p=0.5,
                    n_obs=100,
                    name=ir.name,
                    metadata={"p_value": 0.5},
                )
            },
            extra_primaries=(ir,),
        )
        for i in range(4)
    ]
    out = bhy(results, primary=[ic, ir], q=0.05)
    assert set(out) == {"ic", "ic_ir"}
    assert len(out["ic"]) == 4
    assert len(out["ic_ir"]) == 0


def test_empty_input_raises():
    ic = make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[EvaluationResult\\]"):
        bhy([], primary=[ic], q=0.05)


def test_no_surviving_results_returns_empty_record():
    ic = make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.9, primary=ic) for i in range(5)]
    out = bhy(results, primary=[ic], q=0.05)
    assert len(out["ic"]) == 0


def test_expand_over_forward_periods_partitions_by_horizon():
    ic = make_spec("ic")
    results = [
        make_result(factor=f"f{i}", p=0.001, primary=ic, forward_periods=1)
        for i in range(3)
    ] + [
        make_result(factor=f"f{i}", p=0.9, primary=ic, forward_periods=5)
        for i in range(3)
    ]
    out = bhy(results, primary=[ic], expand_over=("forward_periods",), q=0.05)
    assert out["ic"].expand_over == ("forward_periods",)
    assert set(out["ic"].n_tests) == {(1,), (5,)}
    survivor_factors = {r.factor for r in out["ic"].survivors}
    assert survivor_factors == {"f0", "f1", "f2"}


def test_expand_over_context_key():
    ic = make_spec("ic")
    results = [
        make_result(factor=f"f{i}", p=0.001, primary=ic, context={"region": "US"})
        for i in range(3)
    ] + [
        make_result(factor=f"f{i}", p=0.9, primary=ic, context={"region": "EU"})
        for i in range(3)
    ]
    out = bhy(results, primary=[ic], expand_over=("region",), q=0.05)
    assert set(out["ic"].n_tests) == {("US",), ("EU",)}


def test_mixed_horizons_without_expand_over_warns():
    ic = make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, primary=ic, forward_periods=1),
        make_result(factor="f2", p=0.01, primary=ic, forward_periods=5),
    ]
    with pytest.warns(RuntimeWarning, match="mixes forward_periods"):
        bhy(results, primary=[ic], q=0.5)


def test_singleton_buckets_warn():
    ic = make_spec("ic")
    results = [
        make_result(factor="f1", p=0.001, primary=ic, context={"region": "US"}),
        make_result(factor="f2", p=0.001, primary=ic, context={"region": "EU"}),
    ]
    with pytest.warns(RuntimeWarning, match="single result"):
        bhy(results, primary=[ic], expand_over=("region",), q=0.5)


def test_primary_must_be_list():
    ic = make_spec("ic")
    with pytest.raises(UserInputError, match="always a list"):
        bhy([make_result(factor="f", p=0.01, primary=ic)], primary=ic)  # type: ignore[arg-type]


def test_primary_must_be_non_empty():
    with pytest.raises(UserInputError, match="non-empty"):
        bhy([], primary=[])


def test_primary_element_must_be_metricspec():
    with pytest.raises(UserInputError, match="MetricSpec instance"):
        bhy([], primary=["ic"])  # type: ignore[list-item]


def test_duplicate_factor_without_expand_over_raises():
    ic = make_spec("ic")
    results = [
        make_result(factor="f1", p=0.01, primary=ic),
        make_result(factor="f1", p=0.02, primary=ic),
    ]
    with pytest.raises(UserInputError, match="unique"):
        bhy(results, primary=[ic])


def test_missing_primary_metric_raises():
    ic = make_spec("ic")
    other = make_spec("alpha")
    results = [make_result(factor="f1", p=0.01, primary=ic)]
    with pytest.raises(UserInputError, match="alpha"):
        bhy(results, primary=[other])


def test_nan_p_raises():
    ic = make_spec("ic")
    results = [
        make_result(factor="f1", p=float("nan"), primary=ic),
    ]
    with pytest.raises(UserInputError, match="NaN"):
        bhy(results, primary=[ic])


def test_factor_as_expand_over_key_raises():
    ic = make_spec("ic")
    results = [make_result(factor="f1", p=0.01, primary=ic)]
    with pytest.raises(UserInputError, match="hypothesis identifier"):
        bhy(results, primary=[ic], expand_over=("factor",))


def test_missing_context_key_raises():
    ic = make_spec("ic")
    results = [make_result(factor="f1", p=0.01, primary=ic, context={"region": "US"})]
    with pytest.raises(UserInputError, match="universe"):
        bhy(results, primary=[ic], expand_over=("universe",))


def test_adj_p_monotonic_within_bucket():
    ic = make_spec("ic")
    p_values = [0.001, 0.01, 0.02, 0.5, 0.9]
    results = [
        make_result(factor=f"f{i}", p=p, primary=ic) for i, p in enumerate(p_values)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = bhy(results, primary=[ic], q=1.0)
    assert len(out["ic"]) == len(p_values)
    by_factor = dict(
        zip((r.factor for r in out["ic"].survivors), out["ic"].adj_p, strict=True)
    )
    assert by_factor["f0"] <= by_factor["f1"] <= by_factor["f2"]
    assert np.all(out["ic"].adj_p <= 1.0)


def test_bhy_result_repr_and_html():
    ic = make_spec("ic")
    results = [make_result(factor=f"f{i}", p=0.001, primary=ic) for i in range(3)]
    out = bhy(results, primary=[ic], q=0.5)["ic"]
    text = repr(out)
    assert "BhyResult" in text
    assert "f0" in text
    html = out._repr_html_()
    assert "<table" in html and "adj_p" in html
