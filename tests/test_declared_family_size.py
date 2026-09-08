"""Plan-level family sizes that exceed the submitted screening results (#1072)."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import (
    bhy,
    bhy_across_metrics,
    bhy_hierarchical,
    partial_conjunction,
    partial_conjunction_across_metrics,
)
from factrix._results import MetricResult
from factrix.stats.multiple_testing import bhy_adjusted_p

from .conftest import make_result, make_spec


def _output(name: str, p: float) -> MetricResult:
    return MetricResult(
        value=0.1,
        p_value=p,
        alternative="two-sided",
        n_obs=100,
        n_obs_axis="periods",
        name=name,
        metadata={"p_value": p},
    )


def _cross_result(factor: str, p: float):
    return make_result(
        factor=factor,
        p=p,
        metric="ic",
        extra_outputs={"beta": _output("beta", p)},
    )


def test_larger_family_size_counts_unsubmitted_candidates_separately():
    make_spec("ic")
    out = bhy(
        [make_result(factor="submitted", p=0.01, metric="ic")],
        metrics=["ic"],
        family_size=10,
    )["ic"]

    assert out.family_size == {(): 10}
    assert out.adj_p_all[0] == pytest.approx(
        bhy_adjusted_p(np.array([0.01]), n_tests=10)[0]
    )
    assert out.family.n_candidates_declared == 10
    assert out.family.n_tests_computed == 1
    assert out.family.n_inactive == 0
    assert out.family.n_unsubmitted == 9
    assert out.family.n_tests_adjusted == 10


def test_explicit_observed_family_size_is_the_default_and_smaller_raises():
    make_spec("ic")
    results = [
        make_result(factor="a", p=0.01, metric="ic"),
        make_result(factor="b", p=0.02, metric="ic"),
    ]

    inferred = bhy(results, metrics=["ic"])["ic"]
    explicit = bhy(results, metrics=["ic"], family_size=2)["ic"]
    np.testing.assert_allclose(explicit.adj_p_all, inferred.adj_p_all)
    assert explicit.family.n_unsubmitted == 0

    with pytest.raises(UserInputError, match="at least the 2 submitted"):
        bhy(results, metrics=["ic"], family_size=1)


def test_bucketed_family_size_is_declared_per_bucket():
    make_spec("ic")
    results = [
        make_result(factor="us", p=0.01, metric="ic", params={"universe": "US"}),
        make_result(factor="eu", p=0.01, metric="ic", params={"universe": "EU"}),
    ]

    out = bhy(
        results,
        metrics=["ic"],
        expand_over=("universe",),
        family_size={("US",): 10, ("EU",): 4},
    )["ic"]

    assert out.family_size == {("US",): 10, ("EU",): 4}
    assert out.family.n_candidates_declared == 14
    assert out.family.n_unsubmitted == 12
    assert out.adj_p_all[0] > out.adj_p_all[1]

    with pytest.raises(UserInputError, match="exactly the submitted family keys"):
        bhy(
            results,
            metrics=["ic"],
            expand_over=("universe",),
            family_size={("US",): 10},
        )


def test_unsubmitted_and_submitted_inactive_are_distinct_under_both_policies():
    make_spec("ic")
    live = make_result(factor="live", p=0.01, metric="ic")
    inactive = make_result(
        factor="inactive",
        p=1.0,
        metric="ic",
        metadata={"reason": "insufficient_ic_periods"},
    )

    counted = bhy([live, inactive], metrics=["ic"], family_size=3)["ic"]
    excluded = bhy(
        [live, inactive],
        metrics=["ic"],
        family_size=3,
        inactive_policy="exclude",
    )["ic"]

    assert counted.family_size == {(): 3}
    assert excluded.family_size == {(): 2}
    for out in (counted, excluded):
        assert out.family.n_candidates_declared == 3
        assert out.family.n_tests_computed == 1
        assert out.family.n_inactive == 1
        assert out.family.n_unsubmitted == 1
    assert counted.family.n_tests_adjusted == 3
    assert excluded.family.n_tests_adjusted == 2


def test_all_five_verbs_share_the_family_size_surface():
    make_spec("ic")
    flat = [
        make_result(factor="a", p=0.01, metric="ic"),
        make_result(factor="b", p=0.02, metric="ic"),
    ]
    cross = [_cross_result("a", 0.01), _cross_result("b", 0.02)]
    conditioned = [
        make_result(factor=factor, p=0.01, metric="ic", params={"region": region})
        for factor in ("a", "b")
        for region in ("US", "EU")
    ]
    grouped = [
        make_result(factor=factor, p=0.01, metric="ic", params={"family": family})
        for family, factors in (("momentum", ("a", "b")), ("carry", ("c", "d")))
        for factor in factors
    ]

    outputs = [
        bhy(flat, metrics=["ic"], family_size=3)["ic"],
        bhy_across_metrics(cross, metrics=["ic", "beta"], family_size=6),
        partial_conjunction(
            conditioned,
            metrics=["ic"],
            min_pass=2,
            expand_over=("region",),
            family_size=3,
        )["ic"],
        partial_conjunction_across_metrics(
            cross,
            metrics=["ic", "beta"],
            min_pass=2,
            family_size=3,
        ),
        bhy_hierarchical(
            grouped,
            metrics=["ic"],
            group="family",
            family_size=3,
        )["ic"],
    ]

    assert [out.family.n_unsubmitted for out in outputs] == [1, 2, 2, 2, 2]
    assert [out.family.n_tests_adjusted for out in outputs] == [3, 6, 6, 6, 6]
    assert outputs[0].family_size == {(): 3}
    assert outputs[1].family_size == {(): 6}
    assert set(outputs[2].family_size.values()) == {3}
    assert set(outputs[3].family_size.values()) == {3}
    assert outputs[4].family_size == {("momentum",): 3, ("carry",): 3}

    invalid_calls = [
        lambda: bhy(flat, metrics=["ic"], family_size=1),
        lambda: bhy_across_metrics(cross, metrics=["ic", "beta"], family_size=3),
        lambda: partial_conjunction(
            conditioned,
            metrics=["ic"],
            min_pass=2,
            expand_over=("region",),
            family_size=1,
        ),
        lambda: partial_conjunction_across_metrics(
            cross,
            metrics=["ic", "beta"],
            min_pass=2,
            family_size=1,
        ),
        lambda: bhy_hierarchical(
            grouped,
            metrics=["ic"],
            group="family",
            family_size=1,
        ),
    ]
    for invalid_call in invalid_calls:
        with pytest.raises(UserInputError, match="submitted candidate"):
            invalid_call()


def test_partial_conjunction_missing_conditions_are_inert_non_rejections():
    make_spec("ic")
    conditioned = [
        make_result(factor="a", p=p, metric="ic", params={"region": region})
        for p, region in ((0.01, "US"), (0.02, "EU"))
    ]

    inferred = partial_conjunction(
        conditioned,
        metrics=["ic"],
        min_pass=2,
        expand_over=("region",),
    )["ic"]
    declared = partial_conjunction(
        conditioned,
        metrics=["ic"],
        min_pass=2,
        expand_over=("region",),
        n_conditions=3,
        family_size=3,
    )["ic"]

    assert inferred.pc_p_all[0] == pytest.approx(0.02)
    assert declared.pc_p_all[0] == pytest.approx(0.04)
    assert declared.family.n_unsubmitted == 1
