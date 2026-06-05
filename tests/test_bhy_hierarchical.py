"""``fx.multi_factor.bhy_hierarchical`` on the EvaluationResult contract."""

from __future__ import annotations

import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import HierarchicalBhyResult, bhy_hierarchical

from .conftest import make_result, make_spec


def _grouped(p_by_factor: dict[str, float], group_value: str, primary):
    return [
        make_result(
            factor=factor, p=p, primary=primary, context={"family": group_value}
        )
        for factor, p in p_by_factor.items()
    ]


def test_returns_dict_per_primary():
    make_spec("ic")
    results = _grouped({"mom_1": 0.001, "mom_2": 0.5}, "momentum", "ic") + _grouped(
        {"val_1": 0.001, "val_2": 0.5}, "value", "ic"
    )
    out = bhy_hierarchical(results, primary=["ic"], group="family", q=0.05)
    assert isinstance(out, dict)
    assert set(out) == {"ic"}
    assert isinstance(out["ic"], HierarchicalBhyResult)


def test_n_tests_covers_all_input_groups():
    make_spec("ic")
    results = _grouped({"a": 0.5, "b": 0.5}, "g1", "ic") + _grouped(
        {"c": 0.5, "d": 0.5}, "g2", "ic"
    )
    out = bhy_hierarchical(results, primary=["ic"], group="family", q=0.05)
    assert set(out["ic"].n_tests) == {("g1",), ("g2",)}


def test_single_group_raises():
    make_spec("ic")
    results = _grouped({"a": 0.001, "b": 0.001}, "only", "ic")
    with pytest.raises(UserInputError, match="at least 2"):
        bhy_hierarchical(results, primary=["ic"], group="family")


def test_every_result_own_group_raises():
    make_spec("ic")
    results = [
        make_result(factor=f"f{i}", p=0.01, primary="ic", context={"family": f"g{i}"})
        for i in range(3)
    ]
    with pytest.raises(UserInputError, match="every result is its own group"):
        bhy_hierarchical(results, primary=["ic"], group="family")


def test_singleton_group_warns():
    make_spec("ic")
    results = (
        _grouped({"a": 0.5}, "g1", "ic")
        + _grouped({"b": 0.5}, "g2", "ic")
        + _grouped({"c": 0.5, "d": 0.5}, "g3", "ic")
    )
    with pytest.warns(RuntimeWarning, match="single result"):
        bhy_hierarchical(results, primary=["ic"], group="family", q=0.5)


def test_strong_group_survives_dead_group_does_not():
    make_spec("ic")
    results = _grouped({"hit_1": 1e-6, "hit_2": 1e-6}, "live", "ic") + _grouped(
        {"d1": 0.95, "d2": 0.95}, "dead", "ic"
    )
    out = bhy_hierarchical(results, primary=["ic"], group="family", q=0.05)
    surviving = {r.factor for r in out["ic"].survivors}
    assert surviving == {"hit_1", "hit_2"}


def test_empty_input_raises():
    make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[EvaluationResult\\]"):
        bhy_hierarchical([], primary=["ic"], group="family", q=0.05)


def test_primary_must_be_list_of_str():
    make_spec("ic")
    with pytest.raises(UserInputError, match="always a list"):
        bhy_hierarchical(
            [make_result(factor="f", p=0.01, primary="ic")],
            primary="ic",  # type: ignore[arg-type]
            group="family",
        )
