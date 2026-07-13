"""``fx.multi_factor.partial_conjunction`` on the EvaluationResult contract."""

from __future__ import annotations

import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import PartialConjunctionResult, partial_conjunction

from .conftest import make_result, make_spec


def _replicate(factor: str, ps: list[float], primary, regions=("US", "EU", "JP")):
    return [
        make_result(factor=factor, p=p, metric=primary, params={"region": region})
        for p, region in zip(ps, regions, strict=False)
    ]


def test_returns_dict_per_primary():
    make_spec("ic")
    results = _replicate("alpha_1", [0.001, 0.001, 0.001], "ic") + _replicate(
        "alpha_2", [0.5, 0.5, 0.5], "ic"
    )
    out = partial_conjunction(
        results, metrics=["ic"], min_pass=2, expand_over=("region",), q=0.05
    )
    assert isinstance(out, dict)
    assert set(out) == {"ic"}
    assert isinstance(out["ic"], PartialConjunctionResult)


def test_pc_min_pass_one_raises():
    make_spec("ic")
    results = _replicate("alpha_1", [0.01, 0.01], "ic", regions=("US", "EU"))
    with pytest.raises(UserInputError, match="union semantics"):
        partial_conjunction(
            results, metrics=["ic"], min_pass=1, expand_over=("region",)
        )


def test_pc_min_pass_below_two_raises():
    make_spec("ic")
    results = _replicate("alpha_1", [0.01, 0.01], "ic", regions=("US", "EU"))
    with pytest.raises(UserInputError, match=">= 2"):
        partial_conjunction(
            results, metrics=["ic"], min_pass=0, expand_over=("region",)
        )


def test_pc_empty_expand_over_raises():
    make_spec("ic")
    results = _replicate("alpha_1", [0.01], "ic", regions=("US",))
    with pytest.raises(UserInputError, match="non-empty"):
        partial_conjunction(results, metrics=["ic"], min_pass=2, expand_over=())


def test_pc_empty_results_raises():
    make_spec("ic")
    with pytest.raises(UserInputError, match="non-empty list\\[EvaluationResult\\]"):
        partial_conjunction([], metrics=["ic"], min_pass=2, expand_over=("region",))


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1, float("nan"), True])
def test_pc_q_must_be_open_unit_interval(q):
    make_spec("ic")
    results = _replicate("alpha_1", [0.01, 0.01], "ic", regions=("US", "EU"))
    with pytest.raises(UserInputError, match="open interval"):
        partial_conjunction(
            results,
            metrics=["ic"],
            min_pass=2,
            expand_over=("region",),
            q=q,  # type: ignore[arg-type]
        )


def test_pc_list_of_dict_keys_suggests_values():
    make_spec("ic")
    results = {
        "alpha": make_result(
            factor="alpha", p=0.01, metric="ic", params={"region": "US"}
        )
    }
    mistaken: list[str] = []
    mistaken.extend(results)

    with pytest.raises(UserInputError, match=r"list\(results\.values\(\)\)"):
        partial_conjunction(  # type: ignore[arg-type]
            mistaken, metrics=["ic"], min_pass=2, expand_over=("region",)
        )


def test_pc_n_conditions_strict_mismatch_raises():
    make_spec("ic")
    results = _replicate("alpha_1", [0.01, 0.01], "ic", regions=("US", "EU"))
    with pytest.raises(UserInputError, match="condition"):
        partial_conjunction(
            results,
            metrics=["ic"],
            min_pass=2,
            expand_over=("region",),
            n_conditions=3,
        )


def test_pc_insufficient_conditions_raises():
    make_spec("ic")
    results = _replicate("alpha_1", [0.01], "ic", regions=("US",))
    with pytest.raises(UserInputError, match="condition"):
        partial_conjunction(
            results, metrics=["ic"], min_pass=2, expand_over=("region",)
        )


def test_pc_duplicate_condition_raises():
    make_spec("ic")
    results = [
        make_result(factor="alpha_1", p=0.01, metric="ic", params={"region": "US"}),
        make_result(factor="alpha_1", p=0.02, metric="ic", params={"region": "US"}),
    ]
    with pytest.raises(UserInputError, match="unique"):
        partial_conjunction(
            results, metrics=["ic"], min_pass=2, expand_over=("region",)
        )


def test_pc_strong_signal_survives():
    make_spec("ic")
    results = _replicate("strong", [0.0001, 0.0001, 0.0001], "ic") + _replicate(
        "weak", [0.4, 0.4, 0.4], "ic"
    )
    out = partial_conjunction(
        results, metrics=["ic"], min_pass=2, expand_over=("region",), q=0.05
    )
    factors = {r.factor for r in out["ic"].survivors}
    assert "strong" in factors
    assert "weak" not in factors


def test_pc_heterogeneous_m_warns_in_lenient_mode():
    make_spec("ic")
    results = _replicate("a", [0.01, 0.01], "ic", regions=("US", "EU")) + _replicate(
        "b", [0.01, 0.01, 0.01], "ic", regions=("US", "EU", "JP")
    )
    with pytest.warns(RuntimeWarning, match="heterogeneous condition"):
        partial_conjunction(
            results, metrics=["ic"], min_pass=2, expand_over=("region",), q=0.5
        )


def test_missing_param_key_aggregates_all_offenders():
    make_spec("ic")
    results = [
        make_result(factor="alpha", p=0.01, metric="ic", params={"region": "US"}),
        make_result(factor="alpha", p=0.01, metric="ic", params={}),
    ]
    with pytest.raises(UserInputError) as excinfo:
        partial_conjunction(
            results, metrics=["ic"], min_pass=2, expand_over=("region",)
        )
    assert "factor='alpha' missing 'region'" in str(excinfo.value)


def test_pc_non_condition_param_separates_identities():
    """A knob swept outside the condition axis must not inflate m.

    With `base_tf` on `params`, the four results are two identities of two
    conditions each — not one identity with four mixed-axis "conditions".
    """
    make_spec("ic")
    results = [
        make_result(
            factor="mom", p=p, metric="ic", params={"base_tf": tf, "region": rg}
        )
        for tf, rg, p in [
            ("1h", "US", 0.001),
            ("1h", "EU", 0.002),
            ("4h", "US", 0.9),
            ("4h", "EU", 0.8),
        ]
    ]
    out = partial_conjunction(
        results, metrics=["ic"], min_pass=2, expand_over=("region",), q=0.05
    )
    assert len(out["ic"].n_tests) == 2
    assert set(out["ic"].n_tests.values()) == {2}


def test_pc_mixed_horizons_outside_condition_axis_stay_distinct():
    """`forward_periods` outside `expand_over` splits identities, not conditions."""
    make_spec("ic")
    results = [
        make_result(
            factor="mom",
            p=0.001,
            metric="ic",
            forward_periods=fp,
            params={"region": rg},
        )
        for fp in (5, 10)
        for rg in ("US", "EU")
    ]
    out = partial_conjunction(
        results, metrics=["ic"], min_pass=2, expand_over=("region",), q=0.05
    )
    assert len(out["ic"].n_tests) == 2
    assert set(out["ic"].n_tests.values()) == {2}
