"""``inactive_policy`` — who belongs to the declared family (#1057).

The default ``"count"`` keeps every declared candidate in ``m`` and lets an
inactive one participate as an inert ``p = 1``. ``"exclude"`` drops it from
``m``, which is the sharper screen but assumes the activity filter is
independent of the p-values or pre-specified; that branch's cross-verb
consistency is pinned in ``test_screening_placeholder_policy.py``.

Every numeric expectation here is computed from ``bhy_adjusted_p`` in
``test_the_documented_numeric_example_is_the_one_the_code_produces`` before
being asserted anywhere else, and the same numbers appear in
``docs/api/multi-factor.md``.
"""

from __future__ import annotations

import html

import numpy as np
import pytest
from factrix._codes import WarningCode
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

#: The issue's reproduction: one active hypothesis at p = 0.01 plus nine
#: ``insufficient_ic_periods`` candidates. Recomputed from ``bhy_adjusted_p``
#: in the test named in the module docstring, not hand-copied from the issue.
ADJ_M10 = 0.2928968253968254
ADJ_M1 = 0.01


def _real(factor: str, p: float, *, metric: str = "ic", **kwargs):
    return make_result(factor=factor, p=p, metric=metric, **kwargs)


def _shortage(factor: str, *, metric: str = "ic", **kwargs):
    return make_result(
        factor=factor,
        p=1.0,
        metric=metric,
        metadata={"reason": "insufficient_ic_periods"},
        **kwargs,
    )


def _degenerate(factor: str, *, metric: str = "ic", **kwargs):
    return make_result(
        factor=factor,
        p=None,
        metric=metric,
        warning_codes=(WarningCode.DEGENERATE_VARIANCE.value,),
        **kwargs,
    )


def _output(name: str, p: float | None, *, reason: str | None = None) -> MetricResult:
    metadata: dict[str, object] = {} if p is None else {"p_value": p}
    if reason is not None:
        metadata["reason"] = reason
    return MetricResult(
        value=float("nan") if reason else 0.1,
        p_value=p,
        alternative=None if p is None else "two-sided",
        n_obs=100,
        name=name,
        metadata=metadata,
    )


def _one_active_nine_short() -> list:
    return [_real("a", 0.01)] + [_shortage(f"d{i}") for i in range(9)]


def test_the_documented_numeric_example_is_the_one_the_code_produces():
    """Measure the m=1 vs m=10 claim instead of asserting it.

    ``docs/api/multi-factor.md`` and this module's constants both quote
    these two numbers; this test is where they come from.
    """
    assert bhy_adjusted_p(np.array([0.01]))[0] == pytest.approx(ADJ_M1)
    ten = bhy_adjusted_p(np.array([0.01] + [1.0] * 9))
    assert ten[0] == pytest.approx(ADJ_M10)
    # The nine inert members are themselves never rejectable.
    assert ten[1:].tolist() == [1.0] * 9
    # And the documented rounding is the one the docs print.
    assert f"{ADJ_M10:.3f}" == "0.293"


def test_one_active_plus_nine_insufficient_counts_all_ten_by_default():
    make_spec("ic")
    out = bhy(_one_active_nine_short(), metrics=["ic"])["ic"]

    assert out.family_size == {(): 10}
    assert out.adj_p_all[0] == pytest.approx(ADJ_M10)
    assert out.adj_p_all[1:].tolist() == [1.0] * 9
    assert not out.survivors  # 0.293 > q = 0.05


def test_exclude_is_the_opt_in_that_reproduces_the_sharper_screen():
    make_spec("ic")
    out = bhy(_one_active_nine_short(), metrics=["ic"], inactive_policy="exclude")["ic"]

    assert out.family_size == {(): 1}
    assert out.adj_p_all[0] == pytest.approx(ADJ_M1)
    assert np.isnan(out.adj_p_all[1:]).all()
    assert [r.factor for r in out.survivors] == ["a"]


def test_the_four_counts_and_the_policy_are_disclosed():
    make_spec("ic")
    counted = bhy(_one_active_nine_short(), metrics=["ic"])["ic"]
    excluded = bhy(_one_active_nine_short(), metrics=["ic"], inactive_policy="exclude")[
        "ic"
    ]

    assert counted.family.policy == "count"
    assert counted.family.n_candidates_declared == 10
    assert counted.family.n_tests_computed == 1
    assert counted.family.n_inactive == 9
    assert counted.family.n_tests_adjusted == 10

    assert excluded.family.policy == "exclude"
    assert excluded.family.n_candidates_declared == 10
    assert excluded.family.n_tests_computed == 1
    assert excluded.family.n_inactive == 9
    assert excluded.family.n_tests_adjusted == 1


@pytest.mark.parametrize("policy", ["count", "exclude"])
def test_repr_shows_the_family_counts_and_the_policy(policy):
    """Assert the whole disclosure token, not its words separately."""
    make_spec("ic")
    out = bhy(_one_active_nine_short(), metrics=["ic"], inactive_policy=policy)["ic"]
    adjusted = 10 if policy == "count" else 1

    token = (
        f"family(declared=10, computed=1, inactive=9, "
        f"unsubmitted=0, "
        f"adjusted={adjusted}, policy={policy!r})"
    )
    assert token in repr(out)
    # The HTML caption is escaped, so compare against the escaped token.
    assert html.escape(token) in out._repr_html_()


def test_data_dependent_missingness_changes_the_answer_not_just_the_count():
    """The filter is not innocuous when activity tracks the effect.

    Here the inactive candidates are exactly the ones a data-dependent
    shortage removed. Excluding them leaves a family of small p-values that
    all survive; counting them rejects nothing. Same input, two verdicts —
    which is why the choice is an explicit kwarg rather than a default.
    """
    make_spec("ic")
    results = [_real(f"live{i}", 0.008) for i in range(3)] + [
        _shortage(f"dead{i}") for i in range(7)
    ]

    counted = bhy(results, metrics=["ic"], q=0.05)["ic"]
    excluded = bhy(results, metrics=["ic"], q=0.05, inactive_policy="exclude")["ic"]

    # m = 10: adjusted p 0.0781 > q. m = 3: adjusted p 0.0147 <= q.
    assert counted.survivors == []
    assert [r.factor for r in excluded.survivors] == ["live0", "live1", "live2"]


def test_an_all_inactive_family_is_inert_under_count_and_empty_under_exclude():
    make_spec("ic")
    results = [_shortage("a"), _degenerate("b"), _shortage("c")]

    counted = bhy(results, metrics=["ic"])["ic"]
    assert counted.family_size == {(): 3}
    assert counted.adj_p_all.tolist() == [1.0, 1.0, 1.0]
    assert counted.survivors == []
    assert counted.family.n_tests_computed == 0
    assert counted.family.n_tests_adjusted == 3

    excluded = bhy(results, metrics=["ic"], inactive_policy="exclude")["ic"]
    assert excluded.family_size == {}
    assert np.isnan(excluded.adj_p_all).all()
    assert excluded.family.n_tests_adjusted == 0


def test_a_singleton_family_still_declares_its_one_candidate():
    make_spec("ic")
    single = bhy([_real("a", 0.01)], metrics=["ic"])["ic"]
    assert single.family_size == {(): 1}
    assert single.family.n_candidates_declared == 1
    assert single.family.n_tests_adjusted == 1
    assert single.adj_p_all[0] == pytest.approx(ADJ_M1)

    # One active candidate submitted beside one inactive one is a family of
    # two under the default — the singleton is a consequence of the policy,
    # not of the input.
    pair = bhy([_real("a", 0.01), _shortage("b")], metrics=["ic"])["ic"]
    assert pair.family_size == {(): 2}
    assert pair.adj_p_all[0] == pytest.approx(bhy_adjusted_p(np.array([0.01, 1.0]))[0])


def test_bucketed_family_counts_inactive_candidates_in_their_own_bucket():
    make_spec("ic")
    results = [
        _real("a", 0.01, params={"universe": "US"}),
        _shortage("b", params={"universe": "US"}),
        _real("c", 0.02, params={"universe": "EU"}),
        _real("d", 0.03, params={"universe": "EU"}),
    ]

    counted = bhy(results, metrics=["ic"], expand_over=("universe",))["ic"]
    assert counted.family_size == {("US",): 2, ("EU",): 2}
    assert counted.family.n_tests_adjusted == 4

    with pytest.warns(RuntimeWarning, match="family size 1"):
        excluded = bhy(
            results,
            metrics=["ic"],
            expand_over=("universe",),
            inactive_policy="exclude",
        )["ic"]
    assert excluded.family_size == {("US",): 1, ("EU",): 2}
    assert excluded.family.n_tests_adjusted == 3
    # The US bucket's live hypothesis is corrected against m=2, not m=1.
    assert counted.adj_p_all[0] > excluded.adj_p_all[0]


def test_every_high_level_verb_takes_the_same_kwarg_and_defaults_to_count():
    """One shared policy: the default is ``"count"`` on all five verbs."""
    make_spec("ic")
    flat = [_real("a", 0.01), _shortage("b")]
    conditioned = [
        _real("a", 0.01, params={"region": "US"}),
        _real("a", 0.01, params={"region": "EU"}),
        _shortage("b", params={"region": "US"}),
        _shortage("b", params={"region": "EU"}),
    ]
    grouped = [
        _real("a", 0.01, params={"family": "momentum"}),
        _real("b", 0.02, params={"family": "momentum"}),
        _shortage("c", params={"family": "carry"}),
        _shortage("d", params={"family": "carry"}),
    ]
    cross = [
        make_result(
            factor="a",
            p=0.01,
            metric="ic",
            extra_outputs={
                "beta": _output("beta", 0.02),
                "spread": _output("spread", 1.0, reason="insufficient_assets"),
            },
        ),
        make_result(
            factor="b",
            p=0.5,
            metric="ic",
            extra_outputs={
                "beta": _output("beta", 0.6),
                "spread": _output("spread", 0.7),
            },
        ),
    ]
    metrics = ["ic", "beta", "spread"]

    families = [
        bhy(flat, metrics=["ic"])["ic"].family,
        bhy_across_metrics(cross, metrics=metrics).family,
        partial_conjunction(
            conditioned, metrics=["ic"], min_pass=2, expand_over=("region",)
        )["ic"].family,
        partial_conjunction_across_metrics(cross, metrics=metrics, min_pass=2).family,
        bhy_hierarchical(grouped, metrics=["ic"], group="family")["ic"].family,
    ]
    assert [f.policy for f in families] == ["count"] * 5
    # Every one of them saw its inactive candidates and kept them in m.
    assert [f.n_inactive for f in families] == [1, 1, 2, 1, 2]
    assert [f.n_tests_adjusted == f.n_candidates_declared for f in families] == [
        True
    ] * 5


def test_partial_conjunction_keeps_an_inactive_condition_in_the_denominator():
    """``m`` stays 2, so ``p_PC = (2 - 2 + 1) * p_(2) = 1.0``."""
    make_spec("ic")
    results = [
        _real("a", 0.01, params={"region": "US"}),
        _degenerate("a", params={"region": "EU"}),
        _real("b", 0.005, params={"region": "US"}),
        _real("b", 0.006, params={"region": "EU"}),
    ]
    out = partial_conjunction(
        results, metrics=["ic"], min_pass=2, expand_over=("region",)
    )["ic"]

    assert out.family_size[("a", 5)] == 2
    assert out.pc_p_all[0] == pytest.approx(1.0)
    # The inert condition is never one of the k passes.
    assert out.n_passed_uncorr_all.tolist() == [1, 2]
    assert [r.factor for r in out.survivors] == ["b"]


def test_hierarchical_keeps_an_all_inactive_group_in_g():
    make_spec("ic")
    results = [
        _real("a", 0.001, params={"family": "momentum"}),
        _real("b", 0.002, params={"family": "momentum"}),
        _shortage("c", params={"family": "carry"}),
        _shortage("d", params={"family": "carry"}),
    ]
    counted = bhy_hierarchical(results, metrics=["ic"], group="family")["ic"]
    excluded = bhy_hierarchical(
        results, metrics=["ic"], group="family", inactive_policy="exclude"
    )["ic"]

    # G = 2 under "count" (the dead group enters at a Simes p of 1.0) and
    # G = 1 under "exclude"; the live members pay the wider correction.
    assert set(counted.family_size) == {("momentum",), ("carry",)}
    assert counted.family_size[("carry",)] == 2
    assert set(excluded.family_size) == {("momentum",)}
    assert counted.adj_p_all[0] > excluded.adj_p_all[0]
    assert np.isnan(excluded.adj_p_all[2:]).all()
    assert counted.adj_p_all[2:].tolist() == [1.0, 1.0]


def test_cross_metric_verbs_count_an_inactive_endpoint_in_m():
    make_spec("ic")
    results = [
        make_result(
            factor="a",
            p=0.01,
            metric="ic",
            extra_outputs={
                "beta": _output("beta", 0.02),
                "spread": _output("spread", 1.0, reason="insufficient_assets"),
            },
        ),
        make_result(
            factor="b",
            p=0.5,
            metric="ic",
            extra_outputs={
                "beta": _output("beta", 0.6),
                "spread": _output("spread", 0.7),
            },
        ),
    ]
    metrics = ["ic", "beta", "spread"]

    flat = bhy_across_metrics(results, metrics=metrics)
    assert flat.family_size == {(): 6}
    assert flat.adj_p_all[2] == pytest.approx(1.0)  # the inert endpoint
    assert flat.family.n_tests_computed == 5

    factor_level = partial_conjunction_across_metrics(
        results, metrics=metrics, min_pass=2, q=0.05
    )
    # m stays 3, so p_PC = (3 - 2 + 1) * p_(2) = 2 * 0.02 = 0.04.
    assert factor_level.family_size[("a", 5)] == 3
    assert factor_level.pc_p_all[0] == pytest.approx(0.04)
    assert factor_level.to_frame()["family_size"].to_list() == [3, 3]


@pytest.mark.parametrize(
    ("verb", "call"),
    [
        ("bhy", lambda results, **kw: bhy(results, metrics=["ic"], **kw)),
        (
            "bhy_across_metrics",
            lambda results, **kw: bhy_across_metrics(
                results, metrics=["ic", "beta"], **kw
            ),
        ),
        (
            "partial_conjunction",
            lambda results, **kw: partial_conjunction(
                results, metrics=["ic"], min_pass=2, expand_over=("region",), **kw
            ),
        ),
        (
            "partial_conjunction_across_metrics",
            lambda results, **kw: partial_conjunction_across_metrics(
                results, metrics=["ic", "beta"], min_pass=2, **kw
            ),
        ),
        (
            "bhy_hierarchical",
            lambda results, **kw: bhy_hierarchical(
                results, metrics=["ic"], group="region", **kw
            ),
        ),
    ],
)
def test_an_unknown_policy_is_rejected_by_every_verb(verb, call):
    make_spec("ic")
    results = [
        make_result(
            factor=f,
            p=0.01,
            metric="ic",
            params={"region": region},
            extra_outputs={"beta": _output("beta", 0.02)},
        )
        for f in ("a", "b")
        for region in ("US", "EU")
    ]
    with pytest.raises(UserInputError) as excinfo:
        call(results, inactive_policy="drop")
    message = str(excinfo.value)
    assert "inactive_policy" in message
    assert verb in message
