"""The ``inactive_policy="exclude"`` branch, shared by every screening verb.

A placeholder cell — a data-shortage short-circuit or a
``degenerate_variance`` result — never ran a test. Under the opt-in
``inactive_policy="exclude"`` it leaves every family before any adjustment,
whichever verb screens it; the default ``"count"`` branch is pinned in
``test_inactive_policy.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from factrix._codes import WarningCode
from factrix._multi_factor import (
    bhy,
    bhy_across_metrics,
    bhy_hierarchical,
    partial_conjunction,
    partial_conjunction_across_metrics,
)
from factrix._results import MetricResult

from .conftest import make_result, make_spec


def _real(factor: str, p: float, *, metric: str = "ic", **kwargs):
    return make_result(factor=factor, p=p, metric=metric, **kwargs)


def _shortage(factor: str, *, metric: str = "ic", **kwargs):
    return make_result(
        factor=factor,
        p=1.0,
        metric=metric,
        metadata={"reason": "insufficient_periods"},
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


# ``bhy_adjusted_p([0.01, 0.5])`` on the two real hypotheses.
EXPECTED_ADJ = [0.03, 0.75]


def test_bhy_excludes_placeholders_and_reports_the_count():
    make_spec("ic")
    out = bhy(
        [_real("a", 0.01), _real("b", 0.5), _shortage("c"), _degenerate("d")],
        metrics=["ic"],
        inactive_policy="exclude",
    )["ic"]

    np.testing.assert_allclose(out.adj_p_all[:2], EXPECTED_ADJ)
    assert np.isnan(out.adj_p_all[2:]).all()
    assert out.family_size == {(): 2}
    assert out.family.n_inactive == 2


def test_partial_conjunction_matches_bhy_on_the_same_degenerate_family():
    """Full conjunction (k = m = 2) reduces the PC p to ``max(p)``.

    So the outer step-up sees the same ``[0.01, 0.5]`` family ``bhy`` sees,
    and the shared hypotheses must land on the same adjusted p-values.
    """
    make_spec("ic")
    results = [
        _real("a", 0.01, params={"region": "US"}),
        _real("a", 0.01, params={"region": "EU"}),
        _real("b", 0.5, params={"region": "US"}),
        _real("b", 0.5, params={"region": "EU"}),
        _shortage("c", params={"region": "US"}),
        _degenerate("c", params={"region": "EU"}),
    ]
    out = partial_conjunction(
        results,
        metrics=["ic"],
        min_pass=2,
        expand_over=("region",),
        inactive_policy="exclude",
    )["ic"]

    np.testing.assert_allclose(out.pc_p_all[:2], [0.01, 0.5])
    np.testing.assert_allclose(out.adj_p_all[:2], EXPECTED_ADJ)
    assert np.isnan(out.adj_p_all[2])
    assert out.family_size[("c", 5)] == 0
    assert out.family.n_inactive == 2


def test_hierarchical_matches_bhy_on_the_same_degenerate_family():
    """Two singleton-after-exclusion groups make the inner layer inert.

    ``inner_adj`` is the member's own p and ``q`` is loose enough that the
    outer layer selects both groups (``R = G``, no selection inflation), so
    each member's adjusted p is its outer BHY value — the same ``[0.03,
    0.75]`` the flat screen produces on the two real hypotheses.
    """
    make_spec("ic")
    results = [
        _real("a", 0.01, params={"family": "momentum"}),
        _degenerate("a_dead", params={"family": "momentum"}),
        _real("b", 0.5, params={"family": "value"}),
        _shortage("b_dead", params={"family": "value"}),
        _shortage("c", params={"family": "carry"}),
    ]
    with pytest.warns(RuntimeWarning, match="family size 1"):
        out = bhy_hierarchical(
            results,
            metrics=["ic"],
            group="family",
            q=0.8,
            inactive_policy="exclude",
        )["ic"]

    np.testing.assert_allclose(out.adj_p_all[0], EXPECTED_ADJ[0])
    np.testing.assert_allclose(out.adj_p_all[2], EXPECTED_ADJ[1])
    assert np.isnan(out.adj_p_all[[1, 3, 4]]).all()
    # The all-placeholder group leaves G entirely.
    assert set(out.family_size) == {("momentum",), ("value",)}
    assert out.family.n_inactive == 3


@pytest.mark.parametrize(
    "verb",
    ["bhy", "partial_conjunction", "bhy_hierarchical"],
)
def test_a_placeholder_costs_the_real_hypotheses_nothing(verb):
    """Submitting a placeholder gives the same adjusted p as not submitting it."""
    make_spec("ic")
    if verb == "bhy":
        live = [_real("a", 0.01), _real("b", 0.2), _real("c", 0.5)]
        dead = [_shortage("d"), _degenerate("e")]

        def run(results):
            return bhy(results, metrics=["ic"], inactive_policy="exclude")[
                "ic"
            ].adj_p_all[:3]

    elif verb == "partial_conjunction":
        live = [
            _real(factor, p, params={"region": region})
            for factor, p in (("a", 0.01), ("b", 0.2), ("c", 0.5))
            for region in ("US", "EU")
        ]
        dead = [
            _shortage("d", params={"region": "US"}),
            _degenerate("d", params={"region": "EU"}),
        ]

        def run(results):
            out = partial_conjunction(
                results,
                metrics=["ic"],
                min_pass=2,
                expand_over=("region",),
                inactive_policy="exclude",
            )["ic"]
            return out.adj_p_all[:3]

    else:
        live = [
            _real("a", 0.01, params={"family": "momentum"}),
            _real("b", 0.2, params={"family": "momentum"}),
            _real("c", 0.5, params={"family": "value"}),
            _real("d", 0.6, params={"family": "value"}),
        ]
        dead = [
            _degenerate("e", params={"family": "momentum"}),
            _shortage("f", params={"family": "carry"}),
        ]

        def run(results):
            out = bhy_hierarchical(
                results,
                metrics=["ic"],
                group="family",
                q=0.05,
                inactive_policy="exclude",
            )
            return out["ic"].adj_p_all[:4]

    np.testing.assert_allclose(run(live), run(live + dead))


def test_cross_metric_verbs_report_the_same_inactive_count():
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

    flat = bhy_across_metrics(results, metrics=metrics, inactive_policy="exclude")
    factor_level = partial_conjunction_across_metrics(
        results, metrics=metrics, min_pass=2, q=0.05, inactive_policy="exclude"
    )

    assert flat.family.n_inactive == 1
    assert factor_level.family.n_inactive == 1
    assert np.isnan(flat.adj_p_all[2])
    # m drops to the two real endpoints: p_PC = (2 - 2 + 1) * p_(2) = 0.02.
    assert factor_level.pc_p_all[0] == pytest.approx(0.02)
    assert factor_level.family_size[("a", 5)] == 2


def test_n_passed_uncorr_counts_the_boundary_like_every_rejection_rule():
    """``p == q`` is a rejection everywhere else, so it counts here too."""
    make_spec("ic")
    results = [
        _real("a", 0.05, params={"region": "US"}),
        _real("a", 0.05, params={"region": "EU"}),
    ]
    out = partial_conjunction(
        results,
        metrics=["ic"],
        min_pass=2,
        expand_over=("region",),
        q=0.05,
        inactive_policy="exclude",
    )["ic"]
    assert out.n_passed_uncorr_all.tolist() == [2]

    cross = partial_conjunction_across_metrics(
        [
            make_result(
                factor="a",
                p=0.05,
                metric="ic",
                extra_outputs={
                    "beta": _output("beta", 0.05),
                    "spread": _output("spread", 0.9),
                },
            )
        ],
        metrics=["ic", "beta", "spread"],
        min_pass=2,
        q=0.05,
    )
    assert cross.n_passed_uncorr_all.tolist() == [2]
