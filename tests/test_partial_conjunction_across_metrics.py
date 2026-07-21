"""Factor-level partial conjunction across predeclared metric endpoints."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import (
    CrossMetricPartialConjunctionResult,
    partial_conjunction_across_metrics,
)
from factrix._results import MetricResult

from .conftest import make_result


def _output(name: str, p: float, *, reason: str | None = None) -> MetricResult:
    metadata: dict[str, object] = {"p_value": p}
    if reason is not None:
        metadata["reason"] = reason
    return MetricResult(
        value=float("nan") if reason else 0.1,
        p_value=p,
        alternative="two-sided",
        n_obs=100,
        name=name,
        metadata=metadata,
    )


def _result(factor: str, ps: tuple[float, float, float], **kwargs):
    return make_result(
        factor=factor,
        p=ps[0],
        metric="ic",
        extra_outputs={
            "beta": _output("beta", ps[1]),
            "spread": _output("spread", ps[2]),
        },
        **kwargs,
    )


def test_computes_metric_k_of_m_then_bhy_across_factor_identities():
    results = [
        _result("strong", (0.001, 0.002, 0.8)),
        _result("weak", (0.1, 0.2, 0.3)),
    ]

    out = partial_conjunction_across_metrics(
        results,
        metrics=["ic", "beta", "spread"],
        min_pass=2,
        q=0.05,
    )

    assert isinstance(out, CrossMetricPartialConjunctionResult)
    np.testing.assert_allclose(out.pc_p_all, [0.004, 0.4])
    assert [entry.factor for entry in out.survivors] == ["strong"]
    assert out.n_identities == 2
    assert set(out.n_tests.values()) == {3}
    assert len(out.hypotheses) == 6


def test_insufficient_endpoint_keeps_fixed_m_and_ineligible_factor_is_audited():
    fixed_m = make_result(
        factor="fixed_m",
        p=0.001,
        metric="ic",
        extra_outputs={
            "beta": _output("beta", 0.002),
            "spread": _output("spread", 1.0, reason="insufficient_assets"),
        },
    )
    ineligible = make_result(
        factor="ineligible",
        p=0.001,
        metric="ic",
        extra_outputs={
            "beta": _output("beta", 1.0, reason="insufficient_periods"),
            "spread": _output("spread", 1.0, reason="insufficient_assets"),
        },
    )

    out = partial_conjunction_across_metrics(
        [fixed_m, ineligible],
        metrics=["ic", "beta", "spread"],
        min_pass=2,
        q=0.5,
    )
    frame = out.to_frame()

    # Fixed m=3 gives 2 * p_(2) = 0.004; silently shrinking to m=2 would
    # incorrectly produce 0.002.
    assert out.pc_p_all[0] == pytest.approx(0.004)
    assert np.isnan(out.pc_p_all[1])
    assert out.n_identities == 1
    assert frame["active"].to_list() == [True, False]
    assert frame["n_active"].to_list() == [2, 1]
    assert frame["n_tests"].to_list() == [3, 3]


def test_to_frame_reports_factor_level_contract():
    out = partial_conjunction_across_metrics(
        [_result("f1", (0.001, 0.002, 0.8))],
        metrics=["ic", "beta", "spread"],
        min_pass=2,
        q=0.5,
    )
    assert out.to_frame().columns == [
        "factor",
        "pc_p",
        "adj_p",
        "survived",
        "active",
        "n_tests",
        "n_active",
        "n_passed_uncorr",
    ]


@pytest.mark.parametrize("min_pass", [True, 1, 4, 2.5])
def test_min_pass_must_fit_declared_metric_count(min_pass):
    with pytest.raises(UserInputError, match="min_pass"):
        partial_conjunction_across_metrics(
            [_result("f1", (0.01, 0.02, 0.03))],
            metrics=["ic", "beta", "spread"],
            min_pass=min_pass,
        )


def test_descriptive_endpoint_fails_loudly():
    result = make_result(
        factor="f1",
        p=0.01,
        metric="ic",
        extra_outputs={"shape": MetricResult(value=0.1, name="shape")},
    )
    with pytest.raises(UserInputError, match="FDR control"):
        partial_conjunction_across_metrics(
            [result], metrics=["ic", "shape"], min_pass=2
        )


def test_mixed_horizons_warns_because_outer_bhy_pools_identities():
    results = [
        _result("f1", (0.01, 0.02, 0.03), forward_periods=1),
        _result("f2", (0.01, 0.02, 0.03), forward_periods=5),
    ]
    with pytest.warns(
        RuntimeWarning, match="partial_conjunction_across_metrics"
    ) as warning_records:
        partial_conjunction_across_metrics(
            results,
            metrics=["ic", "beta", "spread"],
            min_pass=2,
            q=0.5,
        )
    assert "filter the input by horizon" in str(warning_records[0].message)


def test_repr_and_html_show_factor_level_screen():
    out = partial_conjunction_across_metrics(
        [_result("f1", (0.001, 0.002, 0.8))],
        metrics=["ic", "beta", "spread"],
        min_pass=2,
        q=0.5,
    )
    assert "CrossMetricPartialConjunctionResult" in repr(out)
    assert "f1" in repr(out)
    assert "<table" in out._repr_html_()
