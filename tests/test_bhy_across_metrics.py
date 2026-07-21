"""Cross-metric pooled BHY on traceable ``EvaluationResult`` cells."""

from __future__ import annotations

import factrix as fx
import numpy as np
import pytest
from factrix._errors import UserInputError
from factrix._multi_factor import CrossMetricBhyResult, bhy_across_metrics
from factrix._results import MetricResult
from factrix.stats.multiple_testing import bhy_adjusted_p

from .conftest import make_result, make_spec


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


def _two_metric_result(factor: str, p_ic: float, p_spread: float, **kwargs):
    return make_result(
        factor=factor,
        p=p_ic,
        metric="ic",
        extra_outputs={"spread": _output("spread", p_spread)},
        **kwargs,
    )


def test_pools_factor_by_metric_cells_into_one_family():
    make_spec("ic")
    make_spec("spread")
    results = [
        _two_metric_result("f1", 0.001, 0.02),
        _two_metric_result("f2", 0.03, 0.4),
    ]

    out = bhy_across_metrics(results, metrics=["ic", "spread"], q=0.05)

    assert isinstance(out, CrossMetricBhyResult)
    assert out.n_tests == {(): 4}
    assert [(e.result.factor, e.metric_name) for e in out.entries] == [
        ("f1", "ic"),
        ("f1", "spread"),
        ("f2", "ic"),
        ("f2", "spread"),
    ]
    np.testing.assert_allclose(
        out.adj_p_all,
        bhy_adjusted_p(np.array([0.001, 0.02, 0.03, 0.4])),
    )


def test_public_multi_factor_namespace_exports_cross_metric_api():
    assert fx.multi_factor.bhy_across_metrics is bhy_across_metrics
    assert fx.multi_factor.CrossMetricBhyResult is CrossMetricBhyResult
    assert fx.multi_factor.MetricHypothesis.__name__ == "MetricHypothesis"


def test_to_frame_keeps_raw_p_metric_and_survival_identity():
    results = [
        _two_metric_result("f1", 0.001, 0.02),
        _two_metric_result("f2", 0.03, 0.4),
    ]
    out = bhy_across_metrics(results, metrics=["ic", "spread"], q=0.5)

    frame = out.to_frame()
    assert frame.columns == [
        "factor",
        "metric",
        "p_value",
        "adj_p",
        "survived",
        "active",
    ]
    assert frame.height == 4
    assert frame["metric"].to_list() == ["ic", "spread", "ic", "spread"]


def test_insufficient_cell_stays_auditable_but_not_active():
    valid = _two_metric_result("valid", 0.01, 0.02)
    thin = make_result(
        factor="thin",
        p=0.03,
        metric="ic",
        extra_outputs={"spread": _output("spread", 1.0, reason="insufficient_assets")},
    )

    out = bhy_across_metrics([valid, thin], metrics=["ic", "spread"], q=0.5)
    frame = out.to_frame()
    inactive = frame.filter(~frame["active"])

    assert out.n_tests == {(): 3}
    assert inactive.height == 1
    assert inactive["factor"].item() == "thin"
    assert inactive["metric"].item() == "spread"
    assert np.isnan(inactive["adj_p"].item())


def test_expand_over_partitions_after_metric_flattening():
    results = [
        _two_metric_result("us", 0.01, 0.02, params={"region": "US"}),
        _two_metric_result("eu", 0.03, 0.04, params={"region": "EU"}),
    ]

    out = bhy_across_metrics(
        results,
        metrics=["ic", "spread"],
        expand_over=("region",),
        q=0.5,
    )

    assert out.n_tests == {("US",): 2, ("EU",): 2}
    assert out.expand_over == ("region",)


def test_descriptive_metric_fails_loudly():
    descriptive = MetricResult(value=0.1, p_value=None, name="shape")
    results = [
        make_result(
            factor="f1",
            p=0.01,
            metric="ic",
            extra_outputs={"shape": descriptive},
        )
    ]
    with pytest.raises(UserInputError, match="FDR control") as excinfo:
        bhy_across_metrics(results, metrics=["ic", "shape"])
    assert "/api/bhy-across-metrics#metrics" in excinfo.value.docs_url


@pytest.mark.parametrize("metrics", [["ic"], ["ic", "ic"]])
def test_metric_axis_requires_two_unique_labels(metrics):
    results = [_two_metric_result("f1", 0.01, 0.02)]
    with pytest.raises(UserInputError, match="metric"):
        bhy_across_metrics(results, metrics=metrics)


def test_metric_container_error_links_to_deployed_page():
    results = [_two_metric_result("f1", 0.01, 0.02)]
    with pytest.raises(UserInputError) as excinfo:
        bhy_across_metrics(results, metrics=("ic", "spread"))  # type: ignore[arg-type]
    assert "/api/bhy-across-metrics#metrics" in excinfo.value.docs_url


def test_mixed_horizons_warns_with_cross_metric_function_name():
    results = [
        _two_metric_result("f1", 0.01, 0.02, forward_periods=1),
        _two_metric_result("f2", 0.03, 0.04, forward_periods=5),
    ]
    with pytest.warns(RuntimeWarning, match="bhy_across_metrics"):
        bhy_across_metrics(results, metrics=["ic", "spread"], q=0.5)


def test_repr_and_html_show_metric_identity():
    out = bhy_across_metrics(
        [_two_metric_result("f1", 0.001, 0.002)],
        metrics=["ic", "spread"],
        q=0.5,
    )
    assert "CrossMetricBhyResult" in repr(out)
    assert "spread" in repr(out)
    assert "<table" in out._repr_html_()
