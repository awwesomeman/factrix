"""Third-party custom metrics: both documented registration paths must run
through ``fx.evaluate`` when the metric is defined in the *user's own* module.

See ``docs/guides/custom-metrics.md``: (1) the ``@metric`` decorator and
(2) ``@metric_spec`` + ``factrix.metrics.register``. The first used to fail
with ``ModuleNotFoundError: factrix.metrics.<user module>`` because the DAG
resolved the callable through a hardcoded ``factrix.metrics.{stem}`` import
path; the second was rejected outright by ``evaluate``'s metrics validation.
Defining the metrics in *this* module is the point of the test.
"""

from __future__ import annotations

import polars as pl
import pytest

import factrix as fx
from factrix import MetricResult, MetricSpec
from factrix._axis import Aggregation, InputShape
from factrix._errors import UserInputError
from factrix._metric_index import (
    _METRIC_REGISTRY,
    SampleThreshold,
    cell,
    _all_specs,
    _first_party_spec_by_name,
    public_specs,
)
from factrix.metrics._primitives._ic import compute_ic
from factrix.metrics._registry import REGISTRY

_CELL = cell(
    fx.FactorScope.INDIVIDUAL,
    fx.FactorDensity.DENSE,
    structure=fx.DataStructure.PANEL,
)


def _clear_caches() -> None:
    from factrix._dag import _registry_callable_table

    _first_party_spec_by_name.cache_clear()
    public_specs.cache_clear()
    _all_specs.cache_clear()
    _registry_callable_table.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Registration is global; keep it from leaking into other test modules."""
    before_third_party = dict(_METRIC_REGISTRY)
    before_classes = dict(REGISTRY)
    yield
    for name in set(_METRIC_REGISTRY) - set(before_third_party):
        _METRIC_REGISTRY.pop(name, None)
        if hasattr(fx.metrics, name):
            delattr(fx.metrics, name)
    for name in set(REGISTRY) - set(before_classes):
        REGISTRY.pop(name, None)
    _clear_caches()


@pytest.fixture
def panel():
    return fx.preprocess.compute_forward_return(
        fx.datasets.make_cs_panel(n_assets=30, n_dates=80, seed=0),
        forward_periods=2,
    )


def test_metric_decorator_path_runs_through_evaluate(panel):
    @fx.metrics.metric(
        cell=_CELL,
        aggregation=Aggregation.CS_THEN_TS,
        input_shape=InputShape.SERIES,
        requires={"ic_df": compute_ic},
        sample_threshold=SampleThreshold(min_periods=10),
    )
    def third_party_trimmed_ic(
        ic_df: pl.DataFrame, trim_ratio: float = 0.05
    ) -> MetricResult:
        """Trimmed mean IC — the worked example from the custom-metrics guide."""
        vals = ic_df["ic"].drop_nulls().sort()
        k = int(len(vals) * trim_ratio)
        trimmed = vals[k : len(vals) - k] if k else vals
        return MetricResult(
            value=float(trimmed.mean()),
            n_obs=len(trimmed),
            n_obs_axis="periods",
            metadata={"trim_ratio": trim_ratio},
        )

    # The decorated class lives in this test module, not in factrix.metrics.
    assert third_party_trimmed_ic.__module__ == __name__
    _clear_caches()

    results = fx.evaluate(
        panel,
        metrics={"trimmed": third_party_trimmed_ic(trim_ratio=0.1)},
        factor_cols=["factor"],
        forward_periods=2,
    )
    res = results["factor"].metrics["trimmed"]
    assert res.metadata["trim_ratio"] == 0.1
    assert res.n_obs > 0


def test_register_path_runs_through_evaluate(panel):
    @fx.metric_spec(
        MetricSpec(
            name="third_party_mean_return",
            cell=_CELL,
            aggregation=Aggregation.CS_THEN_TS,
        )
    )
    def third_party_mean_return(data: pl.DataFrame) -> MetricResult:
        return MetricResult(value=float(data["forward_return"].mean()))

    fx.metrics.register(third_party_mean_return)

    results = fx.evaluate(
        panel,
        metrics={"mean_ret": third_party_mean_return},
        factor_cols=["factor"],
        forward_periods=2,
    )
    res = results["factor"].metrics["mean_ret"]
    assert res.value == pytest.approx(float(panel["forward_return"].mean()))


def test_unstamped_callable_is_still_rejected(panel):
    def not_a_metric(data):
        return MetricResult(value=0.0)

    with pytest.raises(UserInputError, match="metrics"):
        fx.evaluate(
            panel,
            metrics={"nope": not_a_metric},
            factor_cols=["factor"],
            forward_periods=2,
        )
