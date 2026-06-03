"""Third-party metric registration — ``@metric_spec`` decorator + ``register()`` (#444)."""

from __future__ import annotations

import factrix as fx
import polars as pl
import pytest
from factrix._axis import Aggregation, SEMethod, TestMethod
from factrix._metric_index import (
    _METRIC_REGISTRY,
    MetricSpec,
    cell,
    spec_by_name,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Clear any third-party registry entries / namespace attrs between tests."""
    before = dict(_METRIC_REGISTRY)
    yield
    added = set(_METRIC_REGISTRY) - set(before)
    for name in added:
        _METRIC_REGISTRY.pop(name, None)
        if hasattr(fx.metrics, name):
            delattr(fx.metrics, name)
    from factrix._dag import _registry_callable_table

    _registry_callable_table.cache_clear()


class TestMetricSpecDecorator:
    def test_stamps_metric_spec_attribute(self) -> None:
        spec = MetricSpec(
            name="stamp_only", cell=cell(None, None), aggregation=Aggregation.CS_THEN_TS, test_method=TestMethod.T, se_method=SEMethod.HAC
        )

        @fx.metric_spec(spec)
        def stamp_only(panel: pl.DataFrame):
            return 0

        assert stamp_only.__metric_spec__ is spec

    def test_decorator_does_not_auto_register(self) -> None:
        spec = MetricSpec(
            name="no_auto_register",
            cell=cell(None, None),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )

        @fx.metric_spec(spec)
        def no_auto_register(panel: pl.DataFrame):
            return 0

        assert "no_auto_register" not in spec_by_name()
        assert not hasattr(fx.metrics, "no_auto_register")


class TestRegister:
    def test_register_sets_real_module_attr(self) -> None:
        spec = MetricSpec(
            name="custom_ic_smoke",
            cell=cell(None, None),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )

        @fx.metric_spec(spec)
        def custom_ic_smoke(panel: pl.DataFrame):
            return 42

        fx.metrics.register(custom_ic_smoke)

        assert fx.metrics.custom_ic_smoke is custom_ic_smoke
        assert "custom_ic_smoke" in spec_by_name()
        assert spec_by_name()["custom_ic_smoke"] is spec

    def test_register_callable_resolves_through_dag_callable_table(self) -> None:
        from factrix._dag import _registry_callable_table

        spec = MetricSpec(
            name="dag_resolves_me",
            cell=cell(None, None),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )

        @fx.metric_spec(spec)
        def dag_resolves_me(panel: pl.DataFrame):
            return 0

        fx.metrics.register(dag_resolves_me)
        assert _registry_callable_table()["dag_resolves_me"] is dag_resolves_me

    def test_register_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            fx.metrics.register("not a function")  # type: ignore[arg-type]

    def test_register_rejects_callable_without_metric_spec(self) -> None:
        def plain_function():
            return 0

        with pytest.raises(TypeError, match="__metric_spec__"):
            fx.metrics.register(plain_function)

    def test_register_rejects_duplicate(self) -> None:
        spec = MetricSpec(
            name="dup_metric",
            cell=cell(None, None),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )

        @fx.metric_spec(spec)
        def dup_metric():
            return 0

        fx.metrics.register(dup_metric)
        with pytest.raises(ValueError, match="already registered"):
            fx.metrics.register(dup_metric)

    def test_register_rejects_first_party_clash(self) -> None:
        clash_spec = MetricSpec(
            name="ic", cell=cell(None, None), aggregation=Aggregation.CS_THEN_TS, test_method=TestMethod.T, se_method=SEMethod.HAC
        )

        @fx.metric_spec(clash_spec)
        def ic_clash():
            return 0

        with pytest.raises(ValueError, match="first-party"):
            fx.metrics.register(ic_clash)
