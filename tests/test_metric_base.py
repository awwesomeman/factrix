import polars as pl
import pytest
from factrix._axis import Aggregation, OutputShape, SpecRole
from factrix._metric_index import Cell, spec_by_name
from factrix.metrics import MetricBase, metric

_TEST_CELL = Cell(scope=None, density=None, structure=None, raw="(*, *, *)")


def test_metric_base_dataclass_properties():
    from factrix.metrics._registry import REGISTRY

    try:

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            output_shape=OutputShape.PANEL,
            role=SpecRole.PIPELINE,
        )
        def dummy_pipeline(df: pl.DataFrame, shift: int = 1) -> pl.DataFrame:
            return df.with_columns(pl.col("value") + shift)

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            requires={"df_input": dummy_pipeline},
        )
        def dummy_metric(
            df_input: pl.DataFrame, multiplier: float = 2.0, suffix: str = ""
        ) -> str:
            val = df_input["value"][0]
            return f"val={val * multiplier}{suffix}"

        # Verify dummy_metric is a subclass of MetricBase
        assert issubclass(dummy_metric, MetricBase)

        # Verify it is a dataclass
        import dataclasses

        assert dataclasses.is_dataclass(dummy_metric)

        # Instantiate it with config
        m = dummy_metric(multiplier=3.0, suffix="!")
        assert m.multiplier == 3.0
        assert m.suffix == "!"
        assert m.cell == _TEST_CELL
        assert m.aggregation == Aggregation.TS_ONLY
    finally:
        for name in ["dummy_pipeline", "dummy_metric"]:
            if name in REGISTRY:
                del REGISTRY[name]
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()


def test_metric_spec_generation():
    from factrix.metrics._registry import REGISTRY

    try:

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
        )
        def dummy_pipeline(df: pl.DataFrame, shift: int = 1) -> pl.DataFrame:
            return df.with_columns(pl.col("value") + shift)

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            requires={"df_input": dummy_pipeline},
        )
        def dummy_metric(
            df_input: pl.DataFrame, multiplier: float = 2.0, suffix: str = ""
        ) -> str:
            val = df_input["value"][0]
            return f"val={val * multiplier}{suffix}"

        # Verify the spec can be dynamically constructed
        spec = dummy_metric.spec()
        assert spec.name == "dummy_metric"
        assert spec.cell == _TEST_CELL
        assert spec.aggregation == Aggregation.TS_ONLY
        assert spec.requires == {"df_input": dummy_pipeline}
    finally:
        for name in ["dummy_pipeline", "dummy_metric"]:
            if name in REGISTRY:
                del REGISTRY[name]
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()


def test_metric_dual_interface():
    from factrix.metrics._registry import REGISTRY

    try:

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
        )
        def dummy_pipeline(df: pl.DataFrame, shift: int = 1) -> pl.DataFrame:
            return df.with_columns(pl.col("value") + shift)

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            requires={"df_input": dummy_pipeline},
        )
        def dummy_metric(
            df_input: pl.DataFrame, multiplier: float = 2.0, suffix: str = ""
        ) -> str:
            val = df_input["value"][0]
            return f"val={val * multiplier}{suffix}"

        df = pl.DataFrame({"value": [10]})

        # 1. Instantiation + call style
        pipeline_inst = dummy_pipeline(shift=5)
        pipeline_out = pipeline_inst(df)
        assert pipeline_out["value"][0] == 15

        metric_inst = dummy_metric(multiplier=2.0, suffix="-ok")
        res = metric_inst(pipeline_out)
        assert res == "val=30.0-ok"

        # 2. Direct function-call style
        pipeline_out_direct = dummy_pipeline(df, shift=5)
        assert pipeline_out_direct["value"][0] == 15

        res_direct = dummy_metric(pipeline_out_direct, multiplier=2.0, suffix="-ok")
        assert res_direct == "val=30.0-ok"
    finally:
        for name in ["dummy_pipeline", "dummy_metric"]:
            if name in REGISTRY:
                del REGISTRY[name]
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()


def test_registry_integration():
    from factrix.metrics._registry import REGISTRY

    try:

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            output_shape=OutputShape.PANEL,
            role=SpecRole.PIPELINE,
        )
        def dummy_pipeline(df: pl.DataFrame, shift: int = 1) -> pl.DataFrame:
            return df.with_columns(pl.col("value") + shift)

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            requires={"df_input": dummy_pipeline},
        )
        def dummy_metric(
            df_input: pl.DataFrame, multiplier: float = 2.0, suffix: str = ""
        ) -> str:
            val = df_input["value"][0]
            return f"val={val * multiplier}{suffix}"

        # Verify spec is registered in spec_by_name
        specs = spec_by_name()
        assert "dummy_pipeline" in specs
        assert "dummy_metric" in specs

        pipeline_spec = specs["dummy_pipeline"]
        assert pipeline_spec.name == "dummy_pipeline"
        assert pipeline_spec.cell == _TEST_CELL
    finally:
        for name in ["dummy_pipeline", "dummy_metric"]:
            if name in REGISTRY:
                del REGISTRY[name]
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()


def test_registry_validation_raises():
    from factrix.metrics._registry import REGISTRY

    try:

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
        )
        def dummy_pipeline(df: pl.DataFrame, shift: int = 1) -> pl.DataFrame:
            return df.with_columns(pl.col("value") + shift)

        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
            requires={"non_existent_param": dummy_pipeline},
        )
        def invalid_metric(df: pl.DataFrame):
            pass

        # Trigger registry validation by rebuilding specs
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()
        with pytest.raises(ValueError, match=r"requires.*is not a parameter"):
            _all_specs()
    finally:
        # Clean up the registry to avoid polluting other tests!
        for name in ["dummy_pipeline", "invalid_metric"]:
            if name in REGISTRY:
                del REGISTRY[name]
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()


def test_metric_parameter_ordering_and_reflection():
    from factrix.metrics._registry import REGISTRY

    try:
        # A function with default argument before keyword-only non-default argument.
        # Standard dataclasses.make_dataclass would crash unless ordered non-default first.
        @metric(
            cell=_TEST_CELL,
            aggregation=Aggregation.TS_ONLY,
        )
        def ordered_metric(data: pl.DataFrame, a: int = 10, *, b: int) -> str:
            return f"a={a}, b={b}"

        assert issubclass(ordered_metric, MetricBase)
        assert ordered_metric._first_param_name == "data"
        # The fields should be sorted: non-default ("b") first, then default ("a")
        assert ordered_metric._param_names == ("b", "a")

        # Test instantiation & execution (ordering is checked and bound properly)
        inst = ordered_metric(b=42, a=5)
        data = pl.DataFrame({"value": [1]})
        res = inst(data)
        assert res == "a=5, b=42"

        # Test default value preservation
        inst_default = ordered_metric(b=99)
        res_default = inst_default(data)
        assert res_default == "a=10, b=99"
    finally:
        if "ordered_metric" in REGISTRY:
            del REGISTRY["ordered_metric"]
        from factrix._metric_index import _all_specs

        _all_specs.cache_clear()
