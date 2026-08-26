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


class TestPositionalKnobsRejected:
    """Metric knobs are keyword-only past the data argument.

    The old positional mapping followed ``_param_names`` order, which is not
    the body's signature order: dataclass field rules re-sort non-default
    fields first and ``forward_periods`` is removed as an injected param.
    Concretely, ``quantile_spread(df, 3)`` — a call that reads as
    ``forward_periods=3`` straight off the signature — silently bound
    ``n_groups=3`` and reported numbers for the wrong configuration.
    """

    @staticmethod
    def _panel():
        import factrix as fx
        from factrix.preprocess import compute_forward_return

        return compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=30, n_dates=120, seed=0),
            forward_periods=5,
        )

    def test_second_positional_raises(self):
        from factrix.metrics.quantile import quantile_spread

        with pytest.raises(TypeError, match="by keyword"):
            quantile_spread(self._panel(), 3)

    def test_scalar_metric_second_positional_raises(self):
        # Scalar-shaped metrics (no DataFrame) go through the same wrapper.
        from factrix.metrics.tradability import net_spread

        with pytest.raises(TypeError, match="by keyword"):
            net_spread(0.10, 0.5)

    def test_keyword_call_is_unaffected(self):
        import warnings

        from factrix.metrics.quantile import quantile_spread

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = quantile_spread(self._panel(), forward_periods=3, n_groups=3)
        assert result["factor"].n_obs is not None

    def test_constructor_form_is_unaffected(self):
        # ``metric(**knobs)`` then ``instance(data)`` — no data in args, so
        # the guard must not fire on the plain constructor path.
        import warnings

        from factrix.metrics.quantile import quantile_spread

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inst = quantile_spread(n_groups=3)
            result = inst(self._panel())
        assert result["factor"].n_obs is not None


class TestStandaloneReadsTheForwardPeriodsStamp:
    """A standalone call must use the panel's stamped overlap horizon.

    ``compute_forward_return`` stamps the horizon it built ``forward_return``
    with and ``evaluate`` injects it at dispatch. Without the stamp read, a
    standalone call silently fell back to the signature default (5), giving a
    different non-overlapping sample — and a several-fold different p-value —
    from the same panel.
    """

    @staticmethod
    def _panel(forward_periods: int):
        import factrix as fx

        return fx.preprocess.compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=40, n_dates=200, seed=0),
            forward_periods=forward_periods,
        )

    def test_standalone_matches_evaluate_on_a_stamped_panel(self):
        import factrix as fx
        from factrix.metrics.quantile import quantile_spread

        # 20 != the signature default of 5, so the two paths diverge unless
        # the standalone call reads the stamp.
        panel = self._panel(20)
        standalone = quantile_spread(panel, n_groups=3)["factor"]
        evaluated = fx.evaluate(
            panel,
            metrics={"q": quantile_spread(n_groups=3)},
            factor_cols=["factor"],
        )["factor"].metrics["q"]

        assert standalone.n_obs == evaluated.n_obs
        assert standalone.value == evaluated.value
        assert standalone.p_value == evaluated.p_value

    def test_explicit_argument_still_wins_on_an_unstamped_panel(self):
        from factrix._data_input import _FORWARD_PERIODS_COL
        from factrix.metrics.quantile import quantile_spread

        panel = self._panel(20).drop(_FORWARD_PERIODS_COL)
        stamped = quantile_spread(self._panel(20), n_groups=3)["factor"]
        explicit = quantile_spread(panel, n_groups=3, forward_periods=20)["factor"]
        assert explicit.n_obs == stamped.n_obs

    def test_unstamped_panel_falls_back_to_the_signature_default(self):
        from factrix._data_input import _FORWARD_PERIODS_COL
        from factrix.metrics.quantile import quantile_spread

        panel = self._panel(20).drop(_FORWARD_PERIODS_COL)
        out = quantile_spread(panel, n_groups=3)["factor"]
        # 200 periods sampled every 5 -> 40 draws, minus the trailing window.
        assert out.n_obs > 30
