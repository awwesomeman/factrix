"""DAG executor — topo / cycle / dispatch / short-circuit / plan."""

from __future__ import annotations

import math

import factrix as fx
import polars as pl
import pytest
from factrix._axis import (
    Aggregation,
    FactorDensity,
    FactorScope,
    SpecRole,
)
from factrix._codes import WarningCode
from factrix._dag import CycleError, DagExecutor, _Node, _project_factor, _topo_sort
from factrix._metric_index import MetricSpec, cell, spec_by_name
from factrix._results import MetricResult

# ---------------------------------------------------------------------------
# Pure-graph fixtures (no factrix.metrics registry dependency)
# ---------------------------------------------------------------------------


def _make_spec(
    name: str,
    *,
    requires=None,
    batchable: bool = False,
    role: SpecRole = SpecRole.METRIC,
) -> MetricSpec:
    return MetricSpec(
        name=name,
        cell=cell(None, None),
        aggregation=Aggregation.CS_THEN_TS,
        requires=requires or {},
        batchable=batchable,
        role=role,
    )


def _nodes(*specs: MetricSpec) -> list[_Node]:
    """Wrap plain specs as one node each (``node_id == spec.name``).

    The executor takes pre-built ``_Node`` items; these graph tests
    declare plain specs and resolve requires by producer name.
    """
    return [
        _Node(
            spec=spec,
            node_id=spec.name,
            requires_nodes=tuple((k, p.__name__) for k, p in spec.requires.items()),
        )
        for spec in specs
    ]


class TestProjectFactor:
    def test_renames_factor_and_drops_other_factors(self):
        df = pl.DataFrame(
            {
                "date": [1, 2],
                "asset_id": ["a", "a"],
                "forward_return": [0.1, 0.2],
                "alpha": [1.0, 2.0],
                "beta": [3.0, 4.0],
            }
        )
        out = _project_factor(df, "alpha")
        assert out.columns == ["date", "asset_id", "forward_return", "factor"]
        assert out["factor"].to_list() == [1.0, 2.0]

    def test_preserves_price_column_when_present(self):
        # Event metrics need ``price`` to compute price paths; the thin
        # projection must not drop it.
        df = pl.DataFrame(
            {
                "date": [1, 2],
                "asset_id": ["a", "a"],
                "forward_return": [0.1, 0.2],
                "alpha": [1.0, 2.0],
                "price": [100.0, 101.0],
            }
        )
        out = _project_factor(df, "alpha")
        assert "price" in out.columns
        assert out["price"].to_list() == [100.0, 101.0]


class TestTopoSort:
    def test_orders_producers_before_consumers(self):
        def producer():
            return None

        def consumer():
            return None

        p = _make_spec("producer")
        c = _make_spec("consumer", requires={"x": producer})
        producer.__name__ = "producer"
        consumer.__name__ = "consumer"
        ordered = _topo_sort([c, p])
        assert [s.name for s in ordered] == ["producer", "consumer"]

    def test_cycle_raises(self):
        def a():
            return None

        def b():
            return None

        a.__name__ = "a"
        b.__name__ = "b"
        spec_a = _make_spec("a", requires={"y": b})
        spec_b = _make_spec("b", requires={"x": a})
        with pytest.raises(CycleError, match="cycle"):
            _topo_sort([spec_a, spec_b])

    def test_missing_producer_raises(self):
        def producer():
            return None

        producer.__name__ = "producer"
        consumer_spec = _make_spec("consumer", requires={"x": producer})
        with pytest.raises(ValueError, match="close the requires-graph"):
            _topo_sort([consumer_spec])


# ---------------------------------------------------------------------------
# Executor dispatch via local fn_resolver (no registry coupling)
# ---------------------------------------------------------------------------


def _build_panel(n_assets=10, n_dates=30, factor_cols=("alpha",)):
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates)
    panel = fx.preprocess.compute_forward_return(raw, forward_periods=1)
    if "factor" in panel.columns and "alpha" in factor_cols:
        panel = panel.rename({"factor": "alpha"})
    for extra in factor_cols:
        if extra not in panel.columns:
            panel = panel.with_columns(
                pl.col("alpha").alias(extra)
                if "alpha" in panel.columns
                else pl.lit(0.0).alias(extra)
            )
    return panel


class TestBatchablePath:
    def test_batchable_called_once_returns_dict(self):
        call_count = {"n": 0}

        def batch_producer(panel, factor_cols):
            call_count["n"] += 1
            return {c: pl.DataFrame({"x": [1.0, 2.0]}) for c in factor_cols}

        def per_factor_consumer(ic_df):
            return MetricResult(value=float(ic_df["x"].mean()))

        producer_spec = _make_spec(
            "batch_producer", batchable=True, role=SpecRole.PIPELINE
        )
        consumer_spec = _make_spec(
            "per_factor_consumer", requires={"ic_df": batch_producer}
        )
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("a", "b", "c"))

        ex = DagExecutor(
            _nodes(producer_spec, consumer_spec),
            fn_resolver={
                "batch_producer": batch_producer,
                "per_factor_consumer": per_factor_consumer,
            }.__getitem__,
        )
        out = ex.execute(panel, ["a", "b", "c"], **axes)
        assert call_count["n"] == 1
        assert {
            c: out[c].metrics["per_factor_consumer"].value for c in ["a", "b", "c"]
        } == {
            "a": 1.5,
            "b": 1.5,
            "c": 1.5,
        }


class TestPerFactorPath:
    def test_per_factor_no_requires_called_once_per_factor(self):
        called = []

        def panel_consumer(panel):
            called.append(panel.height)
            return MetricResult(value=float(panel.height))

        spec = _make_spec("panel_consumer")
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("a", "b"))
        ex = DagExecutor(
            _nodes(spec), fn_resolver={"panel_consumer": panel_consumer}.__getitem__
        )
        out = ex.execute(panel, ["a", "b"], **axes)
        assert len(called) == 2
        assert out["a"].metrics["panel_consumer"].value > 0


class TestStage1Share:
    def test_two_consumers_share_one_producer_call(self):
        producer_calls = {"n": 0}

        def producer(panel, factor_cols):
            producer_calls["n"] += 1
            return {c: 42.0 for c in factor_cols}

        def consumer_a(ic_df):
            return MetricResult(value=ic_df + 1.0)

        def consumer_b(ic_df):
            return MetricResult(value=ic_df + 2.0)

        producer_spec = _make_spec("producer", batchable=True, role=SpecRole.PIPELINE)
        a_spec = _make_spec("consumer_a", requires={"ic_df": producer})
        b_spec = _make_spec("consumer_b", requires={"ic_df": producer})

        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("x",))
        ex = DagExecutor(
            _nodes(producer_spec, a_spec, b_spec),
            fn_resolver={
                "producer": producer,
                "consumer_a": consumer_a,
                "consumer_b": consumer_b,
            }.__getitem__,
        )
        out = ex.execute(panel, ["x"], **axes)
        assert producer_calls["n"] == 1
        assert out["x"].metrics["consumer_a"].value == 43.0
        assert out["x"].metrics["consumer_b"].value == 44.0


class TestByValueNodes:
    def test_two_configs_one_spec_run_as_distinct_nodes(self):
        # #494: the same callable under two node_ids (different config) runs
        # twice and each result keys by its node_id.
        calls = []

        def per_factor(panel, n=0):
            calls.append(n)
            return MetricResult(value=float(n))

        spec = _make_spec("per_factor")
        nodes = [
            _Node(spec=spec, node_id="per_factor", requires_nodes=()),
            _Node(spec=spec, node_id="per_factor#1", requires_nodes=()),
        ]
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("x",))
        ex = DagExecutor(nodes, fn_resolver={"per_factor": per_factor}.__getitem__)
        out = ex.execute(
            panel,
            ["x"],
            kwargs_by_metric={"per_factor": {"n": 5}, "per_factor#1": {"n": 9}},
            **axes,
        )
        assert out["x"].metrics["per_factor"].value == 5.0
        assert out["x"].metrics["per_factor#1"].value == 9.0
        assert sorted(calls) == [5, 9]

    def test_list_params_dedup_without_unhashable_error(self):
        # List-valued params (e.g. ``event_around_return(offsets=[...])`` must
        # not crash node interning; equal lists dedup to one node and the
        # stored kwargs keep the original list.
        spec = spec_by_name()["event_around_return"]
        nodes, label_to_node, node_kwargs = fx._build_nodes(
            {"a": spec, "b": spec},
            {"a": {"offsets": [-1, 0, 1]}, "b": {"offsets": [-1, 0, 1]}},
        )
        assert label_to_node["a"] == label_to_node["b"]
        assert len(nodes) == 1
        assert node_kwargs[label_to_node["a"]] == {"offsets": [-1, 0, 1]}

    def test_differing_list_params_run_as_distinct_nodes(self):
        spec = spec_by_name()["event_around_return"]
        nodes, label_to_node, _ = fx._build_nodes(
            {"a": spec, "b": spec},
            {"a": {"offsets": [-1, 0, 1]}, "b": {"offsets": [0, 1, 2]}},
        )
        assert label_to_node["a"] != label_to_node["b"]
        assert len(nodes) == 2


class TestWarningCodeLift:
    def test_typed_warning_codes_lift_to_warning_records(self):
        # #516: a metric's typed MetricResult.warning_codes is lifted into
        # per-source Warning records and surfaces on to_frame().
        def flagged(panel):
            return MetricResult(
                value=1.0, warning_codes=(WarningCode.FEW_EVENTS.value,)
            )

        spec = _make_spec("flagged")
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("x",))
        ex = DagExecutor(_nodes(spec), fn_resolver={"flagged": flagged}.__getitem__)
        er = ex.execute(panel, ["x"], **axes)["x"]

        assert (WarningCode.FEW_EVENTS, "flagged") in [
            (w.code, w.source) for w in er.warnings
        ]
        assert (WarningCode.FEW_EVENTS, WarningCode.FEW_EVENTS.description) in [
            (w.code, w.message) for w in er.warnings
        ]
        row = (
            er.to_frame().filter(pl.col("metric_name") == "flagged").row(0, named=True)
        )
        assert row["warning_codes"] == [WarningCode.FEW_EVENTS.value]


class TestShortCircuitPropagation:
    def test_nan_upstream_skips_consumer(self):
        consumer_called = {"n": 0}

        def producer(panel, factor_cols):
            return {
                c: MetricResult(
                    value=float("nan"),
                    metadata={"reason": "insufficient_sample"},
                )
                for c in factor_cols
            }

        def consumer(ic_df):
            consumer_called["n"] += 1
            return MetricResult(value=1.0)

        producer_spec = _make_spec("producer", batchable=True)
        consumer_spec = _make_spec("consumer", requires={"ic_df": producer})
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("x",))
        ex = DagExecutor(
            _nodes(producer_spec, consumer_spec),
            fn_resolver={"producer": producer, "consumer": consumer}.__getitem__,
        )
        out = ex.execute(panel, ["x"], **axes)
        assert consumer_called["n"] == 0
        downstream = out["x"].metrics["consumer"]
        assert math.isnan(downstream.value)
        assert downstream.metadata["reason"] == "upstream_unavailable"
        assert downstream.metadata["upstream"] == "producer"
        assert downstream.metadata["upstream_reason"] == "insufficient_sample"
        assert downstream.metadata["consumer_param"] == "ic_df"
        warning_codes = [w.code.value for w in out["x"].warnings]
        assert "upstream_unavailable" in warning_codes

    def test_self_short_circuit_is_metric_unavailable(self):
        # A root metric that short-circuits on its OWN precondition (missing
        # column / config / sample) is a root failure, not a dependency
        # failure: it must surface METRIC_UNAVAILABLE, not UPSTREAM_UNAVAILABLE.
        def standalone(panel):
            return MetricResult(
                value=float("nan"),
                metadata={"reason": "no_weight_column"},
            )

        spec = _make_spec("standalone")
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 1,
        }
        panel = _build_panel(factor_cols=("x",))
        ex = DagExecutor(
            _nodes(spec), fn_resolver={"standalone": standalone}.__getitem__
        )
        out = ex.execute(panel, ["x"], **axes)
        warnings = [(w.code, w.source, w.message) for w in out["x"].warnings]
        assert (WarningCode.METRIC_UNAVAILABLE, "standalone", "no_weight_column") in (
            warnings
        )
        codes = [w.code for w in out["x"].warnings]
        assert WarningCode.UPSTREAM_UNAVAILABLE not in codes


class TestPlanString:
    def test_plan_lists_role_and_requires(self):
        def producer(panel, factor_cols):
            return {c: 1.0 for c in factor_cols}

        def consumer_a(ic_df):
            return MetricResult(value=ic_df)

        def consumer_b(ic_df):
            return MetricResult(value=ic_df)

        producer_spec = _make_spec("producer", batchable=True, role=SpecRole.PIPELINE)
        a_spec = _make_spec("consumer_a", requires={"ic_df": producer})
        b_spec = _make_spec("consumer_b", requires={"ic_df": producer})
        ex = DagExecutor(
            _nodes(producer_spec, a_spec, b_spec),
            fn_resolver={
                "producer": producer,
                "consumer_a": consumer_a,
                "consumer_b": consumer_b,
            }.__getitem__,
        )
        assert ex.plan == (
            "1. producer [batchable]\n"
            "2. consumer_a [per-factor] requires=producer\n"
            "3. consumer_b [per-factor] requires=producer"
        )


# ---------------------------------------------------------------------------
# Registry-load validation (factrix._metric_index._validate_requires)
# ---------------------------------------------------------------------------


class TestRequiresValidation:
    def test_existing_registry_loads(self):
        # Smoke: real factrix.metrics.* tuples pass validation.
        names = spec_by_name()
        assert "ic" in names
        assert "compute_ic" in names

    def test_unknown_key_raises(self):
        from types import SimpleNamespace

        from factrix._metric_index import _validate_requires

        def producer():
            return None

        def consumer(real_key):
            return None

        producer.__module__ = "factrix.metrics.ic"
        consumer.__module__ = "factrix.metrics.ic"
        spec = _make_spec("consumer", requires={"typoed_key": producer})
        with pytest.raises(ValueError, match="not a parameter"):
            _validate_requires("fake", spec, SimpleNamespace(consumer=consumer))

    def test_non_callable_producer_raises(self):
        from types import SimpleNamespace

        from factrix._metric_index import _validate_requires

        def consumer(x):
            return None

        consumer.__module__ = "factrix.metrics.ic"
        spec = _make_spec("consumer", requires={"x": "not_a_callable"})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="not callable"):
            _validate_requires("fake", spec, SimpleNamespace(consumer=consumer))

    def test_producer_without_metric_spec_raises(self):
        from types import SimpleNamespace

        from factrix._metric_index import _validate_requires

        def orphan_producer():
            return None

        def consumer(x):
            return None

        orphan_producer.__module__ = "factrix._codes"
        consumer.__module__ = "factrix.metrics.ic"
        spec = _make_spec("consumer", requires={"x": orphan_producer})
        with pytest.raises(ValueError, match="not a registered @metric class"):
            _validate_requires("fake", spec, SimpleNamespace(consumer=consumer))


# ---------------------------------------------------------------------------
# End-to-end against real factrix.metrics.ic specs
# ---------------------------------------------------------------------------


class TestEndToEndIcCell:
    def test_real_ic_specs_dispatch_through_executor(self):
        by_name = spec_by_name()
        specs = [by_name["compute_ic"], by_name["ic"], by_name["ic_ir"]]
        axes = {
            "scope": FactorScope.INDIVIDUAL,
            "density": FactorDensity.DENSE,
            "forward_periods": 5,
        }
        raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        ex = DagExecutor(_nodes(*specs))
        out = ex.execute(panel, ["factor"], **axes)
        r = out["factor"]
        assert isinstance(r, fx.EvaluationResult)
        assert r.factor == "factor"
        assert "ic" in r.metrics
        assert "ic_ir" in r.metrics
        assert r.plan.startswith("1. compute_ic [batchable]")
        assert r.metrics["ic"].name == "ic"
