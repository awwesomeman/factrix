import logging
from typing import ClassVar

import factrix as fx
import pytest
from factrix.metrics import ic


def test_dag_logger_plan_and_execution(caplog):
    # Set the level of the factrix.dag logger to DEBUG
    logging.getLogger("factrix.dag").setLevel(logging.DEBUG)

    raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80, seed=0)
    data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    with caplog.at_level(logging.DEBUG, logger="factrix.dag"):
        fx.evaluate(
            data,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
        )

    # Check that topological plan was logged
    assert any(
        "Executing DAG with topological plan" in record.message
        for record in caplog.records
    )
    # Check that batched or individual execution was logged
    assert any(
        "Executing node ic" in record.message or "Batched hit" in record.message
        for record in caplog.records
    )


def test_metric_logger_short_circuit(caplog):
    # Set level to INFO
    logging.getLogger("factrix.metric.ic").setLevel(logging.INFO)

    # Create too thin dataset to force short-circuit (less than MIN_ASSETS_PER_DATE_IC * 5 = 150 dates)
    raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=10, seed=0)
    data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    with caplog.at_level(logging.INFO, logger="factrix.metric.ic"):
        # evaluate will raise UserInputError in strict=True; use strict=False to verify short circuit
        fx.evaluate(
            data,
            metrics={"ic": ic()},
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )

    assert any(
        "Metric ic short-circuited" in record.message for record in caplog.records
    )


def test_metric_logger_exception(caplog):
    # Let's define a metric class that raises an error
    from factrix._axis import (
        Aggregation,
        InputShape,
        OutputShape,
        SEMethod,
        SpecRole,
        TestMethod,
    )
    from factrix.metrics._base import MetricBase

    class dummy_metric(MetricBase):
        cell = fx.metrics.ic.cell
        aggregation = Aggregation.CS_THEN_TS
        test_method = TestMethod.T
        se_method = SEMethod.HAC
        input_shape = InputShape.SERIES
        output_shape = OutputShape.SCALAR
        role = SpecRole.METRIC
        requires: ClassVar[dict] = {}
        batchable = False
        sample_threshold = fx.metrics.ic.sample_threshold

        _first_param_name = "series"
        _param_names = ()

        @classmethod
        def _impl(cls, series):
            raise RuntimeError("forced dummy failure")

    # Set logger to INFO
    logging.getLogger("factrix.metric.dummy_metric").setLevel(logging.INFO)

    with (
        caplog.at_level(logging.INFO, logger="factrix.metric.dummy_metric"),
        pytest.raises(RuntimeError, match="forced dummy failure"),
    ):
        dummy_metric()([1, 2, 3])

    assert any(
        "Metric dummy_metric failed with exception" in record.message
        for record in caplog.records
    )
