from __future__ import annotations
import inspect
import dataclasses
from typing import Any, Callable, ClassVar, Type
from abc import ABC

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
    OutputShape,
    SEMethod,
    SpecRole,
    TestMethod,
)
from factrix._metric_index import Cell, SampleThreshold, MetricSpec
from factrix._results import MetricResult

class MetricMeta(type):
    """Metaclass that intercepts calling the Metric class.

    If the call provides the first parameter (input DataFrame/series) in args
    or kwargs, it instantiates a temporary instance with the remaining kwargs
    and calls it immediately. Otherwise, it performs standard dataclass instantiation.
    """

    def __call__(cls, *args, **kwargs):
        # cls._impl is the original function wrapped by the decorator
        sig = inspect.signature(cls._impl)
        params = list(sig.parameters.values())
        if not params:
            return super().__call__(*args, **kwargs)

        first_param_name = params[0].name

        # Determine if the first parameter is present in the call
        has_first_param = False
        first_arg = None
        if len(args) > 0:
            has_first_param = True
            first_arg = args[0]
            remaining_args = args[1:]
        elif first_param_name in kwargs:
            has_first_param = True
            first_arg = kwargs[first_param_name]
            kwargs = kwargs.copy()
            kwargs.pop(first_param_name)
            remaining_args = args
        else:
            remaining_args = args

        if has_first_param:
            # Instantiate with the remaining fields, then run on the first argument
            instance = cls(*remaining_args, **kwargs)
            return instance(first_arg)
        else:
            # Standard instantiation
            return super().__call__(*args, **kwargs)

class MetricBase(metaclass=MetricMeta):
    """Abstract Base Class for all metrics.

    Provides ClassVar attributes for metadata and builds the MetricSpec dynamically.
    Calling an instance evaluates the underlying metric implementation.
    """

    cell: ClassVar[Cell]
    aggregation: ClassVar[Aggregation]
    test_method: ClassVar[TestMethod]
    se_method: ClassVar[SEMethod]
    input_shape: ClassVar[InputShape]
    output_shape: ClassVar[OutputShape]
    role: ClassVar[SpecRole]
    requires: ClassVar[dict[str, Any]]
    batchable: ClassVar[bool]
    sample_threshold: ClassVar[SampleThreshold]

    _impl: ClassVar[Callable]

    @classmethod
    def spec(cls) -> MetricSpec:
        """Dynamically build and return the MetricSpec for this metric."""
        return MetricSpec(
            name=cls.__name__,
            cell=cls.cell,
            aggregation=cls.aggregation,
            test_method=cls.test_method,
            se_method=cls.se_method,
            input_shape=cls.input_shape,
            output_shape=cls.output_shape,
            role=cls.role,
            requires=cls.requires,
            batchable=cls.batchable,
            sample_threshold=cls.sample_threshold,
        )

    def __call__(self, df: Any) -> MetricResult:
        """Evaluate the metric on the given input DataFrame or series."""
        # Convert self (dataclass fields representing configs) to a dict of kwargs
        kwargs = dataclasses.asdict(self)
        # Call the underlying implementation function (accessed via __class__ to avoid binding)
        return self.__class__._impl(df, **kwargs)
