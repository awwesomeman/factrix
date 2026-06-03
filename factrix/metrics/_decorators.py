from __future__ import annotations
import inspect
import dataclasses
from typing import Any, Callable, Type

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
from factrix._metric_index import Cell, SampleThreshold
from factrix.metrics._base import MetricBase
from factrix.metrics._registry import register

def metric(
    cell: Cell,
    aggregation: Aggregation,
    test_method: TestMethod,
    se_method: SEMethod,
    *,
    input_shape: InputShape = InputShape.PANEL,
    output_shape: OutputShape = OutputShape.SCALAR,
    role: SpecRole = SpecRole.METRIC,
    requires: dict[str, Any] = None,
    batchable: bool = False,
    sample_threshold: SampleThreshold = None,
) -> Callable[[Callable[..., Any]], Type[MetricBase]]:
    """Decorator to define a Metric class from a function definition.

    Constructs a frozen dataclass inheriting from MetricBase and registers it.
    """
    def decorator(fn: Callable[..., Any]) -> Type[MetricBase]:
        # 1. Inspect the function signature to determine fields (skipping the first argument)
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        first_param_name = params[0].name if params else None

        # Sort fields to put non-default arguments before default arguments
        non_default_fields = []
        default_fields = []

        for param in params[1:]:
            # Ignore *args and **kwargs in signature (if any)
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
            if param.default is not inspect.Parameter.empty:
                default_fields.append((param.name, annotation, param.default))
            else:
                non_default_fields.append((param.name, annotation))

        fields = non_default_fields + default_fields

        # 2. Build the class namespace with metadata ClassVars
        cls_attrs = {
            "cell": cell,
            "aggregation": aggregation,
            "test_method": test_method,
            "se_method": se_method,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "role": role,
            "requires": requires or {},
            "batchable": batchable,
            "sample_threshold": sample_threshold or SampleThreshold(),
            "_impl": fn,
            "_first_param_name": first_param_name,
            "_param_names": tuple(f[0] for f in fields),
            "__module__": fn.__module__,
            "__doc__": fn.__doc__,
        }

        # 3. Create the frozen dataclass dynamically
        cls = dataclasses.make_dataclass(
            cls_name=fn.__name__,
            fields=fields,
            bases=(MetricBase,),
            namespace=cls_attrs,
            frozen=True,
            slots=True,
        )

        # 4. Register the class
        register(cls)

        return cls

    return decorator
