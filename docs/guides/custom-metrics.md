---
title: Custom metrics
---

This guide explains how to define, configure, and register custom metrics in `factrix` so that they can be resolved by the DAG executor and evaluated using `fx.evaluate`.

There are two primary ways to create custom metrics:

1. **The `@metric` decorator** (recommended): Converts a Python function into a `MetricBase` subclass and registers it automatically.
2. **The `@metric_spec` decorator + `register()`**: Stamps a `MetricSpec` object onto a function and manually registers it.

---

## 1. Using the `@metric` decorator

The `@metric` decorator is the most common and robust way to define custom metrics. It dynamically wraps your implementation function into a subclass of `MetricBase` and registers it with the global metric registry.

### How it works

When you decorate a function with `@metric`, the decorator:
- Inspects the function signature (ignoring the first argument, which represents the input data frame or series).
- Creates a frozen dataclass whose fields correspond to your function's configuration parameters.
- Registers the class under its function name.
- Returns the class, which behaves like a callable metric instance when called (e.g. `my_metric()`).

### Example

Here is how to implement a custom Information Coefficient (IC) metric that accepts a custom trimming parameter:

```python
import polars as pl
import factrix as fx
from factrix._axis import Aggregation, TestMethod, SEMethod
from factrix._metric_index import cell, SampleThreshold
from factrix import MetricResult
from factrix.metrics import metric

# 1. Define the metric cell and thresholds
_CUSTOM_CELL = cell(
    scope=fx.FactorScope.INDIVIDUAL,
    density=fx.FactorDensity.DENSE,
    structure=fx.DataStructure.PANEL,
)

# 2. Decorate the function
@metric(
    cell=_CUSTOM_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    sample_threshold=SampleThreshold(min_periods=10),
)
def custom_trimmed_ic(
    ic_df: pl.DataFrame,
    trim_ratio: float = 0.05,
) -> MetricResult:
    """Compute a trimmed mean IC significance test."""
    ic_vals = ic_df["ic"].drop_nulls()
    
    # Custom trimming logic
    n = len(ic_vals)
    k = int(n * trim_ratio)
    trimmed = ic_vals.sort()[k : n - k]
    
    mean_val = float(trimmed.mean()) if len(trimmed) > 0 else 0.0
    
    return MetricResult(
        value=mean_val,
        metadata={
            "n_periods": n,
            "trim_ratio": trim_ratio,
        }
    )
```

Now you can evaluate your custom metric:

```python
results = fx.evaluate(
    data,
    metrics={"trimmed_ic": custom_trimmed_ic(trim_ratio=0.1)},
    factor_cols=["factor"],
)
```

---

## 2. Using `@metric_spec` and `register()`

If you prefer a manual, lower-level approach where you retain full control over the function object (rather than wrapping it in a dataclass subclass), you can use `@metric_spec` paired with `factrix.metrics.register`.

### How it works

- `@metric_spec` stamps metadata (a `MetricSpec` object) onto the callable's `__metric_spec__` attribute.
- `factrix.metrics.register()` registers the callable with the toolkit, making it importable and usable within `fx.evaluate`.

### Example

```python
import polars as pl
import factrix as fx
from factrix import MetricSpec, MetricResult, metric_spec
from factrix.metrics import register

# `cell` and the spec enums are not yet on the public surface; import
# from the internal modules until they are re-exported.
from factrix._axis import Aggregation, TestMethod, SEMethod
from factrix._metric_index import cell

# 1. Stamp the spec onto the callable
@metric_spec(
    MetricSpec(
        name="my_custom_stat",
        cell=cell(fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE),
        aggregation=Aggregation.CS_THEN_TS,
        test_method=TestMethod.T,
        se_method=SEMethod.HAC,
    )
)
def my_custom_stat(data: pl.DataFrame) -> MetricResult:
    val = float(data["forward_return"].mean())
    return MetricResult(value=val)

# 2. Register it manually
register(my_custom_stat)
```

---

## Defining the execution shape

When setting up your `MetricSpec` (or passing arguments to `@metric`), you should configure the following parameters to ensure proper DAG routing:

- **`requires`**: A dictionary maps your function's parameter names to the upstream producer callables they depend on (e.g. `requires={"ic_df": compute_ic}`).
- **`input_shape`** / **`output_shape`**: Specifies the shape of the data passed into and out of the metric. Options include `InputShape.PANEL`, `InputShape.SERIES`, or `InputShape.SCALAR`.
- **`batchable`**: If set to `True`, the DAG executor calls the function once for the entire batch of factors (receiving a dictionary of data) instead of looping over factors individually.
