---
title: factrix.EvaluationResult
---

::: factrix.EvaluationResult

<hr>

## Serialization Methods

`EvaluationResult` provides two main methods to serialize evaluation outputs:

### `to_frame()`
Converts the metric results into a stable, long-form Polars `pl.DataFrame`. This makes it easy to stack results across multiple factors using `pl.concat([r.to_frame() for r in results])` before writing to disk (e.g., Parquet).

**Schema:**
- `factor` (`str`): The factor column name.
- `n_assets` (`i64`): Total unique assets.
- `metric_name` (`str`): The metric identifier.
- `value` (`f64` | `null`): The calculated metric value (NaN/Inf are normalized to `null`).
- `p_value` (`f64` | `null`): The p-value, if applicable.
- `stat` (`f64` | `null`): The underlying test statistic, if applicable.
- `n_obs` (`i64` | `null`): Sample size seen by the metric.
- `warning_codes` (`list[str]`): List of warnings attached to the metric.

### `to_dict()`
Converts the result into a JSON-friendly nested dictionary. It normalizes floats (e.g., `NaN` and `Inf` to `None`) so that it can be serialized directly using standard `json.dumps` without raising errors.


---
title: factrix.MetricResultGroup
---

::: factrix.MetricResultGroup

<hr>

## Key Concepts

`MetricResultGroup` organizes metric outputs for a single factor evaluation into distinct categories:

- **`applicable`**: A list of names for all metrics that are valid for the dispatched cell.
- **`primary`**: A list of names for metrics whose output drives the primary decision axis (e.g., driving false discovery rate adjustments).
- **`diagnostic`**: A list of names for metrics that provide supplementary or descriptive context.
- **`outputs`**: A dictionary mapping the metric labels to their respective `MetricResult` objects.

It supports dict-like lookup (`group["ic"]` to get the `MetricResult` for `"ic"`) as well as dictionary iteration methods (`keys()`, `values()`, `items()`, `__iter__`, and `__len__`).


---
title: factrix.MetricResult
---

::: factrix.MetricResult

<hr>

## Key Fields

`MetricResult` represents the outcome of a single metric calculation. Its primary fields are:

- **`value`** (`float`): The calculated numeric output of the metric.
- **`p_value`** (`float` | `None`): The statistical p-value.
    > [!IMPORTANT]
    > In v0.14.0, the `p` attribute was renamed to `p_value` to unify p-value naming conventions. There is no transitional alias.
- **`stat`** (`float` | `None`): The test statistic (e.g. t-statistic, z-statistic).
- **`n_obs`** (`int` | `None`): The number of observations used in this specific metric calculation.
- **`metadata`** (`dict`): Underlying dictionary of metric-specific metadata.
- **`warning_codes`** (`tuple[str, ...]`): Advisory warnings attached to the metric.
- **`name`** (`str`): The name of the metric, automatically populated during dispatch.


---
title: factrix.Warning
---

::: factrix.Warning

<hr>

## Fields

A `Warning` object represents a warning emitted during data inspection or metric evaluation:

- **`code`** (`WarningCode`): The enum identifier for the warning type (e.g. `WarningCode.UNRELIABLE_SE_SHORT_PERIODS`).
- **`source`** (`str` | `None`): The name of the metric that generated the warning. If the warning is data-level or cross-metric, `source` is `None`.
- **`message`** (`str`): A human-readable detail message describing the issue.


