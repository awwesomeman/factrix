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

## The `metrics` mapping

`EvaluationResult.metrics` is a read-only `Mapping[str, MetricResult]` (a `MappingProxyType`) keyed by the caller-supplied metric label, including short-circuit NaN outputs. Access it like any dict — `result.metrics["ic"]` returns the `MetricResult` for `"ic"`, and `keys()` / `values()` / `items()` / `get()` / `in` / `len()` / iteration all work as usual. The mapping is immutable; attempting to assign into it raises `TypeError`.


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

::: factrix.Warning

<hr>

## Fields

A `Warning` object represents a warning emitted during data inspection or metric evaluation:

- **`code`** (`WarningCode`): The enum identifier for the warning type (e.g. `WarningCode.UNRELIABLE_SE_SHORT_PERIODS`).
- **`source`** (`str` | `None`): The name of the metric that generated the warning. If the warning is data-level or cross-metric, `source` is `None`.
- **`message`** (`str`): A human-readable detail message describing the issue.


