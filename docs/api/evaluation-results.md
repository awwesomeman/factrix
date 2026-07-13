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
- `alternative` (`str` | `null`): The tested alternative (`two-sided`, `greater`, or `less`), present exactly when `p_value` is present.
- `stat` (`f64` | `null`): The underlying test statistic, if applicable.
- `n_obs` (`i64` | `null`): Effective sample size the metric's estimator used. `null` only where a single integer count is not meaningful (e.g. a multi-window CAAR series).
- `n_obs_axis` (`str` | `null`): The dimension `n_obs` counts along — `periods` / `events` / `pairs` / `assets`. A bare count is uninterpretable without it (a Fama-MacBeth `n_obs` is `periods`; a pooled-OLS one is `(date, asset)` `pairs`). `null` exactly when `n_obs` is.
- `is_applicable` (`bool`): `false` when `strict=False` returned a short-circuit placeholder for an unsupported metric/input combination.
- `reason` (`str` | `null`): Short-circuit reason when `is_applicable` is `false`.
- `warning_codes` (`list[str]`): List of warnings attached to the metric.

### `to_dict()`
Converts the result into a JSON-friendly nested dictionary. It normalizes floats (e.g., `NaN` and `Inf` to `None`) so that it can be serialized directly using standard `json.dumps` without raising errors.


---

## The `metrics` mapping

`EvaluationResult.metrics` is a read-only `Mapping[str, MetricResult]` (a `MappingProxyType`) keyed by the caller-supplied metric label, including short-circuit NaN outputs. Access it like any dict — `result.metrics["ic"]` returns the `MetricResult` for `"ic"`, and `keys()` / `values()` / `items()` / `get()` / `in` / `len()` / iteration all work as usual. The mapping is immutable; attempting to assign into it raises `TypeError`.

`result.metric("ic")` is a convenience over `result.metrics["ic"]`: the same lookup, but a miss raises `KeyError` listing the available labels (`no metric 'sharpe' on this result; available: ic, ic_ir`) instead of a bare key — handy in interactive sessions where a typo would otherwise fail opaquely.


---

::: factrix.MetricResult

<hr>

## Key Fields

`MetricResult` represents the outcome of a single metric calculation. Its primary fields are:

- **`value`** (`float`): The calculated numeric output of the metric.
- **`p_value`** (`float` | `None`): The calibrated statistical p-value. It must be finite and in `[0, 1]`.
- **`alternative`** (`str` | `None`): The corresponding alternative hypothesis (`two-sided`, `greater`, or `less`). `p_value` and `alternative` must either both be present or both be `None`.
    > [!IMPORTANT]
    > `p_value` is the canonical field for metric p-values.
- **`stat`** (`float` | `None`): The test statistic (e.g. t-statistic, z-statistic).
- **`n_obs`** (`int` | `None`): Effective sample size the estimator used in this specific metric calculation.
- **`n_obs_axis`** (`str` | `None`): Sample dimension `n_obs` counts along — one of `periods` / `events` / `pairs` / `assets`. Stamped by the producer alongside `n_obs`; `None` exactly when `n_obs` is.
- **`is_applicable`** (`bool`): `False` for `strict=False` short-circuit placeholders, so reporting code can filter them without inspecting `metadata["reason"]`.
- **`reason`** (`str` | `None`): Stable short-circuit reason, copied from `metadata["reason"]` when present.
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


