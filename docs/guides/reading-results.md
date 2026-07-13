---
title: Reading results
---

Each entry point in `factrix` returns a frozen result dataclass. This page walks through how to read the two main results you will encounter:

- **`EvaluationResult`**: What `evaluate()` returns for each factor.
- **FDR Result Containers (`BhyResult`, `PartialConjunctionResult`, `HierarchicalBhyResult`)**: What screening functions in `fx.multi_factor` return.

---

## `EvaluationResult` — single-factor `evaluate()` result

```python
results = fx.evaluate(
    data,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor"],
)
result = results["factor"]
```

An `EvaluationResult` represents the outcome of evaluating a single factor column over all specified metrics. Read the fields in the order below:

### 1. Identity & context — what was tested

| Field | Type | Notes |
|---|---|---|
| `factor` | `str` | The name of the factor column. |
| `cell` | `tuple[FactorScope, FactorDensity, DataStructure]` | The resolved analysis cell (scope, density, and structure). |
| `forward_periods` | `int` | The forward periods horizon. |
| `context` | `Mapping[str, Any]` | Extra context labels (e.g. `{"region": "US"}`). |

### 2. Sample size and observations

| Field | Type | Notes |
|---|---|---|
| `n_periods` | `int` | Unique non-null dates in the factor column — the time-series depth. |
| `n_pairs` | `int` | Non-null `(date, asset_id)` pairs — the effective cross-sectional coverage. |
| `n_assets` | `int` | Unique assets in the panel (union across dates). |

### 3. Evaluated metrics (`result.metrics`)

The `metrics` attribute is a read-only `Mapping[str, MetricResult]` mapping the user-supplied label to a `MetricResult`.

For a specific metric `key`, `result.metrics[key]` exposes:

- **`value`**: Raw metric value (e.g. mean IC).
- **`p_value`**: Calibrated p-value for the metric's test (when applicable, or `None`).
- **`alternative`**: The tested tail (`two-sided`, `greater`, or `less`), present exactly when `p_value` is present. Never infer the tail from the sign of `stat`.
- **`stat`**: Test statistic (t, z, W, chi2, ...).
- **`n_obs`**: Observations seen by this specific metric's estimator.
- **`warning_codes`**: Advisory warnings attached by the metric (e.g. `FEW_EVENTS`).
- **`metadata`**: Tool-specific context.

For CAAR event studies, distinguish raw events from the effective test sample:
`metadata["total_events"]` is the raw non-zero event-row count,
`metadata["n_event_periods"]` is the number of event dates after same-date
events are collapsed, and `metadata["n_event_periods_sampled"]` is the
non-overlapping event-date sample used for the headline `p_value`.

### 4. Warnings and Execution Plan

- **`warnings`**: Flat list of `Warning` objects. A per-metric warning carries `source == metric_name`; a panel-level warning carries `source is None`.
- **`plan`**: Multi-line topological execution plan showing how the DAG resolved and batched the metrics.

---

## FDR Result Containers

FDR screening functions under `fx.multi_factor` (like `bhy()`) return a dictionary mapping each mainstream metric to a result container (such as `BhyResult`).

### Reading order for BhyResult

| Field | Type | Meaning |
|---|---|---|
| `metric_name` | `str` | Name of the metric driving the screen. |
| `survivors` | `list[EvaluationResult]` | Surviving factor results. |
| `adj_p` | `np.ndarray` | BHY-adjusted p-values index-aligned with `survivors`. |
| `q` | `float` | Nominal FDR target passed (`0 < q < 1`). |
| `expand_over` | `tuple[str, ...]` | Keys used to partition the input into independent step-ups. |
| `n_tests` | `Mapping[tuple, int]` | Family size per bucket. |

---

## Native HTML Display

In Jupyter notebooks, evaluating `EvaluationResult` or any `fx.multi_factor` result container in a cell automatically displays a formatted, interactive HTML table showing metadata, metrics, and warnings.
