---
title: Reading results
---

Each entry point in `factrix` returns a frozen result dataclass. This page walks through how to read the two main results you will encounter:

- **`EvaluationResult`**: What `evaluate()` returns for each factor.
- **FDR Result Containers (`BhyResult`, `PartialConjunctionResult`, `HierarchicalBhyResult`)**: What screening functions in `fx.multi_factor` return.

---

## `EvaluationResult` — single-factor `evaluate()` result

```python
import factrix as fx
from factrix.metrics import ic

raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80)
data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

results = fx.evaluate(
    data,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor"],
)
result = results["factor"]
```

An `EvaluationResult` represents the outcome of evaluating a single factor column over all specified metrics. Read the fields in the order below:

### 1. Identity & sweep knobs — what was tested

| Field | Type | Notes |
|---|---|---|
| `factor` | `str` | The name of the factor column. |
| `cell` | `tuple[FactorScope, FactorDensity, DataStructure]` | The resolved analysis cell (scope, density, and structure). |
| `forward_periods` | `int` | The economic return horizon, in periods. Joins the hypothesis identity. |
| `overlap_periods` | `int` | The evaluation-grid overlap the inference consumes. Equals `forward_periods` on the full grid; on a panel built with `compute_forward_return(..., dates=)` it is the overlap derived from the grid spacing. Bookkeeping — it does not join the hypothesis identity. |
| `params` | `Mapping[str, Hashable]` | Sweep knobs that define *which* hypothesis this is (e.g. `{"timeframe": "1h"}`). Joins the hypothesis identifier. |
| `metadata` | `Mapping[str, Any]` | Bookkeeping labels (e.g. `{"run_id": ...}`) that never join the identifier or partition a family. |

### 2. Sample size and observations

| Field | Type | Notes |
|---|---|---|
| `n_periods` | `int` | Unique non-null dates in the factor column — the time-series depth. |
| `n_pairs` | `int` | Non-null `(date, asset_id)` pairs — the effective cross-sectional coverage. |
| `n_assets` | `int` | Unique assets in the panel (union across dates). |

### 3. Evaluated metrics (`result.metrics`)

The `metrics` attribute is a read-only `Mapping[str, MetricResult]` mapping
the user-supplied label to a `MetricResult`. The full field list — including
the serialization methods `to_frame()` / `to_dict()` — is specified once in
[`MetricResult` key fields](../api/evaluation-results.md#key-fields); this
page gives only the order to read them in.

Read a `MetricResult` in this order:

1. **`is_applicable`** — `False` marks a `strict=False` short-circuit
   placeholder. Stop here: `value`, `p_value` and `stat` are not results.
2. **`reason`** — the stable short-circuit reason behind an
   `is_applicable is False`. It says which metric/input combination was
   unsupported.
3. **the value itself** — `value`, then `p_value` with its `alternative`
   (never infer the tail from the sign of `stat`), then `stat`, then `n_obs`
   with the `n_obs_axis` that makes the count interpretable.

Then check `warning_codes` before quoting the number, and read `metadata`
for the estimator-specific detail (for `caar`, the raw-versus-effective
event counts are set out on the [caar page](../api/metrics/caar.md)).

### 4. Warnings and Execution Plan

- **`warnings`**: Flat list of `Warning` objects. A per-metric warning carries `source == metric_name`; a panel-level warning carries `source is None`.
- **`unexpected_warnings`**: The subset of `warnings` not named in the metrics' `expected_warnings` — the ones to triage first.
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
