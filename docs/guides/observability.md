# Observability

This guide covers the observability features in `factrix` that help you debug execution flow, trace metric applicability, and understand the internal evaluation steps.

---

## 1. Logger Structure

`factrix` uses structured logger namespaces categorized by their layer of responsibility. You can configure individual logger levels or attach handlers depending on what you want to trace.

### Logger Namespaces

All logger namespaces are prefixed with `factrix.` (for example, `factrix` + `.dag`):

| Logger Name (prefixed with `factrix.`) | Level | Purpose / Description |
| :--- | :--- | :--- |
| `evaluation` | `INFO` / `WARNING` | **Orchestration & Decision Layer**: Logs orchestration-level events (e.g. Benjamini-Hochberg-Yekutieli (BHY) adjustments) and raises warning diagnostics when data characteristics might degrade metric performance. |
| `metrics` | `DEBUG` / `WARNING` | **Per-Metric Correction Layer**: Logs internal execution steps (e.g. non-overlapping sampling intervals, Newey-West lag resolution) and warns when a correction produces degenerate/fractional samples. |
| `dag` | `DEBUG` | **DAG Execution Layer**: Logs the topological order of DAG nodes, batched execution hits, and short-circuit propagation when upstream prerequisites fail. |
| `metric.<name>` | `INFO` | **Individual Metric Call Layer**: Logs individual metric failures (e.g. short-circuits due to lack of periods, or raised exceptions) under the specific metric's lowercase registry name (e.g. `metric.ic`). |

### Example Configuration

To enable verbose debugging of the DAG executor and metric calls in a notebook or script:

```python
import logging

# Configure root logger
logging.basicConfig(level=logging.INFO)

# Enable detailed DAG executor tracing
logging.getLogger("factrix" + ".dag").setLevel(logging.DEBUG)
```

---

## 2. DAG Execution Plan (`EvaluationResult.plan`)

When you run `fx.evaluate()`, the toolkit resolves the required metrics and their dependencies through a Directed Acyclic Graph (DAG) executor.

Every `EvaluationResult` exposes the topological execution plan via the `plan` property. This plan lists the steps of execution, showing:
* Step number
* Node ID (spec name and optional configuration key)
* Mode of execution (`[batchable]` vs `[per-factor]`)
* Upstream requirements (`requires=...`)

### Example Plan Output

```python
results = fx.evaluate(
    data,
    metrics={"ic": ic(), "ic_ir": ic_ir()},
    factor_cols=["factor"],
)
print(results["factor"].plan)
```

**Output:**
```text
1. compute_ic [batchable]
2. ic [per-factor] requires=ic_df
3. ic_ir [per-factor] requires=ic_df
```

This output lets you verify that shared upstream producers (like `compute_ic`) are computed exactly once across all factors before downstream consumers run.

---

## 3. Rich Notebook Formatting (`_repr_html_`)

For interactive analysis in Jupyter notebooks, `factrix` implements native HTML representations (`_repr_html_`) on key return types. When you print these objects as the final statement in a cell, they render as formatted tables.

### `DataInspection`

Calling `fx.inspect_data(data)` returns a `DataInspection` object. In a notebook, it displays:
* **Detected Properties**: Axis classifications (`scope`, `density`, `structure`) alongside sample numerics (`n_assets`, `n_periods`, `n_pairs`, `sparse_ratio`).
* **Axis Reasoning**: Text rationales explaining *why* a particular classification was selected.
* **Metrics Verdict Table**: A detailed list of all registered public metrics, showing their eligibility (`usable` vs `unusable`), cell-match requirements, blockers, and warnings.
* **Data-Level Warnings**: Diagnostic warnings (e.g., NW HAC SE unreliable due to short periods).

### `EvaluationResult`

`EvaluationResult` objects also render as styled tables detailing:
* Target factor name and resolved cell type.
* Resolved `forward_periods` and observations (`n_obs`, `n_assets`).
* **Metrics Table**: Evaluated metric values, test statistics, p-values, and warnings.
