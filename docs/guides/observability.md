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
import factrix as fx
from factrix.metrics import ic, ic_ir

raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80)
data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

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
2. ic [per-factor] requires=compute_ic
3. ic_ir [per-factor] requires=compute_ic
```

This output lets you verify that shared upstream producers (like `compute_ic`) are computed exactly once across all factors before downstream consumers run.

---

## 4. Warnings: one channel, one shape

A `WarningCode` reaches you twice, and both halves come from the same call.

* **The record** — the code lands on `MetricResult.warning_codes` and, at
  result assembly, on `EvaluationResult.warnings` as a `Warning`
  (`code` / `source` / `message` / `expected`). `result.unexpected_warnings`
  is the alert view. The record is kept whatever you declare.
* **The echo** — a `UserWarning` on stderr, so the advisory shows up in a
  notebook or a script run without anyone inspecting the result object.

The two used to disagree: several codes were recorded and never echoed, so a
thin sample was visible only to a reader who went looking. Every code now goes
through one emitter, so if it is on the result it was on stderr — unless you
declared it.

### Message anatomy

Every echo has the same three parts:

```text
<label>: <message> (<code>; declare it in expected_warnings=)
```

* `<label>` — the metric or function that raised it (`ic`, `bmp_z`,
  `compute_forward_return`). This is what you act on.
* `<message>` — the body, carrying the numbers that tripped the threshold.
* `<code>` — the [`WarningCode`](../reference/warning-codes.md) value, and the
  one declaration that silences it.

For example, an 8-asset panel over 90 periods run through
`quantile_spread(n_groups=3)` prints three of them:

```text
compute_spread_series: Median 2 assets per group (n_assets=8, n_groups=3). ... (thin_quantile_groups; declare it in expected_warnings=)
quantile_spread: the inference member tested 17 periods, below the WARN floor of 30; ... (unreliable_se_short_periods; declare it in expected_warnings=)
quantile_spread: the median cross-section holds 8 assets, below MIN_ASSETS_WARN=30; ... (few_assets; declare it in expected_warnings=)
```

### Declaring a regime

`evaluate(..., expected_warnings=("few_assets",))` — and the same keyword on
every metric — marks a code as the study's design. Marked, never dropped: the
`Warning` record stays on the result with `expected=True`, and only the echo
stops.

```python
from factrix.metrics import quantile_spread

results = fx.evaluate(
    data,
    metrics={"quantile_spread": quantile_spread(n_groups=3)},
    factor_cols=["factor"],
    expected_warnings=("few_assets", "unreliable_se_short_periods"),
)
```

Codes attached alongside a NaN short-circuit (`metric_unavailable`,
`upstream_unavailable`) are records only: the metric did not run, the reason
is in `MetricResult.metadata["reason"]`, and the result's repr carries it.

---

## 5. Rich Notebook Formatting (`_repr_html_`)

For interactive analysis in Jupyter notebooks, `factrix` implements native HTML representations (`_repr_html_`) on key return types. When you print these objects as the final statement in a cell, they render as formatted tables.

### `DataInspection`

Calling `fx.inspect_data(data)` returns a `DataInspection` object. In a notebook, it displays:
* **Detected Properties**: Axis classifications (`scope`, `density`, `structure`) alongside sample numerics (`n_assets`, `n_periods`, `n_pairs`, `sparse_ratio`).
* **Axis Reasoning**: Text rationales explaining *why* a particular classification was selected.
* **Metrics Verdict Table**: A detailed list of all registered public metrics, showing their eligibility (`usable` / `degraded` / `unusable`), cell-match requirements, blockers, and warnings.
* **Data-Level Warnings**: Diagnostic warnings (e.g., NW HAC SE unreliable due to short periods).

### `EvaluationResult`

`EvaluationResult` objects also render as styled tables detailing:
* Target factor name and resolved cell type.
* Resolved `forward_periods` and observations (`n_obs`, `n_assets`).
* **Metrics Table**: Evaluated metric values, test statistics, p-values, and warnings.
