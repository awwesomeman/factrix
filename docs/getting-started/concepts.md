---
title: Concepts
---

This page introduces the core mental model and architecture of `factrix`.

## Core workflow

The standard research workflow in `factrix` is divided into three distinct phases:

```
1. inspect_data (Pre-flight) ──▶ 2. evaluate (DAG Evaluation) ──▶ 3. bhy (FDR Screening)
```

### 1. `inspect_data` (Pre-flight)
Before executing computationally intensive metrics, you scan your input data. 
Calling `inspect_data(data)` returns a `DataInspection` object. It detects data properties (number of assets, periods, and sparsity) and evaluates them against the statistical requirements of every registered metric. This partitions the metrics into three tiers:
- **`usable`**: The data satisfies all sample size and structural requirements.
- **`degraded`**: The metric can run, but results may carry a warning due to a smaller sample size.
- **`unusable`**: The sample is too thin to compute the metric (running it will short-circuit to `NaN`).

### 2. `evaluate` (DAG Evaluation)
Once you select your metrics, you run the evaluation. 
`evaluate()` takes your data, a dictionary of metric instances, and a list of factor columns. Under the hood, a **Directed Acyclic Graph (DAG) executor** resolves all metric dependencies, optimizes the execution sequence, and returns a list of `EvaluationResult` objects (one per factor).

### 3. `bhy` (FDR Screening)
If you are screening multiple candidate factors, you must correct for multiple testing to control false discoveries.
Passing the list of `EvaluationResult` objects to `fx.multi_factor.bhy` applies the Benjamini-Hochberg-Yekutieli (BHY) step-up procedure. This controls the False Discovery Rate (FDR) under arbitrary dependency, returning the subset of factors that truly possess predictive edge.

---

## The execution DAG: specs, dependencies, and batching

The DAG executor relies on specific metadata attached to each metric class:

### `MetricSpec`
Every metric declares a `MetricSpec` as its single source of truth (SSOT). The spec defines:
- **`cell`**: The target data scope and density the metric applies to.
- **`aggregation`**: How cross-sectional and time-series reductions compose.
- **`requires`**: The upstream intermediate data dependencies.
- **`batchable`**: Whether the metric can process multiple factors in a single operation.

### `requires` (Dependency Injection)
Metrics do not always compute directly from raw panels. For example, the Information Coefficient (`ic`) statistic does not consume the raw asset returns directly; it requires the daily time-series of ICs. 
The `ic` spec declares `requires={"ic_df": compute_ic}`. The DAG executor detects this, schedules `compute_ic` to run first, and injects its output into the `ic` metric.

### `batchable` (Shared Computations)
Many upstream calculations (like sorting, grouping, or ranking assets to compute ICs or quantiles) are identical across all factor candidates. 
If a spec is marked `batchable=True`, the executor runs it once across all factors in the batch, significantly reducing overhead compared to looping through each factor individually.

---

## Three orthogonal design axes

An evaluation cell is defined by three orthogonal axes:

| Axis | Values | Description |
|------|--------|-------------|
| **FactorScope** | `INDIVIDUAL` / `COMMON` | Does each asset have its own factor value (e.g. P/E ratio), or do all assets share one value (e.g. VIX index)? |
| **FactorDensity** | `DENSE` / `SPARSE` | Is the signal a continuous numeric exposure, or a sparse event trigger (non-events are zero; event magnitude is a real number)? |
| **DataStructure** | `PANEL` / `TIMESERIES` | Derived from the asset count at evaluate-time: `PANEL` for `n_assets >= 2`, and `TIMESERIES` for `n_assets == 1`. |

`SPARSE` is a zero-encoded event contract, not a generic discrete-factor
label. The detector counts explicit `0` values over non-null factor cells:
nulls are missing values, not non-events. If missing upstream rows should
mean "no event", fill them to `0`; if the research question is event-study
shaped, map the event-of-interest into `{0, R}` before using sparse metrics.

A metric's cell declares which `DataStructure` it applies to. A `PANEL`-cell metric needs a cross-section, so on `n_assets == 1` data — `Individual × Continuous` (`ic`, `fm`) or `Common × Continuous` (`common_beta`) — `evaluate` raises `IncompatibleAxisError` (or returns NaN + `structure_mismatch` under `strict=False`). Single-asset dense data uses `predictive_beta` for the direct HAC predictive-regression slope and `directional_hit_rate` for sign prediction. Single-asset sparse data is served by `(*, SPARSE, *)` metrics. Two-column diagnostics (`positive_rate`, `oos_decay`, `ic_trend`) remain standalone `(date, value)` tools, and in `evaluate()` they are layered on panel IC series rather than raw single-asset dense panels.
