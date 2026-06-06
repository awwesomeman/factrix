---
title: Panel vs timeseries
---

!!! abstract "Answers"
    What `PanelMode.PANEL` vs `PanelMode.TIMESERIES` mean, when each is dispatched, and the sample-guard contract for each.
    For the conventions table (column names, alignment), see [Timeseries-mode conventions](../reference/ts-mode-conventions.md).
    For the `mode=` parameter on `evaluate()`, see [`evaluate`](../api/evaluate.md).
    For sample-guard error surfacing (`InsufficientSampleError`, `IncompatibleAxisError`), see [Errors](../api/errors.md).

## Sample guards

Time-series length `n_periods` and asset count `n_assets` are gated **independently** — factrix does not use a combined `n_periods × n_assets` observation count, because per-date statistic variance is driven primarily by `n_assets`, while time-series aggregation power is driven by `n_periods`.

### Two-axis guard structure

| Axis | Hard block | Soft warning | Clean |
|------|-----------|--------------|-------|
| `n_periods` (T) | T < 20 → `InsufficientSampleError` | 20 ≤ T < 30 → `UNRELIABLE_SE_SHORT_PERIODS` | T ≥ 30 |
| `n_assets` (N) | none | N < 10 → `SMALL_CROSS_SECTION_N`; 10 ≤ N < 30 → `BORDERLINE_CROSS_SECTION_N` | N ≥ 30 |

`n_assets` is never hard-blocked because the cross-asset t-test on E[β] is mathematically well-defined for N ≥ 2 — only its statistical power degrades. A hard block would force users to choose between "can't run" and "don't know there's a problem"; the warning provides the result while surfacing the issue.

### Behaviour matrix by factory and N

| Factory | N=1 | N=2..9 | N=10..29 | N≥30 |
|---|---|---|---|---|
| `individual_continuous(IC)` | raises `IncompatibleAxisError` | all dates dropped (MIN_ASSETS_PER_DATE_IC=10) → NaN | normal PANEL | normal PANEL |
| `individual_continuous(FM)` | raises `IncompatibleAxisError` | per-date guard; low df | normal PANEL | normal PANEL |
| `common_continuous` | TIMESERIES single-series β | emits `SMALL_CROSS_SECTION_N` | emits `BORDERLINE_CROSS_SECTION_N` | normal PANEL |
| `individual_sparse` / `common_sparse` | `_SCOPE_COLLAPSED` → TS dummy; `SCOPE_AXIS_COLLAPSED` info | normal PANEL CAAR | normal PANEL CAAR | normal PANEL CAAR |

## Sample-deficiency surfacing by entry point

The same insufficient-sample condition surfaces differently across the three public entry points. This is deliberate — each entry point answers a different caller contract — but the divergence is easy to mistake for inconsistency.

| Entry point | HARD violation | Caller contract |
|---|---|---|
| `evaluate(panel, config)` | raises [`InsufficientSampleError`](../api/errors.md#factrix.InsufficientSampleError) carrying `.actual_periods` / `.required_periods` | Returns a single headline `primary_p` — silently emitting `NaN` would propagate into Benjamini-Hochberg-Yekutieli (BHY) and partial-conjunction FDR with no recoverable signal. |
| Standalone metric callable (e.g. [`quantile_spread`](../api/metrics/quantile.md#factrix.metrics.quantile.quantile_spread)) | returns short-circuit `MetricResult(value=NaN, metadata={"reason", "n_obs", "min_required"})` | Diagnostic-grade output; `MetricResult` is a typed failure carrier so callers can inspect `result.metadata["reason"]` without `try` / `except` flow control. |
| `run_metrics(panel)` batch | metric joins the `MetricsBundle.skipped` map keyed by metric name → reason string; other metrics in the batch continue | Multi-metric diagnostic battery — one metric's local failure must not abort the whole bundle. |

`n_assets` deficiency follows a separate rule that holds across all three entry points: the cross-asset axis **never** hard-blocks, because the t-test on `E[β]` is well-defined for `N ≥ 2` — only statistical power degrades. Small `N` surfaces as [`SMALL_CROSS_SECTION_N`](../reference/warning-codes.md) / [`BORDERLINE_CROSS_SECTION_N`](../reference/warning-codes.md) warnings on the resulting `FactorProfile` / `MetricResult`, regardless of entry point. The two-axis guard table above remains the source of truth for which axis triggers which behaviour.

A separate "why did this metric not run" category — metrics that consume stage-2 frames (`caar`, `fama_macbeth`, `ts_beta`, ...) or per-date series (`hit_rate`, `ic_trend`, `oos_decay`) — is excluded from auto-discovery because the runner cannot synthesise their inputs from the panel alone. These appear in the `skipped` map with a `consumes …; call X then Y(…) directly` reason that points at the explicit composition recipe; they are not a sample-size issue.

## Aggregation order

PANEL procedures split into **cross-section first** (`cs-first` —
`individual_continuous` information coefficient (IC) / FM, sparse CAAR) and **time-series first**
(`ts-first` — `common_continuous`, `common_sparse`). The order determines
small-sample failure modes and the N=1 collapse behaviour: `common_continuous`
at N=1 degenerates to a single-series β test (null: β=0, not E[β]=0), which is
still well-defined; `individual_continuous` at N=1 has no cross-section to
aggregate over, so it raises. The `cs-first` / `ts-first` / `ts-only` /
`static-cs` / `per-event` shorthand is the canonical vocabulary used across
the metric matrix and `list_metrics()` output — see
[Reference § Metric pipelines § Aggregation vocabulary](../reference/metric-pipelines.md#aggregation-vocabulary).

Full per-procedure pseudocode for all 7 registered pipelines lives in [Development § Procedure pipelines](../development/architecture.md#procedure-pipelines).

