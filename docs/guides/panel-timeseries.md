---
title: Panel vs timeseries
---

!!! abstract "Answers"
    What `DataStructure.PANEL` vs `DataStructure.TIMESERIES` mean, when each is dispatched, and the sample-guard contract for each.
    For the conventions table (column names, alignment), see [Timeseries-mode conventions](../reference/ts-mode-conventions.md).
    For the `evaluate()` entry point, see [`evaluate`](../api/evaluate.md).
    For sample-guard error surfacing (`InsufficientSampleError`, `IncompatibleAxisError`), see [Errors](../api/errors.md).

## Sample guards

Time-series length `n_periods` and asset count `n_assets` are gated **independently** — `factrix` does not use a combined `n_periods × n_assets` observation count, because per-date statistic variance is driven primarily by `n_assets`, while time-series aggregation power is driven by `n_periods`.

### Two-axis guard structure

| Axis | Hard block | Soft warning | Clean |
|---|---|---|---|
| `n_periods` (T) | T < 20 → `InsufficientSampleError` | 20 ≤ T < 30 → `UNRELIABLE_SE_SHORT_PERIODS` | T ≥ 30 |
| `n_assets` (N) | none | N < 10 → `SMALL_CROSS_SECTION_N`; 10 ≤ N < 30 → `BORDERLINE_CROSS_SECTION_N` | N ≥ 30 |

`n_assets` is never hard-blocked because the cross-asset t-test on E[β] is mathematically well-defined for N ≥ 2 — only its statistical power degrades. A hard block would force users to choose between "can't run" and "don't know there's a problem"; the warning provides the result while surfacing the issue.

### Behaviour matrix by density and N

| Density / Scope | N=1 | N=2..9 | N=10..29 | N≥30 |
|---|---|---|---|---|
| `INDIVIDUAL` × `DENSE` (IC) | raises `UserInputError` or `IncompatibleAxisError` | all dates dropped (MIN_ASSETS_PER_DATE_IC=10) → NaN | normal PANEL | normal PANEL |
| `INDIVIDUAL` × `DENSE` (FM) | raises `UserInputError` or `IncompatibleAxisError` | per-date guard; low df | normal PANEL | normal PANEL |
| `COMMON` × `DENSE` | TIMESERIES single-series β | emits `SMALL_CROSS_SECTION_N` | emits `BORDERLINE_CROSS_SECTION_N` | normal PANEL |
| `INDIVIDUAL` × `SPARSE` / `COMMON` × `SPARSE` | `_SCOPE_COLLAPSED` → TS dummy | normal PANEL CAAR | normal PANEL CAAR | normal PANEL CAAR |

## Sample-deficiency surfacing

The same insufficient-sample condition surfaces differently depending on `strict` setting and metric call style:

- `evaluate(..., strict=True)` (default): raises [`InsufficientSampleError`](../api/errors.md#factrix.InsufficientSampleError) carrying `.actual_periods` / `.required_periods`.
- `evaluate(..., strict=False)`: keeps inapplicable metrics as `NaN` values with warnings in the returned `EvaluationResult`.
- Standalone metric callable (e.g. [`quantile_spread`](../api/metrics/quantile.md)): returns a short-circuit `MetricResult(value=NaN, metadata={"reason": ..., "n_obs": ...})`.

## Aggregation order

PANEL procedures split into **cross-section first** (`cs-first` — `individual` density metrics like IC / FM, sparse CAAR) and **time-series first** (`ts-first` — `common` density metrics). The order determines small-sample failure modes and the N=1 collapse behaviour: `common_continuous` at N=1 degenerates to a single-series β test (null: β=0, not E[β]=0), which is still well-defined; `individual_continuous` at N=1 has no cross-section to aggregate over, so it raises.
