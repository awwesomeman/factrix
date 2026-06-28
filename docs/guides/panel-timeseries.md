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
| `n_assets` | none | `n_assets < 30` → `FEW_ASSETS` (severity scales with `n_assets`) | `n_assets >= 30` |

`n_assets` is never hard-blocked because the cross-asset t-test on E[β] is mathematically well-defined for `n_assets >= 2` — only its statistical power degrades. A hard block would force users to choose between "can't run" and "don't know there's a problem"; the warning provides the result while surfacing the issue.

### Behaviour matrix by density and `n_assets`

| Density / Scope | `n_assets == 1` | `n_assets = 2..9` | `n_assets = 10..29` | `n_assets >= 30` |
|---|---|---|---|---|
| `INDIVIDUAL` × `DENSE` (IC) | raises `UserInputError` or `IncompatibleAxisError` | runs with `FEW_ASSETS` if pairwise-complete per-date `n_assets` is 2..9; dates with `n_assets < 2` are dropped | normal IC; panel-level thin-`n_assets` warnings may still apply | normal PANEL |
| `INDIVIDUAL` × `DENSE` (FM) | raises `UserInputError` or `IncompatibleAxisError` | per-date guard; low df | normal PANEL | normal PANEL |
| `COMMON` × `DENSE` | raises `IncompatibleAxisError` (no cross-section) | emits `FEW_ASSETS` | emits `FEW_ASSETS` | normal PANEL |
| `INDIVIDUAL` × `SPARSE` / `COMMON` × `SPARSE` | TIMESERIES sparse path; no scope-collapse step | normal PANEL CAAR | normal PANEL CAAR | normal PANEL CAAR |

## Sample-deficiency surfacing

The same insufficient-sample condition surfaces differently depending on `strict` setting and metric call style:

- `evaluate(..., strict=True)` (default): raises [`InsufficientSampleError`](../api/errors.md#factrix.InsufficientSampleError) carrying `.actual_periods` / `.required_periods`.
- `evaluate(..., strict=False)`: keeps inapplicable metrics as `NaN` values with warnings in the returned `EvaluationResult`.
- Standalone metric callable (e.g. [`quantile_spread`](../api/metrics/quantile.md)): returns a short-circuit `MetricResult(value=NaN, metadata={"reason": ..., "n_obs": ...})`.

## Aggregation order

PANEL procedures split into **cross-section first** (`cs-first` — `individual` density metrics like IC / FM, sparse CAAR) and **time-series first** (`ts-first` — `common` density metrics). The order determines small-sample failure modes. At `n_assets == 1` the PANEL dense metrics raise: `common_continuous` (`ts_beta`) has no asset cross-section to aggregate the per-asset βs over, and `individual_continuous` (IC / FM) has no cross-section to rank/regress within — both declare `cell.structure = PANEL`, so `evaluate` raises `IncompatibleAxisError`. Single-asset dense workflows use `predictive_beta` for the direct HAC predictive-regression slope and `directional_hit_rate` for sign prediction. Single-asset sparse workflows are served by sparse metrics whose cell wildcard allows `TIMESERIES`. Two-column diagnostics such as `hit_rate` / `oos_decay` / `ic_trend` are standalone `(date, value)` tools; in `evaluate()` they layer on panel IC series rather than raw single-asset dense panels.
