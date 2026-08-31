---
title: Panel vs timeseries
---

!!! abstract "Answers"
    What `DataStructure.PANEL` vs `DataStructure.TIMESERIES` mean, when each is dispatched, and the sample-guard contract for each.
    For stage-1 regressions inside common-factor panel metrics, see [Common-factor regression conventions](../reference/ts-mode-conventions.md).
    For the `evaluate()` entry point, see [`evaluate`](../api/evaluate.md).
    For sample-guard error surfacing (`InsufficientSampleError`, `IncompatibleAxisError`), see [Errors](../api/errors.md).

## Sample guards

Time-series length `n_periods` and asset count `n_assets` are gated **independently** — `factrix` does not use a combined `n_periods × n_assets` observation count, because per-period statistic variance is driven primarily by `n_assets`, while time-series aggregation power is driven by `n_periods`.

### Two-axis guard structure

| Axis | Hard block | Soft warning | Clean |
|---|---|---|---|
| `n_periods` (T) | T below the metric's own periods floor → `InsufficientSampleError` under `strict=True` | `T < MIN_PERIODS_WARN` (= 30) → `UNRELIABLE_SE_SHORT_PERIODS` | T ≥ 30 |
| `n_assets` | none | `n_assets < 30` → `FEW_ASSETS` (severity scales with `n_assets`) | `n_assets >= 30` |

!!! warning "The periods floor is per metric, and it is checked on the **effective** sample"

    There is no single global `T < 20` rule. `MIN_PERIODS_HARD` (= 20) is the
    floor for the HAC / Newey-West time-series path, where a biased HAC SE is
    the failure mode. A metric that sub-samples to non-overlapping dates is
    gated on `MIN_SERIES_PERIODS_HARD` (= 10) applied to the **post-stride**
    count — the sample the t-test actually runs on — plus a scaled floor of
    `10 x overlap_periods` on the raw dates so the stride has something to
    consume. The two constants are deliberately different: they guard different
    estimators on the same axis, and neither may be read as the other (see the
    naming grammar in `factrix/_types.py`).

    So at `T = 80, h = 6`, `ic` strides 74 usable dates down to 13 effective
    periods, clears the floor of 10, returns a p-value, and tags it
    `UNRELIABLE_SE_SHORT_PERIODS` because 13 < 30. That is the contract, not a
    gap in it. `MetricResult.n_obs` always reports the effective count the
    floor was checked against.

    The binding axis is not always `periods`. A bucketed metric
    (`monotonicity`, `quantile_spread`, `notional_turnover`, `k_spread`) fails
    on the **assets** axis when the cross-section cannot fill its buckets,
    however long the panel — `exc.axis` says which.

`n_assets` is never hard-blocked because the cross-asset t-test on E[β] is mathematically well-defined for `n_assets >= 2` — only its statistical power degrades. A hard block would force users to choose between "can't run" and "don't know there's a problem"; the warning provides the result while surfacing the issue.

`FEW_ASSETS` never changes the estimator. Spread metrics, IC, Fama–MacBeth
and common-beta paths all retain their documented estimator and use the
warning to flag thin ranks, low residual degrees of freedom, or unstable
cross-asset aggregation — see the [shared small-N note](../reference/stat-keys-by-metric.md#shared-small-n-note)
for the measured sizes behind that choice on the spread metrics. Read the
metric metadata and method, not the warning code alone, to identify the
inference path.

### Behaviour matrix by density and `n_assets`

| Density / Scope | `n_assets == 1` | `n_assets = 2..9` | `n_assets = 10..29` | `n_assets >= 30` |
|---|---|---|---|---|
| `INDIVIDUAL` × `DENSE` (IC) | raises `UserInputError` or `IncompatibleAxisError` | runs with `FEW_ASSETS` if pairwise-complete per-period `n_assets` is 2..9; dates with `n_assets < 2` are dropped | normal IC; panel-level thin-`n_assets` warnings may still apply | normal PANEL |
| `INDIVIDUAL` × `DENSE` (FM) | raises `UserInputError` or `IncompatibleAxisError` | per-period guard; low df | normal PANEL | normal PANEL |
| `COMMON` × `DENSE` | raises `IncompatibleAxisError` (no cross-section) | emits `FEW_ASSETS` | emits `FEW_ASSETS` | normal PANEL |
| `INDIVIDUAL` × `SPARSE` / `COMMON` × `SPARSE` | TIMESERIES sparse path; no scope-collapse step | normal PANEL CAAR | normal PANEL CAAR | normal PANEL CAAR |

## Sample-deficiency surfacing

The same insufficient-sample condition surfaces differently depending on `strict` setting and metric call style:

- `evaluate(..., strict=True)` (default): raises [`InsufficientSampleError`](../api/errors.md#factrix.InsufficientSampleError) carrying `.axis` (the binding axis), `.actual` / `.required` (counts on that axis) and `.shortfalls` (one entry per failing metric). A missing input column or config raises `UserInputError` instead — the reason vocabulary splits `insufficient_*` from `no_*`.
- `evaluate(..., strict=False)`: keeps inapplicable metrics as `NaN` values with warnings in the returned `EvaluationResult`.
- Standalone metric callable (e.g. [`quantile_spread`](../api/metrics/quantile.md)): returns a short-circuit `MetricResult(value=NaN, metadata={"reason": ..., "n_obs": ...})`.

## Aggregation order

PANEL procedures split into **cross-section first** (`cs-first` — `individual` density metrics like IC / FM, sparse CAAR) and **time-series first** (`ts-first` — `common` density metrics). The order determines small-sample failure modes. At `n_assets == 1` the PANEL dense metrics raise: `common_continuous` (`common_beta`) has no asset cross-section to aggregate the per-asset βs over, and `individual_continuous` (IC / FM) has no cross-section to rank/regress within — both declare `cell.structure = PANEL`, so `evaluate` raises `IncompatibleAxisError`. Single-asset dense workflows use `predictive_beta` for the direct HAC predictive-regression slope and `directional_hit_rate` for sign prediction. For beta stability, derive pre-declared rolling / expanding `predictive_beta` windows and read them descriptively; do not treat overlapping-window p-values as a multiple-testing family. Single-asset sparse workflows are served by sparse metrics whose cell wildcard allows `TIMESERIES`. Two-column diagnostics such as `positive_rate` / `oos_decay` / `ic_trend` are standalone `(date, value)` tools; in `evaluate()` they layer on panel IC series rather than raw single-asset dense panels.
