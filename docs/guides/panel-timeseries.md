# PANEL vs TIMESERIES

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
| `individual_continuous(IC)` | raises `ModeAxisError` | all dates dropped (MIN_ASSETS_PER_DATE_IC=10) → NaN | normal PANEL | normal PANEL |
| `individual_continuous(FM)` | raises `ModeAxisError` | per-date guard; low df | normal PANEL | normal PANEL |
| `common_continuous` | TIMESERIES single-series β | emits `SMALL_CROSS_SECTION_N` | emits `BORDERLINE_CROSS_SECTION_N` | normal PANEL |
| `individual_sparse` / `common_sparse` | `_SCOPE_COLLAPSED` → TS dummy; `SCOPE_AXIS_COLLAPSED` info | normal PANEL CAAR | normal PANEL CAAR | normal PANEL CAAR |

## Aggregation order

PANEL procedures split into **cross-section first** (`individual_continuous` IC / FM, sparse CAAR) and **time-series first** (`common_continuous`, `common_sparse`). The order determines small-sample failure modes and the N=1 collapse behaviour: `common_continuous` at N=1 degenerates to a single-series β test (null: β=0, not E[β]=0), which is still well-defined; `individual_continuous` at N=1 has no cross-section to aggregate over, so it raises.

Full per-procedure pseudocode for all 7 registered pipelines lives in [Development § Procedure pipelines](../development/architecture.md#procedure-pipelines).

## Introspection

```python
# heuristic config suggestion + risk warnings
result = fl.suggest_config(panel)
print(result.reasoning, result.warnings)

# list all cells and their MIN_PERIODS thresholds
print(fl.describe_analysis_modes())
```
