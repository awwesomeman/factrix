---
title: Stock factor evaluation
---

Evaluate whether a per-stock factor (one value per `(date, asset_id)`)
carries cross-sectional return predictability.

Runnable notebook: [`examples/stock_factor_evaluation.ipynb`](https://github.com/awwesomeman/factrix/blob/main/examples/stock_factor_evaluation.ipynb).

## Factor type

This recipe uses `AnalysisConfig.individual_continuous(metric=Metric.IC)`
— axes `(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC)`.

Procedure: per-date Spearman ρ between factor and forward return,
aggregated to a Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t-statistic on `E[IC]`. PANEL mode
(`N ≥ 2`); `N = 1` raises [`ModeAxisError`][factrix.ModeAxisError]
since there is no cross-section to rank within.

Literature: [Grinold (1989)](../reference/bibliography.md);
[Newey & West (1987)](../reference/bibliography.md).

## Use this when

- Factor varies across assets at each date (per-stock signal, e.g.
  momentum, value, quality).
- Cross-section is wide (`N ≥ 30` for clean inference; `10 ≤ N < 30`
  emits `BORDERLINE_CROSS_SECTION_N`).
- Time series is at least 30 periods (`T < 20` is hard-blocked).

## What it tests

Null hypothesis `E[IC] = 0` — the factor has no rank-based predictive
ordering of forward returns across assets, on average across dates.
Standard error is NW HAC over the per-date information coefficient (IC) series.

## Output to read

1. `profile.primary_p` — IC NW HAC p-value. For single-factor
   pre-registered analysis, compare against your nominal `α` directly;
   for N candidate factors, route through `multi_factor.bhy` to
   control false discovery rate (FDR).
2. `profile.stats[StatCode.MEAN]` — sign + magnitude of average IC
   (cell identity is on `profile.config.metric`; the StatCode is
   intentionally cell-agnostic — see #187).
3. `profile.stats[StatCode.T_NW]` — the t-statistic that produced
   the p-value. Compare to ±2 as a rough sanity check.
4. `profile.warnings` — `UNRELIABLE_SE_SHORT_PERIODS` etc. flag
   data-quality risks that should change interpretation regardless of
   `primary_p`.

## 1. Setup

```python
import factrix as fx
from factrix.preprocess import compute_forward_return
```

## 2. Synthesise a cross-sectional panel

`make_cs_panel` produces a canonical panel with a target IC built in.
`ic_target=0.08` is a realistic effect size for a working
single-factor strategy.

```python
raw = fx.datasets.make_cs_panel(
    n_assets=100,
    n_dates=500,
    ic_target=0.08,
    seed=2024,
)
panel = compute_forward_return(raw, forward_periods=5)
print(f"panel shape={panel.shape}  N={panel['asset_id'].n_unique()}")
```

Illustrative output:

```text
panel shape=(49400, 5)  N=100
```

## 3. Evaluate

One factory call, one `evaluate()`. The factory commits to the three
axes; `evaluate()` derives `Mode` from the panel shape and dispatches
to the registered procedure.

```python
cfg = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.IC,
    forward_periods=5,
)
profile = fx.evaluate(panel, cfg)

print(f"primary_p    = {profile.primary_p:.4g}")
print(f"mode         = {profile.mode}")
print(f"ic_mean      = {profile.stats[fx.StatCode.MEAN]:+.4f}")
print(f"ic_t_nw      = {profile.stats[fx.StatCode.T_NW]:+.2f}")
```

Illustrative output:

```text
primary_p    = 2.129e-40
mode         = panel
ic_mean      = +0.0722
ic_t_nw      = +14.60
```

## 4. Inspect the full diagnose dict

`diagnose()` is the structured triage interface — same content the
individual-stat reads above derive from, in one dict for human
inspection or AI agent consumption.

```python
import json

print(json.dumps(profile.diagnose(), indent=2, default=str))
```

Illustrative output:

```json
{
  "identity": {"factor_id": "factor", "forward_periods": 5},
  "context": {},
  "cell": {"scope": "individual", "signal": "continuous", "metric": "ic", "mode": "panel"},
  "n_obs": 494,
  "n_pairs": 49400,
  "n_periods": 494,
  "n_assets": 100,
  "primary_p": 2.13e-40,
  "primary_stat": 14.60,
  "primary_stat_name": "t_nw",
  "warnings": [],
  "info_notes": [],
  "stats": {
    "mean": 0.0722,
    "t_nw": 14.60,
    "p_nw": 2.13e-40,
    "t_hh": 14.38,
    "p_hh": 2.13e-39
  },
  "metadata": {
    "t_nw": {"nw_lags": 5},
    "p_nw": {"nw_lags": 5},
    "t_hh": {"kernel": "rectangular", "variance_clamped": false},
    "p_hh": {"kernel": "rectangular", "variance_clamped": false}
  }
}
```

## 5. Sample-guard edge cases

This recipe runs the happy path. For the full `n_assets` × factory
behaviour matrix (small-N warnings, `N=1` fallbacks, `T<20` hard
block) see [Guides § PANEL vs
TIMESERIES](../guides/panel-timeseries.md). Two notes for this cell
specifically:

- `N < 30` emits `BORDERLINE_CROSS_SECTION_N` /
  `SMALL_CROSS_SECTION_N`.
- `N = 1` raises [`ModeAxisError`][factrix.ModeAxisError] with
  `suggested_fix=common_continuous(...)` — the factor would no
  longer be cross-sectional.
