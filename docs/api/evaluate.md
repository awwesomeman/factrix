# evaluate

Single-factor evaluation entry point. Routes a
`(raw, AnalysisConfig)` pair to the procedure registered for the
dispatch cell and returns a [`FactorProfile`](factor-profile.md).

## Call shape

```python
import factrix as fl

config = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC)
profile = fl.evaluate(raw_panel, config)
profile.diagnose()
```

`raw_panel` is a long DataFrame with at least `date, asset_id, factor`
plus the columns the chosen cell needs (`forward_return` for most
panels; `price` for event-window metrics). `config` selects the cell
(`Scope × Signal × Metric`); `evaluate` looks up the procedure
registered for that cell, runs the registered metrics, and packages
the outputs as a `FactorProfile`.

Dispatch is **explicit** — there is no auto-fallback when the panel
shape does not match the cell. `N == 1` is the one exception:
`Common × Continuous` auto-routes to the TIMESERIES single-series path
(`profile.mode == "TIMESERIES"`) so single-asset macro factors still
flow through.

For runnable recipes see
[Examples](../examples/index.md).

## Required columns per cell

Every procedure declared in the dispatch registry imposes the same
floor at `INPUT_SCHEMA` — `(date, asset_id, factor, forward_return)`.
Some downstream metrics within a cell consume **optional** columns; if
the optional column is absent, those specific metrics short-circuit
gracefully (returning `NaN` with a `reason`) and the rest of the cell
runs normally.

| Cell | Required | Optional column → enables |
|---|---|---|
| Individual × Continuous (`ic`, `fama_macbeth`) | `date, asset_id, factor, forward_return` | `market_cap` (or any column passed as `weight_col=`) → `quantile_spread_vw` value-weighting |
| Individual × Sparse (event studies) | `date, asset_id, factor, forward_return` | `price` → `event_around_return`, `multi_horizon_hit_rate`, `mfe_mae_summary` (degrade gracefully if absent) |
| Common × Continuous (broadcast macro factor) | `date, asset_id, factor, forward_return` | — |
| Common × Sparse (broadcast event dummy) | `date, asset_id, factor, forward_return` | — |

`forward_return` is treated as part of the input contract rather than
computed inside `evaluate`. Attach it via
[`compute_forward_return`](preprocess.md) before the call so the
horizon (`h`) is explicit in the panel and aligned with
`AnalysisConfig.forward_periods`. The two synthetic dataset
generators (`make_cs_panel`, `make_event_panel`) emit
`(date, asset_id, factor, price)` and require the same preprocessing
step.

::: factrix.evaluate
