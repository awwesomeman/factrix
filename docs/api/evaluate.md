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

::: factrix.evaluate
