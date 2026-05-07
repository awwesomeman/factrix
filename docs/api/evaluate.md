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

## `factor_col=` — non-default signal column name

Panels often arrive with the signal column named something other than
`"factor"` — e.g. `"alpha"`, `"score"`, or a domain-specific label.
Pass `factor_col=` to evaluate without renaming first:

```python
profile = fl.evaluate(panel, config, factor_col="alpha")
```

Internally the column is renamed to `"factor"` before dispatch so the
procedure's `INPUT_SCHEMA` still sees the canonical schema. Two error
cases:

- `factor_col` not present on the panel → `ValueError` listing the
  panel's actual columns.
- Both `"factor"` and `factor_col` present and they differ → `ValueError`
  flagging the ambiguity. Drop the unused column before calling.

!!! warning "Single-factor convenience only"
    `factor_col=` is for panels with one signal whose column happens
    to be named differently. **Do not** use it to evaluate multiple
    signals on the same panel via a comprehension:

    ```python
    # Anti-pattern: re-pays per-date cross-section overhead per signal.
    results = {f: fl.evaluate(panel, config, factor_col=f)
               for f in factor_names}
    ```

    Each `evaluate` call repeats the per-date cross-section work
    (sort / group-by / rank / HHI), so this scales as
    `O(n_factors × per_date_cost)`. For multi-signal panels use
    [`factrix.multi_factor`](multi-factor.md) — it shares the
    cross-section cost across signals and produces one
    `FactorProfile` per factor.

::: factrix.evaluate
