# evaluate

Single-factor evaluation entry point. Routes a
`(raw, AnalysisConfig)` pair to the procedure registered for the
dispatch cell and returns a [`FactorProfile`](factor-profile.md).

## Call shape

```python
import factrix as fx

config = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
profile = fx.evaluate(raw_panel, config)
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

!!! tip "`profile.mode` — PANEL vs TIMESERIES at a glance"
    | `profile.mode` | When | Inference |
    |---|---|---|
    | `"PANEL"` | `N ≥ 2` cross-sectional / event cells | per-date statistic → time-series mean with NW HAC |
    | `"TIMESERIES"` | `Common × Continuous` with `N == 1` | single-series OLS with plain SE; HAC only on stage-2 aggregation |

    For the full conventions table (column names, alignment, stat keys), see [TIMESERIES-mode conventions](../reference/ts-mode-conventions.md). For when each Mode is dispatched and the sample-guard contract, see [PANEL vs TIMESERIES](../guides/panel-timeseries.md).

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
| Individual × Sparse (event studies) | `date, asset_id, factor, forward_return` | `price` → `event_around_return`, `mfe_mae_summary` (degrade gracefully if absent) |
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
profile = fx.evaluate(panel, config, factor_col="alpha")
```

Internally the column is renamed to `"factor"` before dispatch so the
procedure's `INPUT_SCHEMA` still sees the canonical schema. Two error
cases:

- `factor_col` not present on the panel → `ValueError` listing the
  panel's actual columns.
- Both `"factor"` and `factor_col` present and they differ → `ValueError`
  flagging the ambiguity. Drop the unused column before calling.

For wide multi-factor panels, looping `evaluate` with different
`factor_col=` values per candidate is the canonical pattern; the
[batch screening guide](../guides/batch-screening.md) walks through it
end-to-end with the BHY FDR step. Each `evaluate` call repeats the
per-date cross-section work (sort / group-by / rank / HHI) on its own,
so the cost scales as `O(n_factors × per_date_cost)` — there is no
shared-pass primitive in factrix today; that cost is intrinsic to
producing one `FactorProfile` per signal.
[`factrix.multi_factor.bhy`](multi-factor.md) operates on the
resulting profile list for FDR control; it does **not** reduce the
per-signal evaluation cost.

::: factrix.evaluate
