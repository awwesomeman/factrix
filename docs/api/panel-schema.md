# Panel schema

Single-source contract for every `factrix` entry point that consumes a
panel. Every dispatch cell registered through `evaluate` /
`run_metrics` floors its `INPUT_SCHEMA` at the same four columns
described here. Per-cell extensions (optional weight / price columns)
are listed under [Optional columns](#optional-columns).

## Four-column contract

| Column | dtype | Semantics |
|---|---|---|
| `date` | `Date` (preferred) or `Datetime` | Observation timestamp. Sorted ascending per asset. Frequency-agnostic — factrix shifts rows, never calendar time. |
| `asset_id` | `Utf8` / `Categorical` | Cross-section identifier. Identical for COMMON-scope factors (`df.group_by("date").agg(pl.col("factor").n_unique() == 1).all()` is `True`). |
| `factor` | `Float64` | The signal value. Continuous: real-valued (z-score, IC-rankable). Sparse: `{0, R}` event trigger — `0` for non-events, arbitrary real magnitude otherwise; expect ≥ 50% zeros. |
| `forward_return` | `Float64` | Look-ahead return over the horizon used at evaluate time. Attach via [`compute_forward_return`](preprocess.md) so the horizon is explicit and aligned with `AnalysisConfig.forward_periods`. |

The minimal panel is therefore long-format
`(date, asset_id, factor, forward_return)`. A 3-row preview:

```python
import polars as pl
from datetime import date

panel = pl.DataFrame({
    "date":           [date(2024, 1, 1), date(2024, 1, 1),
                       date(2024, 1, 2), date(2024, 1, 2),
                       date(2024, 1, 3), date(2024, 1, 3)],
    "asset_id":       ["A", "B", "A", "B", "A", "B"],
    "factor":         [0.12, -0.08, 0.20, 0.04, -0.15, 0.18],
    "forward_return": [0.01,  0.00, 0.02, 0.00, -0.01, 0.03],
})
```

The two synthetic dataset generators emit this layout (plus a `price`
column) ready for `compute_forward_return`:
[`fx.datasets.make_cs_panel`](datasets.md) (cross-sectional) and
`fx.datasets.make_event_panel` (event-study).

## `factor_col=` — non-default signal column name

Panels often arrive with the signal column named something other than
`"factor"` (e.g. `"alpha"`, `"score"`, `"momentum_12_1"`). Pass
`factor_col=` to rename in place at dispatch time without mutating the
caller's frame:

```python
profile = fx.evaluate(panel, cfg, factor_col="alpha")
bundle  = fx.run_metrics(panel, cfg, factor_col="momentum_12_1")
```

Behaviour:

- The column is renamed to `"factor"` internally so every procedure's
  `INPUT_SCHEMA` still sees the canonical schema.
- `bundle.identity = (factor_col, cfg.forward_periods)` — looping
  `factor_col=name` over a wide multi-factor panel is the canonical
  pattern for batch screening; see
  [Batch screening guide](../guides/batch-screening.md).
- Each call repeats the per-date cross-section work, so cost scales as
  `O(n_factors × per_date_cost)` — there is no shared-pass primitive.

Error cases (both raise [`UserInputError`](errors.md)):

| Trigger | Message hint |
|---|---|
| `factor_col` not present on the panel | Lists the actual columns; suggests a fuzzy match. |
| Both `"factor"` and `factor_col` present, values differ | Flags the ambiguity. Drop the unused column before calling. |

## Optional columns

Per-cell extensions activate additional standalone metrics when present
and short-circuit (`NaN` with `reason`) when absent — they never gate the
core procedure.

| Column | Activates | Cell |
|---|---|---|
| `market_cap` (or any name passed as `weight_col=`) | `quantile_spread_vw` value-weighting | Individual × Continuous |
| `price` | `event_around_return`, `mfe_mae_summary`, event-window diagnostics | Individual × Sparse |

## Common errors

Schema-related failures and their fix paths:

| Message | Trigger | Fix |
|---|---|---|
| `factor_col 'X' not in panel columns` | Typo / wrong column name | Check `panel.columns`; pass the actual name to `factor_col=`. |
| `Both 'factor' and 'X' present` | Wide panel still has stale `"factor"` column | `panel.drop("factor")` before calling. |
| `MissingConfigError: evaluate(raw) needs AnalysisConfig` | Called `evaluate(panel)` with no `cfg` | `fx.suggest_config(panel)` recovers a starting cell. |
| `forward_return column missing` | Forgot the preprocess step | `panel = compute_forward_return(raw, forward_periods=h)` before `evaluate`. |

Full error taxonomy and recovery patterns: [Errors](errors.md).

## Preprocess pipeline

The canonical pipeline from raw price/event data to evaluate-ready panel:

```
raw price panel  ──compute_forward_return(h)──▶  (date, asset_id, factor, forward_return)
                                                          │
                                                          ▼
                                                       evaluate / run_metrics / by_slice / ...
```

Pre-attachment helpers live in [`factrix.preprocess`](preprocess.md);
synthetic panels in [`factrix.datasets`](datasets.md). Wide-format
multi-factor inputs are handled by looping `evaluate` / `run_metrics`
with `factor_col=` rather than by reshaping the panel — see the
[Batch screening guide](../guides/batch-screening.md).

## See also

- [`evaluate`](evaluate.md) — single-factor dispatch entry
- [`run_metrics`](run-metrics.md) — descriptive metric bundle
- [`AnalysisConfig`](analysis-config.md) — axis selection
- [Concepts](../getting-started/concepts.md) — three-axis taxonomy and dispatch cells
