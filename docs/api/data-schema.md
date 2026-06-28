---
title: Data schema
---

Single-source contract for every `factrix` entry point that consumes a panel. Every dispatch cell `evaluate` runs floors its input schema at the same four columns described here. Per-cell extensions (optional weight / price columns) are listed under [Optional columns](#optional-columns).

## Four-column contract

| Column | dtype | Semantics |
|---|---|---|
| `date` | `Date` (preferred) or `Datetime` | Observation timestamp. Sorted ascending per asset. Frequency-agnostic — factrix shifts rows, never calendar time. |
| `asset_id` | `Utf8` / `Categorical` | Cross-section identifier. Identical for COMMON-scope factors (`df.group_by("date").agg(pl.col("factor").n_unique() == 1).all()` is `True`). |
| `factor` | `Float64` | The signal value. Dense: real-valued exposure (z-score, IC-rankable). Sparse: zero-encoded `{0, R}` event trigger, where `0` marks non-events. |
| `forward_return` | `Float64` | Look-ahead return over the horizon used at evaluate time. Attach via [`compute_forward_return`](preprocess.md) so the horizon is explicit and aligned with `forward_periods`. |

The minimal panel is therefore long-format `(date, asset_id, factor, forward_return)`. A 3-row preview:

For sparse factors, null factor cells are missing values, not non-events, and
are excluded from sparse-ratio detection. Fill missing values to `0` only when
that is the intended event contract; see
[Sparse and event signals](../guides/preparing-data.md#7-sparse-and-event-signals).

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

The two synthetic dataset generators emit this layout (plus a `price` column) ready for `compute_forward_return`: [`fx.datasets.make_cs_panel`](datasets.md) (cross-sectional) and `fx.datasets.make_event_panel` (event-study).

## Accepted input type: `DataInput`

Every data-consuming entry point annotates its first argument as `DataInput` — an eager `pl.DataFrame` or a `pl.LazyFrame`. A `LazyFrame` is collected internally, so the choice is purely ergonomic; the schema contract above is identical either way.

::: factrix.DataInput
    options:
      show_root_toc_entry: false
      heading_level: 4

---

## `factor_cols=` — signal column names

Panels often arrive with the signal column named something other than `"factor"` (e.g. `"alpha"`, `"score"`, `"momentum_12_1"`). Pass a list of column names in `factor_cols=` to `fx.evaluate` to evaluate them:

```python
from factrix.metrics import ic

results = fx.evaluate(
    panel,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["momentum_12_1"],
    forward_periods=5,
)
```

Behaviour:

- `evaluate` projects each entry in `factor_cols` to the canonical `"factor"` name internally so every metric callable still sees the four-column schema.
- `factor_cols=[...]` accepts a list of column names — IC stage-1 and batch-native primitives share one polars query across the batch.
- Each returned `EvaluationResult` has `result.factor` and `result.forward_periods`.

Error cases (both raise [`UserInputError`][factrix.UserInputError]):

| Trigger | Message hint |
|---|---|
| `factor_cols` not present on the panel | Lists the actual columns; suggests a fuzzy match. |

---

## Optional columns

Per-cell extensions activate additional standalone metrics when present and short-circuit (`NaN` with `reason`) when absent — they never gate the core procedure.

| Column | Activates | Cell |
|---|---|---|
| `market_cap` (or any name passed as `weight_col=`) | `quantile_spread_vw` value-weighting | Individual × Continuous |
| `price` | `event_around_return`, `mfe_mae`, event-window diagnostics | Individual × Sparse |

---

## Common errors

Schema-related failures and their fix paths:

| Message | Trigger | Fix |
|---|---|---|
| `factor_cols 'X' not in panel columns` | Typo / wrong column name | Check `panel.columns`; pass the actual name to `factor_cols=`. |
| `forward_return column missing` | Forgot the preprocess step | `panel = compute_forward_return(raw, forward_periods=h)` before `evaluate`. |

Full error taxonomy and recovery patterns: [Errors](errors.md).

---

## Preprocess pipeline

The canonical pipeline from raw price/event data to evaluate-ready panel:

```
raw price panel  ──compute_forward_return(h)──▶  (date, asset_id, factor, forward_return)
                                                           │
                                                           ▼
                                                        evaluate / by_slice / ...
```

Pre-attachment helpers live in [`factrix.preprocess`](preprocess.md); synthetic panels in [`factrix.datasets`](datasets.md). Wide-format multi-factor inputs are handled by passing the column names through `factor_cols=` on a single `evaluate` call rather than by reshaping the panel.

---

## See also

- [`evaluate`](evaluate.md) — dispatch entry
- [Concepts](../getting-started/concepts.md) — three-axis taxonomy and dispatch cells
