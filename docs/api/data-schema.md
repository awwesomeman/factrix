---
title: Data schema
---

Single-source contract for every `factrix` entry point that consumes a panel. Every dispatch cell `evaluate` runs floors its input schema at the same four columns described here. Per-cell extensions (optional weight / price columns) are listed under [Optional columns](#optional-columns).

## Four-column contract

| Column | dtype | Semantics |
|---|---|---|
| `date` | `Date` or `Datetime` (**required** — a `String` or integer date is rejected) | Observation timestamp. Ordering key only — the horizon is measured in periods of the panel's own distinct-date grid, never in calendar time ([Period grid, not calendar](../development/architecture.md#period-grid-not-calendar)). |
| `asset_id` | `Utf8` / `Categorical` | Cross-section identifier. Identical for COMMON-scope factors (`df.group_by("date").agg((pl.col("factor").n_unique() == 1).alias("is_common"))["is_common"].all()` is `True`). |
| `factor` | numeric (`Int*` / `Float*`) | The signal value. Dense: real-valued exposure (z-score, IC-rankable). Sparse: zero-encoded `{0, R}` event trigger, where `0` marks non-events. |
| `forward_return` | `Float64` | Look-ahead return over the horizon used at evaluate time. Attach via [`compute_forward_return`](preprocess.md) so the horizon is explicit and aligned with `forward_periods`. |

The minimal panel is therefore long-format `(date, asset_id, factor, forward_return)`. Two assets over three periods:

## Ingestion contract

Every entry point that consumes a panel (`evaluate`, `evaluate_horizons`,
`inspect_data`, `by_slice`, `compute_forward_return`) normalises it once at
the boundary. Three rules are enforced, and one rewrite is applied:

| Rule | On violation |
|---|---|
| `date` is `Date` or `Datetime` | `UserInputError` — a `String` date sorts lexicographically and silently reorders any non-ISO format; parse it first (`pl.col("date").str.to_date(fmt)`). |
| `(date, asset_id)` is unique | `UserInputError` — a duplicate makes the "next period" the same date's twin and fabricates a `0.0` forward return. |
| Per-asset period grids agree | `ragged_period_grid` warning — an asset missing periods pairs `t+1` with `t+1+h` on *its own* grid, so its horizon differs from the others'. |
| Non-finite float cells | `NaN` and `±inf` in every **float** column are rewritten to `null` (integer, boolean, `Decimal` and string columns are untouched). A `+inf` `price` therefore becomes a gap rather than a fabricated `-100 %` return; a `NaN` factor cell is a missing value like `null`. |

The normalisation is idempotent and cheap (≈0.15 s on 3 M rows), so
`compute_forward_return` followed by `evaluate` runs it twice without cost.


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
| `market_cap` | `quantile_spread_vw` value-weighting | Individual × Continuous |
| `price` | `event_around_return`, `mfe_mae`, event-window diagnostics | Individual × Sparse |

Through `evaluate` the column must carry its declared name above: the panel is
projected per factor before a metric's keyword arguments are known, so a
`weight_col=` override only applies to a direct `quantile_spread_vw(...)` call.
`inspect_data` reports `quantile_spread_vw` as unusable, with a blocker naming
the column, on a panel that has no `market_cap`.

## Reserved columns

[`compute_forward_return`](preprocess.md) stamps two constant `Int32`
columns on the panel. They are never treated as factor columns, are read by
`evaluate` / `by_slice` / `sample_requirements` / the `slice_period_*` tests
as the single source of truth, and are stripped before dispatch, so they never
reach a metric or `to_frame()`. Do not write them by hand.

| Column | Carries | Surfaces as |
|---|---|---|
| `_forward_periods` | The return horizon — the `forward_periods` the return was built with, in periods of the price grid. Names the hypothesis. | `EvaluationResult.forward_periods`; the `(factor, forward_periods, *params)` identity |
| `_overlap_periods` | The overlap of adjacent observations on the evaluation grid — what inference consumes (HAC bandwidth and df, non-overlapping stride, stride-scaled floors). Equal to the horizon on the full grid; derived from `dates=` on a coarser one ([Evaluating on a coarser grid](preprocess.md#evaluating-on-a-coarser-grid)). | `EvaluationResult.overlap_periods` (bookkeeping, not identity); `metadata["overlap_periods"]` on every metric |

A constant column is the one carrier that survives the ordinary polars
transforms a panel goes through between construction and evaluation
(`with_columns`, joins, `partition_by` in `by_slice`); DataFrame-level
metadata does not. A self-attached `forward_return` panel carries neither
stamp and declares `forward_periods=` (and, on a coarser grid,
`overlap_periods=`) on `evaluate`.

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
