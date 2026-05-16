---
title: Efficient data loading for large panels
---

When a factor exploration sweep reaches the 100–1000+ factor scale,
the **data loading step** — not factrix's compute path — is usually
what runs the machine out of memory. This guide shows the recipes
that keep peak RSS bounded, and explains the contract `evaluate()` /
`run_metrics()` follow at the API boundary.

For the column contract itself, see
[Panel schema](../api/panel-schema.md); for the reshape steps that
turn raw inputs into a panel, see [Preparing data](preparing-data.md).
This guide is purely about **getting that panel into memory without
blowing up**.

## The shape of the problem

A typical large-panel workload: 1000 candidate factor columns,
2500 trading dates, 3000 assets. Naïvely loading every factor in
long format:

```python
# Don't do this.
df = pl.read_parquet("factors.parquet")   # ~128 GB at long-format,
                                          # ~32 GB at wide-format
```

…will OOM a 64 GB workstation in either layout. The fix is not a
faster machine — it is **lazy loading with projection / predicate
pushdown**, scoped to the columns and date range you actually need
for the call you are about to make.

## What factrix's API accepts

factrix is polars-native. `evaluate()` and `run_metrics()` accept:

| Input type | Behaviour at the boundary |
|---|---|
| `pl.DataFrame` | passes through unchanged |
| `pl.LazyFrame` | `.collect()` is called immediately by factrix |
| `pd.DataFrame` | **rejected** with a `TypeError` pointing at the conversion paths below |

The `LazyFrame` branch is **convenience sugar at the boundary**, not
end-to-end lazy execution. factrix does not apply projection or
predicate pushdown for you — its internal pipeline is eager because
the per-date `groupby` / rank / correlation / bootstrap operations
need materialised arrays anyway. The win from `LazyFrame` comes
entirely from the work you do **before** handing the frame over.

### pandas users

Two clean entry points, picked by what shape your pandas frame is in.

**Vendor-named OHLCV / price source — use `adapt`** (converts + renames
in one step; downstream you still attach the factor column and call
`compute_forward_return` to assemble the 4-column panel
`evaluate` / `run_metrics` expect):

```python
from factrix import adapt
from factrix.preprocess import compute_forward_return

raw = adapt(
    my_pandas_df, date="trade_date", asset_id="ticker", price="close_adj",
)
raw = raw.with_columns(my_factor_expr.alias("factor"))
panel = compute_forward_return(raw, forward_periods=5)
profile = fx.evaluate(panel, cfg)
```

**Already-canonical 4-column panel — explicit Arrow conversion**:

```python
panel = pl.from_pandas(my_pandas_df)  # has date, asset_id, factor, forward_return
profile = fx.evaluate(panel, cfg)
```

The conversion happens once at the seam between your data pipeline
and factrix; from then on the panel lives in polars. This keeps the
type contract honest (`evaluate` / `run_metrics` are polars-only) and
makes the cost of pandas → polars visible at the call site instead of
hidden in every API call.

### Direct metric calls (`factrix.metrics.*`)

The metric primitives in `factrix.metrics` (`compute_ic`,
`quantile_spread`, `monotonicity`, …) are polars-only by signature
and do not run the input gateway — they trust the caller. When using
them directly (outside `evaluate` / `run_metrics`), convert pandas
upstream with `pl.from_pandas` or `factrix.adapt` first, and
materialise any `LazyFrame` with `.collect()` before passing in.
Passing a non-`pl.DataFrame` will surface as a `polars` error (e.g.
`AttributeError`, `ColumnNotFoundError`) at the first column access
rather than the guiding `TypeError` the top-level entry points
raise.

## Recipe: `scan_parquet` + projection + predicate pushdown

The five-line pattern that turns the load step from "OOM" into
"hundreds of MB":

```python
from datetime import date

import polars as pl
import factrix as fx

required = ["date", "asset_id", "forward_return", "momentum_12_1"]

panel = (
    pl.scan_parquet("factors.parquet")           # 1. lazy scan, no data loaded
    .select(required)                            # 2. projection pushdown
    .filter(pl.col("date").is_between(           # 3. predicate pushdown
        date(2020, 1, 1), date(2024, 12, 31),
    ))
    .collect()                                   # 4. materialise the slice
)

profile = fx.evaluate(panel, cfg, factor_col="momentum_12_1")
```

Why this works:

- `scan_parquet` opens the file but reads nothing.
- `.select(...)` pushes the column list down into the parquet reader,
  so only those columns leave disk.
- `.filter(...)` on a partition / row-group-aligned column lets the
  reader skip entire row groups instead of materialising and filtering
  them.
- The `.collect()` you write is the only `.collect()` in the chain.

You can equally pass the `LazyFrame` directly to factrix and skip the
explicit `.collect()` — factrix collects at the boundary either way:

```python
panel = (
    pl.scan_parquet("factors.parquet")
    .select(required)
    .filter(pl.col("date").is_between(date(2020, 1, 1), date(2024, 12, 31)))
)
profile = fx.evaluate(panel, cfg, factor_col="momentum_12_1")  # collects inside
```

Either form materialises the same in-memory frame.

## Renaming columns inside the lazy chain

`factrix.adapt` preserves the input type, so column renaming composes
inside a `LazyFrame` without breaking the lazy plan:

```python
from factrix import adapt

panel = (
    pl.scan_parquet("vendor.parquet")
    .pipe(adapt, date="trade_date", asset_id="ticker", price="close_adj")
    .select(["date", "asset_id", "price", "momentum_12_1"])
    .collect()
)
```

`adapt(LazyFrame)` returns `LazyFrame`; `adapt(pl.DataFrame)` returns
`pl.DataFrame`; `adapt(pd.DataFrame)` returns `pl.DataFrame` (pandas
has no lazy equivalent).

## Batching across factors

For a 1000-factor sweep, even with projection the union of all factor
columns is too wide to hold simultaneously. Load and evaluate one
batch of factors at a time:

```python
from itertools import batched

all_factors = ["mom_1", "mom_3", "mom_6", ..., "value_1", "value_3"]
results = []

for batch in batched(all_factors, 50):
    panel = (
        pl.scan_parquet("factors.parquet")
        .select(["date", "asset_id", "forward_return", *batch])
        .filter(pl.col("date").is_between(date(2020, 1, 1), date(2024, 12, 31)))
        .collect()
    )
    for col in batch:
        results.append(fx.evaluate(panel, cfg, factor_col=col))
    del panel   # release the thin panel before the next batch
```

Peak RSS is bounded by the **batch panel size**, not the full universe.
For the 1000-factor example above, a batch of 50 keeps peak load to
roughly `50 / 1000` of the naïve scenario.

## What stays out of scope

factrix's optimisation responsibility starts when `panel` enters
`evaluate()` / `run_metrics()` and ends when the result is returned.
Everything upstream — the parquet layout, your row-group partitioning,
whether you stream from S3 or local disk, how often you refresh
cached panels — is your data pipeline's concern. The recipes above
are the canonical patterns; if your storage layer has different
pushdown semantics (DuckDB views, Delta Lake, …), the principle is
the same: project and filter as early as possible, hand factrix a
thin frame.
