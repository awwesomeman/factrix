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

## Choosing an entry point

For a 1000-factor sweep, even with projection the union of all factor
columns is too wide to hold simultaneously. factrix exposes three
entry points; pick by factor count, available RAM, and whether you
need each factor's result the moment it is ready:

| Entry point | Use when | Peak RSS bound |
|---|---|---|
| `run_metrics` (or `evaluate`) | All factors fit comfortably (~one batch of `chunk_size` factors at panel-width) | full `factor_cols` panel + per-factor query intermediates |
| `run_metrics_chunked` | Factor count or panel width would OOM at full breadth | one `chunk_size`-wide slice at a time |
| `run_metrics_iter` | Streaming-flow ergonomic: write to a sink / update progress / break early without paying the full-batch latency | full panel (no chunking) but per-factor consumer pass |

Decision rule of thumb:

- `n_factors × panel_height × 8 B × ~4` (per-factor overhead) fits in
  ~25% of available RAM → `run_metrics` directly.
- Otherwise → `run_metrics_chunked` (auto-sizes `chunk_size` against
  `psutil.virtual_memory().available` when left as `None`).
- Need first-factor latency / live sink writes → `run_metrics_iter`
  (no chunk boundary, but full-batch RSS — combine with
  `run_metrics_chunked` if you also need the memory bound).

## Recipe: `scan_parquet` + `run_metrics_chunked` + streaming sink

The canonical pattern for a 1000-factor screen on a single machine:

```python
import polars as pl
import factrix as fx

all_factors = [f"alpha_{i}" for i in range(1000)]

lazy_panel = (
    pl.scan_parquet("factors.parquet")
    .select(["date", "asset_id", "forward_return", *all_factors])
)

with sink.open("metrics.parquet") as out:
    for bundles in fx.run_metrics_chunked(
        lazy_panel, cfg, factor_cols=all_factors,
    ):
        for fid, bundle in bundles.items():
            out.write(fid, bundle.to_frame())
```

What this buys over a hand-rolled loop:

- **Projection pushdown per chunk**: passing the `LazyFrame` lets
  `run_metrics_chunked` call `panel.select([base_cols + chunk]).collect()`
  on each iteration, so only the chunk's factor columns leave disk
  even though the lazy plan names all 1000.
- **`chunk_size=None` auto-sizes** against
  `psutil.virtual_memory().available` (requires `psutil`; install via
  `pip install psutil` or `pip install 'factrix[bench]'`). Override
  with an explicit integer when the downstream sink has its own
  batching cadence.
- **Base columns and `base_cols` defaults are wired in** (`date`,
  `asset_id`, `forward_return`) — no per-call risk of dropping the
  forward-return column from the projection list. Override `base_cols`
  only when a metric needs an extra column (e.g. `weight_col` for
  `quantile_spread_vw`).

## Picking `chunk_size`

`chunk_size=None` (default) is the right answer on most machines.
Override when you have a reason: the sink batches at a different
cadence, you are co-tenanting with another memory-hungry process, or
you want deterministic chunk boundaries for testing.

When sizing manually, the relationship the auto-sizer encodes is:

```
peak_rss_per_chunk ≈ chunk_size × n_rows × 8 B × 4
```

The `× 4` is empirical overhead factor — covers panel materialise plus
the `_rank__<f>` intermediates `compute_ic` adds per factor. Tracks
observed peak RSS within ±20% across the `small` / `large` benchmark
presets (see [`bench/baselines/`](https://github.com/awwesomeman/factrix/tree/main/bench/baselines)).

Reference: the `small` preset (cross-sectional, ~2500 dates ×
500 assets) reaches roughly **3 GB peak RSS at `chunk_size=100`**
(`n_rows × 8 × 4 × 100 ≈ 4 GB`, within the ±20% band). Linear in
`chunk_size`, so:

| Budget (peak RSS) | Approx `chunk_size` on the `small` preset |
|---:|---:|
| 1 GB | 30 |
| 4 GB | 100 |
| 16 GB | 400 |
| 32 GB | 800 |

For your own panel, multiply by `small_n_rows / your_n_rows` — wider
or longer panels need smaller chunks at the same RSS budget. The
auto-sizer applies this proportionality directly.

## Streaming with `run_metrics_iter`

When the friction is **per-factor latency** rather than peak RSS —
you want to write each factor's bundle the moment it lands, drive a
progress bar, or short-circuit on the first factor that meets a
threshold — `run_metrics_iter` yields `(factor_id, bundle)` pairs as
each factor's consumer pass finishes:

```python
for fid, bundle in fx.run_metrics_iter(panel, cfg, factor_cols=cols):
    sink.write(fid, bundle.to_frame())
    if bundle["ic"].headline_value > threshold:
        break
```

Cross-factor work (IC stage-1, batch-native primitives) is still
amortised once before the first yield. The trade-off is full-batch
RSS — no chunking. Combine with `run_metrics_chunked` when you need
both bounds:

```python
for bundles in fx.run_metrics_chunked(lazy_panel, cfg, factor_cols=all_factors):
    for fid, bundle in bundles.items():     # already in-order within chunk
        sink.write(fid, bundle.to_frame())
```

The chunked path's inner `dict` is already insertion-ordered by
`factor_cols`, so iterating its items inside the outer loop gives the
same per-factor streaming shape as `run_metrics_iter`, with the
chunk-level RSS bound layered on top.

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
