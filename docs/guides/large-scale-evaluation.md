---
title: Large-scale evaluation and memory protection
---

When evaluating hundreds or thousands of factors over large historical datasets, memory protection becomes a key design requirement.

This guide explains how to structure your factor screens using a **user-side batched loop** with Polars LazyFrames.

## Design trade-off: why factrix has no built-in batch API

`factrix` exposes no chunked-evaluation or iterator-streaming API. Owning that internally costs more than it buys:

- **Complex internal state**: execution state, lazy-to-eager evaluation boundaries, and memory disposal all have to be tracked inside the DAG executor.
- **Redundant memory pressure**: stashing intermediate chunks in memory before returning them defeats the purpose of streaming.
- **Loss of control**: callers cannot easily steer the file scanning, projection pushdown, or GC behavior of an under-the-hood engine.

Delegating large-scale panels to a user-side loop keeps the API surface small and lets Polars do what it does best: optimize column reads using projection pushdown, while allowing Python's garbage collector to immediately reclaim memory when a chunk's results fall out of scope.

## The pattern: user-side batched loop

The most memory-efficient way to screen a wide panel (e.g. 500 candidate factor columns) is to:
1. Scan the Parquet file lazily.
2. Chunk the candidate factor columns.
3. In each iteration, select only the baseline columns plus the current chunk's factor columns, collect the subset, and evaluate.
4. Let the collected Polars DataFrame and its evaluation results fall out of scope (or serialize them directly to disk).

Here is the complete pattern; `panel_path` is your own wide panel on disk:

```python
import polars as pl
import factrix as fx
from factrix.metrics import ic

# 1. Scan metadata only — nothing is read from disk yet
lazy_panel = pl.scan_parquet(panel_path)

# 2. Separate the fixed baseline columns from candidate factor columns.
# If the parquet was written from a `compute_forward_return()` panel, it
# also carries "price" and two reserved stamp columns — "_forward_periods"
# (the return horizon) and "_overlap_periods" (the evaluation-grid overlap).
# All of them must be excluded here, or they get swept into factor_cols and
# `evaluate()` fails: it strips the stamps internally before dispatch, but
# does not filter the caller's factor_cols.
schema_cols = lazy_panel.collect_schema().names()
reserved_cols = {
    "date",
    "asset_id",
    "price",
    "forward_return",
    "_forward_periods",
    "_overlap_periods",
}
baseline_cols = [c for c in schema_cols if c in reserved_cols]
factor_cols = [c for c in schema_cols if c not in reserved_cols]

# 3. Process candidate factors in chunks
chunk_size = 50
all_results = []

for i in range(0, len(factor_cols), chunk_size):
    chunk_cols = factor_cols[i : i + chunk_size]

    # Projection pushdown: only the baseline + chunk factor columns are read
    chunk_data = (
        lazy_panel
        .select(baseline_cols + chunk_cols)
        .collect()
    )

    # Evaluate the active chunk
    chunk_results = fx.evaluate(
        chunk_data,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=chunk_cols,
        forward_periods=5,
        strict=False,  # Keep inapplicable metrics as NaN with warnings instead of raising
    )

    all_results.extend(chunk_results.values())

    # chunk_data is now free to be garbage collected
```

## Why this keeps memory flat

- **Projection pushdown**: `scan_parquet(...).select(...).collect()` reads only the columns required for the current chunk. Polars avoids loading the rest of the file into memory.
- **Chunked dispatch**: Each `evaluate` call only processes `chunk_size` factors at a time, limiting the peak resident memory (RSS).
- **Garbage collection**: At the end of each iteration, the reference to `chunk_data` is overwritten, freeing its memory back to the OS or system allocator.

## Choosing the chunk size

A larger chunk size (e.g., 100–200 factors) amortizes shared computations, such as ranking or grouping assets, but increases peak memory. A smaller chunk size (e.g., 20–50 factors) minimizes peak memory at the cost of slight loop overhead. 

We recommend targeting a working set per chunk that fits within 20-30% of your available RAM.
