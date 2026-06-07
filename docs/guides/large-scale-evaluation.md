---
title: Large-scale evaluation and memory protection
---

When evaluating hundreds or thousands of factors over large historical datasets, memory protection becomes a key design requirement.

This guide explains how to structure your factor screens using a **user-side batched loop** with Polars LazyFrames. This approach replaces the retired built-in streaming/chunking APIs (`run_metrics_iter`, `run_metrics_chunked`, `evaluate_iter`, and `evaluate_chunked`).

## Design trade-off: why built-in streaming was removed

In earlier versions, `factrix` attempted to manage chunked evaluation and iterator streaming internally. However, this introduced several drawbacks:

- **Complex internal state**: Managing execution state, lazy-to-eager evaluation boundaries, and memory disposal internally added significant complexity to the DAG executor.
- **Redundant memory pressure**: Stashing intermediate chunks in memory before returning them often defeated the purpose of streaming.
- **Loss of control**: Callers could not easily control the file scanning, projection pushdown, or GC behavior of the under-the-hood engine.

By removing the built-in batch/streaming APIs, the library's API surface was simplified. Evaluating large-scale panels is now delegated to a user-side loop. This design lets Polars do what it does best: optimize column reads using projection pushdown, while allowing Python's garbage collector to immediately reclaim memory when a chunk's results fall out of scope.

## The pattern: user-side batched loop

The most memory-efficient way to screen a wide panel (e.g. 500 candidate factor columns) is to:
1. Scan the Parquet file lazily.
2. Chunk the candidate factor columns.
3. In each iteration, select only the baseline columns plus the current chunk's factor columns, collect the subset, and evaluate.
4. Let the collected Polars DataFrame and its evaluation results fall out of scope (or serialize them directly to disk).

Here is the complete executable pattern:

```python
import polars as pl
import factrix as fx
from factrix.metrics import ic_newey_west

# 1. Scan metadata only — nothing is read from disk yet
lazy_panel = pl.scan_parquet("large_panel.parquet")

# 2. Separate the fixed baseline columns from candidate factor columns
baseline_cols = ["date", "asset_id", "forward_return"]
factor_cols = [
    c for c in lazy_panel.collect_schema().names() if c not in baseline_cols
]

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
        metrics={"ic": ic_newey_west()},
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
