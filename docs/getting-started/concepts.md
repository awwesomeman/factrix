---
title: Concepts
---

This page introduces the user-facing mental model of `factrix`.

## Core workflow

The standard workflow has three phases:

```
1. inspect_data (Pre-flight) ──▶ 2. evaluate (DAG Evaluation) ──▶ 3. bhy (FDR Screening)
```

### 1. `inspect_data` (pre-flight)

`inspect_data(data)` detects asset count, period count, density, and other
properties before evaluation. It groups registered metrics as `usable`,
`degraded`, or `unusable`, so you can identify sample and structure problems
before running a large metric set.

### 2. `evaluate` (evaluation)

`evaluate()` takes the data, a mapping of metric instances, and factor columns.
It validates metric applicability and returns an `EvaluationResult` mapping
keyed by factor column.

### 3. `bhy` (FDR screening)

When you screen multiple candidate factors, pass their results to
`fx.multi_factor.bhy`. The Benjamini-Hochberg-Yekutieli (BHY) procedure
controls the false discovery rate (FDR) under arbitrary dependence and returns
the factors that survive the declared screen. It does not prove that a factor
has economic value or will survive portfolio costs.

---

## Three orthogonal design axes

An evaluation cell is defined by three orthogonal axes:

| Axis | Values | Description |
|------|--------|-------------|
| **FactorScope** | `INDIVIDUAL` / `COMMON` | Does each asset have its own factor value (e.g. P/E ratio), or do all assets share one value (e.g. VIX index)? |
| **FactorDensity** | `DENSE` / `SPARSE` | Is the signal a continuous numeric exposure, or a sparse event trigger (non-events are zero; event magnitude is a real number)? |
| **DataStructure** | `PANEL` / `TIMESERIES` | Derived from the asset count at evaluate-time: `PANEL` for `n_assets >= 2`, and `TIMESERIES` for `n_assets == 1`. |

`SPARSE` is a zero-encoded event contract, not a generic discrete-factor
label. Explicit zero means no event; null means missing. Map an event factor to
`{0, R}` before using sparse metrics.

The data-structure axis determines whether a requested metric has the
cross-section it needs:

| Input | Use |
|---|---|
| Multiple assets, dense individual factor | `ic`, `fm_beta`, and cross-sectional diagnostics |
| Multiple assets, dense common factor | `common_beta` and common-factor diagnostics |
| One asset, dense factor | `predictive_beta` and sign diagnostics |
| Sparse factor, one or many assets | Event metrics whose specification admits the detected structure |

For sample guards, strict-mode behaviour, and the full dispatch matrix, see
[Panel vs timeseries](../guides/panel-timeseries.md). For dependency resolution
and batching internals, see [Architecture](../development/architecture.md).
