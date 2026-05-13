---
title: factrix.run_metrics
---

::: factrix.run_metrics

Cell-level descriptive batch runner — the descriptive twin of
[`evaluate`](evaluate.md). Where `evaluate` runs a cell's primary
inferential procedure and returns a [`FactorProfile`](factor-profile.md),
`run_metrics` fans out across the cell's standalone descriptive metrics
in `factrix.metrics.*` and returns a [`MetricsBundle`](#metricsbundle).

The two paths share the `(panel, cfg)` entry contract; their result
types are disjoint by design.

> **Input contract** — the panel must satisfy the four-column floor
> documented in [Panel schema](panel-schema.md).

| Function | Returns | Use when |
|---|---|---|
| [`evaluate`](evaluate.md) | `FactorProfile` (primary p, drives FDR) | you want the single inferential decision for the cell |
| `run_metrics` | `MetricsBundle` (cell's descriptive surface) | you want every standalone metric the cell exposes for plotting / dashboards / cross-factor comparison |

Both can be called on the same `(panel, cfg)`; neither is a
prerequisite for the other.

## Call shape

```python
import factrix as fx

cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
bundle = fx.run_metrics(panel, cfg, factor_col="momentum_12_1")

bundle["ic"].value             # dict-style access
bundle.identity                # ("momentum_12_1", 5) — (factor_id, fwd)
bundle.to_frame()              # long-form pl.DataFrame
dict(bundle.skipped)           # metrics that could not auto-run
```

`panel` follows the same canonical schema as `evaluate`
(`date, asset_id, factor, forward_return`); `factor_col` renames an
alternate signal column to `"factor"` internally before dispatch.
`bundle.identity = (factor_col, cfg.forward_periods)` — when looping
over candidate signals, pass `factor_col=name` for each so
`bundle.identity` stays unique across the sweep (otherwise every
bundle reports `("factor", h)` and concatenated `to_frame()` outputs
collide).

## What auto-discover runs

`metrics=None` (default) runs every metric `list_metrics(cfg.scope,
cfg.signal)` exposes for the cell, after three filters:

1. `input_kind == "panel"` — drops scalar-input utilities
   (`breakeven_cost`, `net_spread`).
2. `_STAGE1_HELPERS` — drops shared `compute_*` helpers (they produce
   stage-1 frames consumed by other metrics).
3. `_AUTO_DISCOVER_EXCLUDED` — drops metrics that need explicit kwargs
   `run_metrics` does not thread (per-row reason; surfaced on
   `bundle.skipped`).

In v1 the IC family (`ic`, `ic_newey_west`, `ic_ir`) shares a single
`compute_ic(panel)` per call. Other stage-1 consumers
(`caar`, `fama_macbeth`, `ts_beta`, `mfe_mae_summary`, plus series /
spread consumers) live in the auto-discover exclusion set; the bundle's
`skipped` map carries the explicit-import recipe for each. v1.x will
extend stage-1 wiring per cell.

## Explicit subset

```python
fx.run_metrics(panel, cfg, metrics=["ic", "monotonicity"])
```

Unknown names raise [`UserInputError`][factrix.UserInputError] with a fuzzy
suggestion plus the full candidate list. Names registered for the
cell but in the auto-discover exclusion set raise the same error type with
the documented reason and the explicit-call recipe — `run_metrics`
never silently drops a name the caller asked for.

## Cross-horizon and cross-universe analysis

`run_metrics` runs at exactly one horizon — `cfg.forward_periods` — and
on exactly the panel passed in. Sweeps go through the existing functions;
`run_metrics` does not ship a `horizons=[...]` helper because
cross-horizon analysis is the job of `compare(bundles)` (descriptive,
v1.x — see #148) or `bhy(expand_over=["forward_periods"])` (FDR
controlled, anti-shopping defense per #160). Keeping `run_metrics`
single-horizon preserves the `(panel, cfg) → bundle` contract
symmetric with `evaluate`.

```python
horizons = [1, 5, 21]

# Descriptive sweep — no FDR claim
bundles = [
    fx.run_metrics(panel, cfg.replace(forward_periods=h))
    for h in horizons
]
# compare(bundles)  # descriptive cross-factor view; v1.x function, see #148

# Inferential sweep — FDR-controlled
profiles = [
    fx.evaluate(panel, cfg.replace(forward_periods=h))
    for h in horizons
]
fx.multi_factor.bhy(profiles, expand_over=["forward_periods"])
```

Universe / regime works the same way: filter the panel, optionally
stamp `bundle.context` via `dataclasses.replace`, then concatenate
through `compare` / `bhy`. See the [Identity / context](identity.md)
guide.

## MetricsBundle

Frozen dataclass keyed by `identity = (factor_id, forward_periods)`
(matches `FactorProfile.identity` per #160).

| Member | Type | Description |
|---|---|---|
| `identity` | `(str, int)` | hypothesis dimensions |
| `metrics` | `Mapping[str, MetricOutput]` | every metric that produced a value (incl. short-circuit `NaN` outputs with `metadata["reason"]`) |
| `skipped` | `Mapping[str, str]` | metric → reason for everything excluded from auto-discover |
| `context` | `Mapping[str, Any]` | sample-restriction dimensions; v1 always empty, populated by downstream slicers or by user via `dataclasses.replace` after panel-side filtering |

Access patterns:

- `bundle["ic"]` — dict-style metric lookup
- `"ic" in bundle` / `list(bundle)` / `iter(bundle)` — operate on the
  metric keys
- `bundle.to_frame()` — long-form `pl.DataFrame` (one row per metric);
  fixed 8-column schema for stable `pl.concat([b.to_frame() ...])`

Hashing is disabled (`__hash__ = None`) because the bundle holds
`MetricOutput` instances whose `metadata` is a mutable dict. Group
bundles by `identity` (a hashable tuple).

## to_frame schema

| Column | Type | Source |
|---|---|---|
| `factor_id` | `str` | `identity[0]` |
| `forward_periods` | `int` | `identity[1]` |
| `metric` | `str` | mapping key |
| `value` | `float` | `MetricOutput.value` |
| `stat` | `float \| null` | `MetricOutput.stat` |
| `significance` | `str \| null` | `MetricOutput.significance` |
| `p_value` | `float \| null` | `metadata["p_value"]` |
| `short_circuit_reason` | `str \| null` | `metadata["reason"]` |

`metadata` is **not** flattened — its shape is heterogeneous across
metrics (per-regime dicts, per-horizon entries, KP source labels…).
Reach into `bundle["name"].metadata` directly. `context.*` is **not**
flattened in v1 (the column would always be empty).

## Error handling

Three classes, three responses:

| Class | Source | What `run_metrics` does |
|---|---|---|
| **A** Sample-floor / data-quality | `InsufficientSampleError`, metric-internal `_short_circuit_output` | Convert to a short-circuit `MetricOutput` (`value=NaN`, `metadata["reason"]=...`) inside the bundle. Other metrics keep running. |
| **B** User input | unknown / excluded `metrics=[...]`, missing / colliding `factor_col` | Raise [`UserInputError`][factrix.UserInputError] with fuzzy suggestion + fix path. |
| **C** Unexpected | bug in a metric callable or stage-1 helper | Raise [`RunMetricsError`][factrix.RunMetricsError] wrapping the original exception (chained via `__cause__`); attributes `.cell`, `.metric_name`, `.stage` identify which metric broke. |

A single `logging.info` line at logger name `factrix.run_metrics`
summarises ran / skipped counts per call (observability hook for
batch jobs; not the primary user surface — `bundle.skipped` is).
