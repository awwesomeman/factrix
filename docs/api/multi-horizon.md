---
title: Multi-horizon analysis
---

The `multi_horizon_ic` and `multi_horizon_hit_rate` functions were
deprecated in v0.11.0 and removed in v0.12.0. This page documents the
two supported migration paths.

## Why they were removed

Sweeping a metric across `[1, 5, 10, 20]` forward periods is a
**dispatcher** concern, not a per-cell metric: it fans the same
underlying metric across a horizon axis exactly the way
[`by_slice`](by-slice.md) fans across a date or label axis. Burying the horizon loop inside a metric callable
created three structural problems:

1. **Cross-cuts the metric registry.** The horizon loop produced one
   `MetricResult` aggregating `k` horizons, but the registry is keyed
   per `(scope, density, metric)`.
2. **Conflicted with the identity-as-family contract.**
   `EvaluationResult` carries `forward_periods` precisely to make
   horizon shopping explicit at the false discovery rate (FDR) layer; the in-metric horizon
   loop collapsed `k` horizons into one identity entry, defeating that
   defense.
3. **Two Benjamini-Hochberg-Yekutieli (BHY) paths.** The family-function
   layer ([`multi_factor.bhy`][bhy] with `expand_over=("forward_periods",)`)
   is the single source of truth for FDR control across horizons.

The dispatcher framing also lets descriptive metrics (`mfe_mae`,
`caar`, `oos`, `monotonicity`, ...) inherit horizon-sweep support
automatically.

## Migration recipes

### Descriptive sweep — horizon-by-metric magnitudes

Call [`evaluate`](evaluate.md) per horizon and assemble a long
horizon × metric table from each `EvaluationResult.to_frame()`.
No FDR claim is made.

```python
import factrix as fx
import polars as pl
from factrix.metrics import ic

horizons = [1, 5, 10, 20]
results = []
for h in horizons:
    res = fx.evaluate(
        panel,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=["mom_12_1"],
        forward_periods=h,
    )
    results.extend(res.values())

table = pl.concat([r.to_frame() for r in results])  # long-form: horizon x metric
```

### Inferential sweep — FDR-controlled across horizons

Use [`evaluate`](evaluate.md) per horizon and
[`multi_factor.bhy`][bhy] with `expand_over=("forward_periods",)`. The
family-function layer partitions the BHY null per horizon so the step-up threshold is correct.

```python
import factrix as fx
from factrix.metrics import ic

horizons = [1, 5, 10, 20]
results = []
for h in horizons:
    res = fx.evaluate(
        panel,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=["mom_12_1"],
        forward_periods=h,
    )
    results.extend(res.values())

fdr_res = fx.multi_factor.bhy(
    results,
    metrics=["ic"],
    expand_over=("forward_periods",),
)

bhy_ic = fdr_res["ic"]
survivors = bhy_ic.survivors
adj_p = bhy_ic.adj_p
```

[bhy]: multi-factor.md
