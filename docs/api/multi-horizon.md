---
title: Multi-horizon analysis
---

The `multi_horizon_ic` and `multi_horizon_hit_rate` functions were
deprecated in v0.11.0 and removed in v0.12.0. This page documents the
two supported migration paths. See [#186][i186] for the design history.

## Why they were removed

Sweeping a metric across `[1, 5, 10, 20]` forward periods is a
**dispatcher** concern, not a per-cell metric: it fans the same
underlying metric across a horizon axis exactly the way
[`by_slice`](by-slice.md) fans across a date or label axis. Burying the horizon loop inside a metric callable
created three structural problems:

1. **Cross-cuts the metric registry.** The horizon loop produced one
   `MetricResult` aggregating `k` horizons, but the registry is keyed
   per `(scope, signal, metric)` — so a horizon-loop result had to
   pretend to be a single metric, hiding the loop in `metadata`.
2. **Conflicted with the identity-as-family contract** ([#160][i160]).
   `FactorProfile.identity` carries `forward_periods` precisely to make
   horizon shopping explicit at the false discovery rate (FDR) layer; the in-metric horizon
   loop collapsed `k` horizons into one identity entry, defeating that
   defense.
3. **Two Benjamini-Hochberg-Yekutieli (BHY) paths.** The information coefficient (IC) variant ran its own internal BHY
   adjustment and wrote `metadata["p_adjusted_bhy"]`. The family-function
   layer ([`multi_factor.bhy`][bhy] with `expand_over=["forward_periods"]`)
   is the single source of truth for FDR control across horizons.

The dispatcher framing also lets descriptive metrics (`mfe_mae`,
`caar`, `oos`, `monotonicity`, ...) inherit horizon-sweep support
automatically; the previous shape special-cased IC and hit-rate.

## Migration recipes

### Descriptive sweep — horizon-by-metric magnitudes

Call [`evaluate`](evaluate.md) per horizon and assemble a long
horizon × metric table from each `EvaluationResult.to_frame()`.
No FDR claim is made.

```python
import factrix as fx
import polars as pl
from factrix._metric_index import spec_by_name

specs = spec_by_name()
horizons = [1, 5, 10, 20]
results = [
    fx.evaluate(
        panel,
        metrics=[specs["ic"]],
        factor_cols=["mom_12_1"],
        forward_periods=h,
    )[0]
    for h in horizons
]
table = pl.concat([r.to_frame() for r in results])  # long-form: horizon x metric
```

### Inferential sweep — FDR-controlled across horizons

Use [`evaluate`](evaluate.md) per horizon and
[`multi_factor.bhy`][bhy] with `expand_over=["forward_periods"]`. The
family-function layer partitions the BHY null per horizon (and per any
other declared `expand_over` key) so the step-up threshold is correct.

```python
import factrix as fx
from factrix._metric_index import spec_by_name

specs = spec_by_name()
results = [
    fx.evaluate(
        panel,
        metrics=[specs["ic"]],
        factor_cols=["mom_12_1"],
        forward_periods=h,
    )[0]
    for h in horizons
]
survivors = fx.multi_factor.bhy(
    results,
    expand_over=["forward_periods"],
)
```

`survivors.profiles` carries the kept rows; `survivors.adj_p` carries
the bucket-local BHY-adjusted p-values.

[i160]: https://github.com/awwesomeman/factrix/issues/160
[i186]: https://github.com/awwesomeman/factrix/issues/186
[bhy]: multi-factor.md
