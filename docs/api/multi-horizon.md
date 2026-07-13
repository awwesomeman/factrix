---
title: Multi-horizon analysis
---

Sweeping a metric across several forward-return horizons (e.g.
`[1, 5, 10, 20]`) is a **dispatcher** concern, not a per-cell metric:
[`evaluate_horizons`](evaluate.md) fans the same underlying metric across a
horizon axis exactly the way [`by_slice`](by-slice.md) fans across a date or
label axis. This page documents the two supported recipes.

## Why horizon sweep lives in the dispatcher

Keeping the horizon loop out of the metric callable avoids three structural
problems:

1. **Cross-cuts the metric registry.** A horizon loop inside a metric would
   produce one `MetricResult` aggregating `k` horizons, but the registry is
   keyed per `(scope, density, metric)`.
2. **Conflicts with the identity-as-family contract.**
   `EvaluationResult` carries `forward_periods` precisely to make
   horizon shopping explicit at the false discovery rate (FDR) layer; an
   in-metric horizon loop would collapse `k` horizons into one identity
   entry, defeating that defense.
3. **One Benjamini-Hochberg-Yekutieli (BHY) path.** The family-function
   layer ([`multi_factor.bhy`][bhy]) is the single source of truth for FDR
   control. The caller declares whether horizons compete in one family or are
   predeclared, separately reported buckets.

The dispatcher framing also lets descriptive metrics (`mfe_mae`,
`caar`, `oos`, `monotonicity`, ...) inherit horizon-sweep support
automatically.

## Recipes

[`evaluate_horizons`](evaluate.md) is the convenience entry for both
recipes: it rebuilds an evaluation panel from the raw input with
[`compute_forward_return`](preprocess.md) per horizon, runs
[`evaluate`](evaluate.md) at each, and flattens to a single
`list[EvaluationResult]` — one entry per `(factor, forward_periods)`,
ready for [`compare`](compare.md) and [`multi_factor.bhy`][bhy]. It takes
a **raw** panel (no `forward_return` attached) and computes only the
forward return; for per-horizon winsorize / abnormal-return, fall back
to the explicit `for h in horizons: evaluate(...)` loop instead.

### Descriptive sweep — horizon-by-metric magnitudes

Sweep with [`evaluate_horizons`](evaluate.md) and assemble a long
horizon × metric table from each `EvaluationResult.to_frame()`.
No FDR claim is made.

```python
import factrix as fx
import polars as pl
from factrix.metrics import ic

raw = fx.datasets.make_cs_panel(n_assets=80, n_dates=240)

results = fx.evaluate_horizons(
    raw,  # no forward_return attached
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["mom_12_1"],
    forward_periods=[1, 5, 10, 20],
)

table = pl.concat([r.to_frame() for r in results])  # long-form: horizon x metric
```

### Inferential sweep — FDR-controlled across factors and horizons

Feed the swept list to [`multi_factor.bhy`][bhy] without `expand_over` when the
research process may select any factor × horizon combination. This pools every
searched hypothesis into the controlled family. A runtime warning makes that
choice visible; it is informational, not an error.

```python
import factrix as fx
from factrix.metrics import ic

raw = fx.datasets.make_cs_panel(n_assets=80, n_dates=240)

results = fx.evaluate_horizons(
    raw,  # no forward_return attached
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["mom_12_1"],
    forward_periods=[1, 5, 10, 20],
)

fdr_res = fx.multi_factor.bhy(
    results,
    metrics=["ic"],
)

bhy_ic = fdr_res["ic"]
survivors = bhy_ic.survivors
adj_p = bhy_ic.adj_p
```

Use `expand_over=("forward_periods",)` only for horizon-specific screens that
were predeclared and will be selected and reported separately. It runs one
step-up per horizon and does not control a later choice of the best horizon.

Cross-horizon comparability is a scale alignment only: the `/ forward_periods` in
`compute_forward_return` makes rank-IC comparable across horizons, but
signed-return-mean metrics carry a compounding bias that grows with `forward_periods`
(see [`compute_forward_return`](preprocess.md)).

[bhy]: multi-factor.md
