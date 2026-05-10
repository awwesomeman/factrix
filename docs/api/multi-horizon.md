# Multi-horizon analysis

`factrix.metrics.multi_horizon_ic` and `factrix.metrics.multi_horizon_hit_rate`
are deprecated. Both functions remain importable and runnable for one
release cycle, but they emit `DeprecationWarning` on call and no longer
appear in [`list_metrics`](list-metrics.md) output.

## Why the deprecation

Sweeping a metric across `[1, 5, 10, 20]` forward periods is a
**dispatcher** concern, not a per-cell metric: it fans the same
underlying metric across a horizon axis exactly the way
[`by_regime`](by-regime.md) / [`by_slice`](by-slice.md) fan across a
date or label axis. Burying the horizon loop inside a metric callable
created three structural problems:

1. **Cross-cuts the metric registry.** The horizon loop produces one
   `MetricOutput` aggregating `k` horizons, but the registry is keyed
   per `(scope, signal, metric)` — so a horizon-loop result has to
   pretend to be a single metric, hiding the loop in `metadata`.
2. **Conflicts with the identity-as-family contract** ([#160][i160]).
   `FactorProfile.identity` carries `forward_periods` precisely to make
   horizon shopping explicit at the FDR layer; `multi_horizon_*`
   collapsed `k` horizons into one identity entry, defeating that
   defense.
3. **Two BHY paths.** `multi_horizon_ic` ran its own internal BHY
   adjustment and wrote `metadata["p_adjusted_bhy"]`. The family-verb
   layer (`multi_factor.bhy(profiles, expand_over=["forward_periods"])`)
   is the single source of truth for FDR control across horizons.

The dispatcher framing also lets future descriptive metrics
(`mfe_mae`, `caar`, `oos`, `monotonicity`, ...) inherit horizon-sweep
support automatically; the previous shape special-cased IC and
hit-rate.

## Migration recipes

### Descriptive sweep — looking at horizon-by-metric magnitudes

Use [`run_metrics`](run-metrics.md) per horizon and assemble a long
horizon × metric table from each bundle's `.to_frame()`. No FDR claim
is made.

```python
import factrix as fx
import polars as pl

cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
horizons = [1, 5, 10, 20]
bundles = [
    fx.run_metrics(panel, cfg.replace(forward_periods=h), factor_col="mom_12_1")
    for h in horizons
]
table = pl.concat([b.to_frame() for b in bundles])  # long-form: horizon x metric
```

Every metric the cell exposes — not just IC — gets a horizon view
through this single call. A higher-level `compare(bundles)` verb that
pivots this long-form table is tracked in #148 as a follow-up; today
the `pl.concat` recipe above is the supported path.

### Inferential sweep — controlling FDR across horizons

Use [`evaluate`](evaluate.md) per horizon and
[`multi_factor.bhy`][bhy] with `expand_over=["forward_periods"]`. The
family-verb layer partitions the BHY null per horizon (and per any
other declared `expand_over` key) so the step-up threshold is correct.

```python
import factrix as fx

cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC)
profiles = [
    fx.evaluate(panel, cfg.replace(forward_periods=h))
    for h in horizons
]
survivors = fx.multi_factor.bhy(
    profiles,
    expand_over=["forward_periods"],
)
```

`survivors.profiles` carries the kept rows; `survivors.adj_q` carries
the bucket-local BHY-adjusted p-values.

## Removal timeline

Removal lands in a future major-bump release-train. The exact version
is fixed at bump time — see the `BUMP-TIME` markers in
`factrix/metrics/ic.py` and `factrix/metrics/event_horizon.py`.
Tracked in [#186][i186].

[i160]: https://github.com/awwesomeman/factrix/issues/160
[i186]: https://github.com/awwesomeman/factrix/issues/186
[bhy]: multi-factor.md
