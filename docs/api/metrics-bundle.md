# MetricsBundle

Result type returned by [`run_metrics`](run-metrics.md) — holds every
standalone metric the cell successfully evaluated for one factor at one
horizon, plus a `skipped` map for metrics that could not auto-run.

Parallel to [`FactorProfile`](factor-profile.md): both share the
identity tuple `(factor_id, forward_periods)` so downstream verbs
([`compare`](compare.md), survivors workflows) can join the two
artifact families row-by-row.

| Bundle | Profile |
|---|---|
| `run_metrics` — descriptive surface (no FDR claim) | `evaluate` — primary inferential decision (drives FDR) |
| Many metrics per cell, each a [`MetricOutput`](metric-output.md) | One primary stat + sidecar `stats` dict |
| `compare(bundles)` → cross-factor descriptive view | `compare(profiles)` / `compare(survivors)` |

## Typical use

```python
import factrix as fx

cfg    = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC, forward_periods=5)
bundle = fx.run_metrics(panel, cfg, factor_col="momentum_12_1")

bundle.identity           # ('momentum_12_1', 5)
bundle.metrics            # dict[str, MetricOutput]
bundle.to_frame()         # pl.DataFrame — flat tabular view for compare / plotting
bundle.skipped            # {metric_name: reason} — metrics that short-circuited
```

For the wide multi-factor pattern (looping `run_metrics` with
`factor_col=` over candidate signals) see the
[Batch screening guide](../guides/batch-screening.md).

::: factrix.MetricsBundle
