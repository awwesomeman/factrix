---
title: Metric pipelines
---

!!! abstract "Answers"
    How is each metric computed — aggregation order, inference SE machinery.
    For applicability gates and sample thresholds, see [Metric applicability](metric-applicability.md).
    For output schema and metadata keys, see [Stat keys by metric](stat-keys-by-metric.md).
    For the research-question → metric mapping, see [Choosing a metric](../guides/choosing-metric.md).

Cross-module index of every module under `factrix/metrics/`. Use this
page to pick the existing aggregation pattern a new metric should
match, or to mechanically check that a candidate metric satisfies the
[Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) discipline](statistical-methods.md) the rest of the suite
follows.

The matrix lists **all** metric modules — both the metrics
[`evaluate()`](../api/evaluate.md) runs for each cell
(`ic`, `fm_beta`, `caar`, `ts_beta`)
and the standalone helpers users can call directly on their declared input
shape (`quantile`, `monotonicity`, `tradability`, `clustering`, `corrado`, …).
The [`list_metrics`](../api/metrics/index.md#factrix.list_metrics) runtime API
exposes the same public spec set as a family-grouped catalog.

The table below is auto-generated from the public `MetricSpec` registry. It
surfaces the three columns most relevant to understanding
*calculation logic*: which factor type the module applies to, how it
aggregates, and what inference procedure it uses. For the full function list
and internal primitives, click the module link to the source.

## Aggregation vocabulary

The `agg_order` column uses one canonical lowercase-hyphen form across
the matrix and every metric's registered aggregation metadata:

- **`cs-first`** — aggregate cross-section per date first, then aggregate
  the resulting time series. Pairs with the
  [Guides § Aggregation order](../guides/panel-timeseries.md#aggregation-order)
  *cross-section first* prose.
- **`ts-first`** — aggregate time-series per asset first, then aggregate
  across assets. Pairs with the *time-series first* prose.
- **`ts-only`** — single-series time-series operation; no cross-section
  step.
- **`static-cs`** — single cross-section, no time-axis aggregation.
- **`per-event`** — aggregation centred on event dates (per-event-date
  step), then cross-event aggregation.

## Matrix

--8<-- "docs/reference/_generated_metric_matrix.md"

For per-module formula derivations, read each module's top-level
docstring (linked above); for the underlying paper references and
inference-SE rationale, see [Statistical methods](statistical-methods.md);
for `n_obs` / `n_assets` thresholds per metric, see
[Metric applicability](metric-applicability.md). For the runtime API
that returns the same public specs grouped by family, see
[`list_metrics`](../api/metrics/index.md#factrix.list_metrics).
