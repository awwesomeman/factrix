# Standalone metric pipelines

Cross-module index of every module under `factrix/metrics/`. Use this
page to pick the existing aggregation pattern a new metric should
match, or to mechanically check that a candidate metric satisfies the
[NW HAC discipline](statistical-methods.md) the rest of the suite
follows.

The table below is auto-generated from `Matrix-row:` tags in each module's
docstring. It surfaces the three columns most relevant to understanding
*calculation logic*: which factor type the module applies to, how it
aggregates, and what inference procedure it uses. For the full function list
and internal primitives, click the module link to the source.

## Aggregation vocabulary

- **CS-first** — aggregate cross-section per date first, then aggregate
  the resulting time series.
- **TS-first** — aggregate time-series per asset first, then aggregate
  across assets.
- **TS-only** — single-series time-series operation; no cross-section
  step.
- **Static CS** — single cross-section, no time-axis aggregation.
- **Per-event** — aggregation centred on event dates (per-event-date
  step), then cross-event aggregation.

## Matrix

--8<-- "docs/reference/_generated_metric_matrix.md"

For per-module formula derivations, read each module's top-level
docstring (linked above); for the underlying paper references and
inference-SE rationale, see [Statistical methods](statistical-methods.md);
for `n_obs` / `n_assets` thresholds per metric, see
[Metric applicability](metric-applicability.md).
