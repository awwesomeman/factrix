# Metric pipelines

Cross-module index of every module under `factrix/metrics/`. Use this
page to pick the existing aggregation pattern a new metric should
match, or to mechanically check that a candidate metric satisfies the
[NW HAC discipline](statistical-methods.md) the rest of the suite
follows.

The matrix lists **all** metric modules — both the procedure-canonical
primaries that `evaluate()` invokes internally
(`ic`, `fama_macbeth`, `caar`, `ts_beta`) and the standalone helpers
the user calls directly on a `FactorProfile` (`quantile`, `monotonicity`,
`tradability`, `clustering`, `corrado`, …). The
[`list_metrics`](../api/list-metrics.md) runtime API filters this same
data to the standalone subset by `(scope, signal)`.

The table below is auto-generated from `Matrix-row:` tags in each module's
docstring. It surfaces the three columns most relevant to understanding
*calculation logic*: which factor type the module applies to, how it
aggregates, and what inference procedure it uses. For the full function list
and internal primitives, click the module link to the source.

## Aggregation vocabulary

The `agg_order` column uses one canonical lowercase-hyphen form across
the matrix, every `Matrix-row:` tag in `factrix/metrics/*.py`, and the
`list_metrics()` JSON output:

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
that returns the same data filtered by `(scope, signal)`, see
[`list_metrics`](../api/list-metrics.md).
