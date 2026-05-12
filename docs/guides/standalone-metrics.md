# Standalone metrics

[`evaluate()`][factrix.evaluate] runs one canonical procedure per cell
and returns a [`FactorProfile`][factrix.FactorProfile] with a single
`primary_p`. Every other module under `factrix.metrics` is a
**standalone metric** — a `MetricOutput`-returning helper the user
invokes directly to add diagnostics around the canonical inference.

This guide covers the three things a user needs to wire them in:
which metrics apply to a given cell, what input shape each one
expects, and where they sit in a screening pipeline.

## Scope

factrix evaluates **factor signal validity** — predictive power
(IC, FM λ), robustness (NW HAC, BHY, regime stability), and event
shape (CAAR, BMP, Corrado). It does **not** compute portfolio-level
performance: no Sharpe / drawdown / Sortino / Calmar / return
attribution. Those metrics belong downstream of factor selection
and live in dedicated backtest libraries (e.g.
[`vectorbt`](https://vectorbt.dev/) or an internal framework).

The closest crossover inside factrix is `ic_ir` — the information
ratio of the per-date IC series. It is explicitly a *signal-quality*
IR (mean / std of IC), **not** a portfolio Sharpe (mean / std of
realised returns).

For the cross-module matrix (every module, aggregation pattern,
inference SE), see
[Reference § Metric pipelines](../reference/metric-pipelines.md).
For the `(scope, signal)` filter at runtime, see
[`list_metrics`](../api/list-metrics.md).

## When to use which

| Want | Reach for |
|---|---|
| `primary_p` + diagnose | [`evaluate()`][factrix.evaluate] |
| Quintile spread, monotonicity, top concentration | `quantile_spread`, `monotonicity`, `top_concentration` |
| Tradability / cost break-even | `notional_turnover`, `breakeven_cost`, `net_spread`, `turnover` |
| Spanning regression vs an existing pool | `spanning_alpha`, `greedy_forward_selection` |
| Event-side robustness on top of CAAR | `bmp_test`, `corrado_rank_test`, `event_hit_rate`, `clustering_diagnostic` |
| Per-event return shape | `mfe_mae_summary`, `event_around_return` |
| Asymmetry / quantile-spread on a broadcast factor | `ts_asymmetry`, `ts_quantile_spread` |
| OOS decay / trend / hit rate on any `(date, value)` series | `multi_split_oos_decay`, `ic_trend`, `hit_rate` |

`evaluate()` only writes the `StatCode` keys listed for its cell on
[`FactorProfile`](../api/factor-profile.md#stats-keys-by-cell);
standalone metrics return their own
[`MetricOutput`](../api/metric-output.md) and never mutate
`profile.stats`. The two surfaces compose without coordination.

## Input shapes

Standalone metrics group into three input families:

### 1. Panel `(date, asset_id, factor, forward_return)` — same shape as `evaluate()`

The IC / FM / quantile / Fama-MacBeth / spanning / tradability
families read the same wide panel `evaluate()` consumes. Pass the
exact dataframe you handed to `evaluate()`:

```python
import factrix as fx

raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
profile = fx.evaluate(panel, fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC))

spread = fx.metrics.quantile_spread(panel, forward_periods=5, n_groups=5)
mono = fx.metrics.monotonicity(panel, forward_periods=5, n_groups=5)
```

### 2. Event panel `{factor ∈ {0, R}}` — sparse signal cells

`caar`, `event_quality`, `event_horizon`, `mfe_mae`, `clustering`, and
`corrado` read a panel where `factor` is zero on non-event entries
and arbitrary real magnitude on event entries (canonical `{-1, 0, +1}`,
also valid: `{0, R≥0}` or `{-R, 0, +R}`). Build via
`fx.datasets.make_event_panel` or filter your own panel.

```python
events = fx.datasets.make_event_panel(n_assets=80, n_dates=500, event_rate=0.05, seed=7)
events = fx.preprocess.compute_forward_return(events, forward_periods=5)

profile = fx.evaluate(events, fx.AnalysisConfig.individual_sparse(forward_periods=5))

# Robustness on top of the canonical CAAR:
hit = fx.metrics.event_hit_rate(events)
rank_p = fx.metrics.corrado_rank_test(events, forward_periods=5)
```

### 3. Derived `(date, value)` series — series diagnostics

`hit_rate`, `ic_trend`, and `multi_split_oos_decay` are axis-agnostic:
they take whatever per-date series an upstream cell metric produced.
This decouples the diagnostic from the cell contract — a per-date IC
series, a per-date quantile spread, and a per-date FM λ all share the
same shape and feed the same diagnostics.

```python
import polars as pl

ic_series: pl.DataFrame = ...  # columns: date, value (per-date IC)

decay = fx.metrics.multi_split_oos_decay(ic_series)
trend = fx.metrics.ic_trend(ic_series)
hits = fx.metrics.hit_rate(ic_series, value_col="value")
```

`Mode.TIMESERIES` (the dispatch regime for `n_assets == 1`) is a
distinct concept — these diagnostics are series-shaped regardless of
which Mode the upstream procedure ran in.

## Post-`evaluate()` integration

A typical screening recipe chains `evaluate()` for the canonical inference, then
appends standalone metrics for shape and tradability diagnostics. The
profile and the metric outputs travel together; nothing on either
side needs to know about the other:

```python
profile = fx.evaluate(panel, fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC))
diagnostics = {
    "primary_p": profile.primary_p,
    "spread": fx.metrics.quantile_spread(panel, forward_periods=5).value,
    "monotonicity_p": fx.metrics.monotonicity(panel, forward_periods=5).p_value,
    "breakeven_bps": fx.metrics.breakeven_cost(panel).value,
}
```

For the BHY family across many factors, `multi_factor.bhy` consumes
[`FactorProfile`][factrix.FactorProfile] objects only — standalone
metrics are not in the multiple-testing partition. Run them in a
separate pass after BHY narrows the candidate set.

## Discovery

[`list_metrics(scope, signal)`](../api/list-metrics.md) returns the
standalone subset for a given cell, sorted by module then function.
Use it as the runtime equivalent of the static matrix:

```python
fx.list_metrics(fx.FactorScope.INDIVIDUAL, fx.Signal.CONTINUOUS)
# ['concentration.top_concentration', 'fama_macbeth.beta_sign_consistency', ...]
```
