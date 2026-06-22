---
title: Standalone metrics
---

Every metric under `factrix.metrics` can be run either as part of a multi-metric execution plan using [`evaluate()`][factrix.evaluate] or invoked directly as a **standalone metric** helper on a Polars DataFrame.

This guide covers how to call standalone metrics, their input shapes, and how to integrate them into your evaluation workflow.

## When to use which metric

| Analysis Target | Metric | Return Type |
|---|---|---|
| Quantile spread & monotonicity | `quantile_spread`, `quantile_spread_vw`, `monotonicity` | `dict[str, MetricResult]` (batchable) |
| Tradability & cost break-even | `turnover`, `notional_turnover`, `breakeven_cost`, `net_spread` | `MetricResult` |
| Spanning regression vs existing pool | `spanning_alpha`, `greedy_forward_selection` | `MetricResult` |
| Event study analysis | `caar`, `bmp_z`, `corrado_rank`, `event_hit_rate`, `clustering_hhi` | `MetricResult` |
| Event return shape | `mfe_mae`, `event_around_return` | `MetricResult` |
| Series diagnostics | `oos_decay`, `ic_trend`, `hit_rate` | `MetricResult` |

---

## 1. Direct standalone calls

You can call any metric callable directly. If the first argument is a Polars DataFrame or Series, the metric runs immediately and returns its results.

### Panel-input metrics

Metrics such as `quantile_spread` or `monotonicity` operate on a long-format panel containing `(date, asset_id, <factor_col>, forward_return)`.

```python
import factrix as fx
from factrix.metrics import quantile_spread, monotonicity

# Generate synthetic data
raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=200, seed=42)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

# Call metrics directly
spread_res = quantile_spread(panel, forward_periods=5, n_groups=5)
# Returns a dictionary: {factor_name: MetricResult}
factor_spread = spread_res["factor"]
print(factor_spread.value)  # Mean spread
print(factor_spread.p_value)  # Non-overlapping t-test p-value
```

### Series diagnostics

Diagnostics like `hit_rate`, `ic_trend`, and `oos_decay` take a two-column time-series DataFrame of `(date, value)` (such as a series of per-date ICs generated upstream).

```python
import polars as pl
from factrix.metrics import oos_decay

# Given a time series of values (e.g. daily IC values)
ic_series = pl.DataFrame({
    "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
    "value": [0.05, -0.02, 0.08]
})

decay_res = oos_decay(ic_series)
print(decay_res.value)
```

---

## 2. Integrated evaluation with `evaluate()`

Instead of calling multiple metrics manually and managing intermediate outputs, you can pass them together in the `metrics` dictionary of `fx.evaluate()`. The DAG executor automatically schedules and resolves any shared dependencies (like `compute_ic` or bucketing) to ensure optimal performance.

```python
import factrix as fx
from factrix.metrics import ic, quantile_spread, monotonicity

results = fx.evaluate(
    panel,
    metrics={
        "ic": ic(inference=fx.inference.NEWEY_WEST),
        "spread": quantile_spread(n_groups=5),
        "mono": monotonicity(n_groups=5),
    },
    factor_cols=["factor"],
    forward_periods=5,
)

res = results["factor"]
print(res.metrics["ic"].value)
print(res.metrics["spread"].value)
```

## Metric Discovery

To programmatic inspect the public metrics catalog, use `list_metrics()`:

```python
overview = fx.list_metrics()
# Returns dict[family_name, list[MetricSpec]]
print(overview.keys())
```

To find only the metrics that are statistically applicable to a specific panel's dimensions, use `inspect_data()`:

```python
inspection = fx.inspect_data(panel)
usable_metrics = [m.name for m in inspection.usable]
```
