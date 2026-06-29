---
title: Standalone metrics
---

Every metric under `factrix.metrics` can be run either as part of a multi-metric execution plan using [`evaluate()`][factrix.evaluate] or invoked directly as a **standalone metric** helper on a Polars DataFrame.

This guide covers direct-call mechanics: input shape, return shape, and
when to prefer `evaluate()` so the DAG can resolve shared dependencies. For
metric selection by research question, use [Choosing a metric](choosing-metric.md).

## Direct-call shapes

| Shape | Typical callables | Return shape |
|---|---|---|
| Long panel `(date, asset_id, factor, forward_return)` | `quantile_spread`, `monotonicity`, `directional_hit_rate`, `rank_turnover` | `MetricResult` or `dict[str, MetricResult]` for batchable helpers |
| Two-column series `(date, value)` | `oos_decay`, `ic_trend`, `positive_rate` | `MetricResult` |
| Producer output / aligned auxiliary input | `caar`, `spanning_alpha`, `greedy_forward_selection`, `breakeven_cost`, `net_spread` | `MetricResult` |

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

Diagnostics like `positive_rate`, `ic_trend`, and `oos_decay` take a two-column time-series DataFrame of `(date, value)` (such as a series of per-date ICs generated upstream).

```python
import numpy as np
import polars as pl
from datetime import date, timedelta
from factrix.metrics import oos_decay

# oos_decay splits the series into in-sample / out-of-sample halves, so it
# needs enough points to estimate both — a handful of rows returns nan.
# Here: 24 monthly IC values that decay from ~0.10 to ~0.02.
rng = np.random.default_rng(0)
n = 24
ic_series = pl.DataFrame({
    "date": [(date(2026, 1, 1) + timedelta(days=30 * i)).isoformat() for i in range(n)],
    "value": np.linspace(0.10, 0.02, n) + rng.normal(0, 0.01, n),
})

decay_res = oos_decay(ic_series)
print(decay_res.value)  # OOS/IS retention ratio
```

---

## 2. Integrated evaluation with `evaluate()`

Instead of calling multiple metrics manually and managing intermediate outputs, you can pass them together in the `metrics` dictionary of `fx.evaluate()`. The DAG executor automatically schedules and resolves any shared dependencies (like `compute_ic` or bucketing) to ensure optimal performance.

```python
import factrix as fx
from factrix.metrics import ic, quantile_spread, monotonicity

raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=200, seed=42)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

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
import factrix as fx

overview = fx.list_metrics()
# Returns dict[family_name, list[MetricSpec]]
print(overview.keys())
```

To find only the metrics that are statistically applicable to a specific panel's dimensions, use `inspect_data()`:

```python
import factrix as fx

raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=200, seed=42)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)

inspection = fx.inspect_data(panel)
usable_metrics = [m.name for m in inspection.usable]
print(usable_metrics)
```
