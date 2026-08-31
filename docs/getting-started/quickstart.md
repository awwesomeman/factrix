---
title: Quickstart
---

!!! warning "`forward_periods` counts periods, not calendar time"
    factrix never reads the calendar: every horizon, window and lag is a
    count of periods on the panel's own date grid. `forward_periods=5` is
    five periods of whatever one period represents on your grid; factrix
    never infers that from the dates. Aligning the factor and price sources
    onto one grid is the caller's job. See
    [Period grid, not calendar](../development/architecture.md#period-grid-not-calendar).

## 30-second smoke test

```python
import factrix as fx
from factrix.preprocess import compute_forward_return
from factrix.metrics import ic

# 1. Generate synthetic panel data and compute forward returns
raw   = fx.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, rng=2024)
data  = compute_forward_return(raw, forward_periods=5)

# 2. Run single-factor evaluation using the ic metric with Newey-West
results = fx.evaluate(
    data,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor"],
    forward_periods=5,
)

res = results["factor"]
ic_res = res.metrics["ic"]

print('ic_mean =', round(ic_res.value, 4))
# → ic_mean = 0.0722
print('p_value =', round(ic_res.p_value, 4))
# → p_value = 0.0
```

See [Concepts](concepts.md) for what each axis means.

---

## Bringing your own data

The smoke test uses synthetic data that already carries factrix's
canonical column names (`date`, `asset_id`, `price`). Real-world panels
rarely do, so `adapt` is the **first** step of the pipeline — it renames
your columns to the canonical names (and optionally cleans non-finite
values), *before* `compute_forward_return`:

```python title="Illustrative"
import factrix as fx
from factrix.adapt import adapt
from factrix.preprocess import compute_forward_return

raw = adapt(
    your_df,
    date="trade_date", asset_id="ticker", price="close_adj",
    fill_forward=True,   # map NaN/±inf → null, then forward-fill per asset
)
data = compute_forward_return(raw, forward_periods=5)
results = fx.evaluate(data, metrics={"ic": fx.metrics.ic()}, factor_cols=["factor"])
```

So the full pipeline is **`adapt` → `compute_forward_return` →
`evaluate`**. `fill_forward` is opt-in: leave it `False` (default) if
your panel is already clean, or set it `True` for raw OHLCV that may
contain sporadic missing or non-finite values.

---

## Choose metrics

Pass metric instances from `factrix.metrics` directly to the `metrics`
parameter of `evaluate()`; there is no central configuration object.

To learn how to choose the right metrics and configure them, see [Choosing a metric](../guides/choosing-metric.md) and [Concepts](concepts.md).

---

## Read the result

Read a metric's `value` and `p_value` together with `n_obs`, metadata, and
the enclosing result's warnings. A small p-value is not sufficient evidence
when the result also reports a persistence, overlap, clustering, or thin-sample
warning.

`EvaluationResult.to_dict()` returns the complete result as a flat,
JSON-friendly mapping. See [Reading results](../guides/reading-results.md) for
the field-by-field contract, [Warning codes](../reference/warning-codes.md) for
trigger conditions, and [Errors](../api/errors.md) for strict-mode failures.

---

## Next steps

You have the evaluation results for one or more factors. The common follow-ups:

| You want to… | Reach for | Guide / Reference |
|---|---|---|
| Screen candidate factors with false discovery rate (FDR) control | [`multi_factor.bhy(results)`](../api/bhy.md) — or `partial_conjunction` / `bhy_hierarchical` for nested structure | [multi_factor overview](../api/multi-factor.md) |
| Rank factors after screening | [`compare(results)`](../api/compare.md) — leaderboard with rank | — |
| Explore one metric across slices (sector / regime / universe / ADV bucket) | [`by_slice`](../api/by-slice.md) → `dict[str, EvaluationResult]` | [Slice analysis](../guides/slice-analysis.md) |
| Test whether slices differ statistically | [`slice_pairwise_test`](../api/slice-test.md) / [`slice_joint_test`](../api/slice-test.md) | [Slice analysis](../guides/slice-analysis.md) |

For function semantics and the input contract, see the [API reference landing](../api/index.md) and [Reading results](../guides/reading-results.md).
