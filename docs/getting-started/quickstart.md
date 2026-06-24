---
title: Quickstart
---

!!! warning "`forward_periods` counts rows, not calendar time"
    factrix is frequency-agnostic — it only shifts row indices.
    `forward_periods=5` on a daily panel means 5 trading days; on a
    weekly panel, 5 weeks. The caller is responsible for ensuring the
    panel is sorted per asset and has regular time spacing.

## 30-second smoke test

```python
import factrix as fx
from factrix.preprocess import compute_forward_return
from factrix.metrics import ic

# 1. Generate synthetic panel data and compute forward returns
raw   = fx.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
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

## Research question → metric mapping

In `factrix`, rather than constructing a central config object, you pass metric instances imported from `factrix.metrics` directly to the `metrics` parameter of `fx.evaluate()`.

To learn how to choose the right metrics and configure them, see [Choosing a metric](../guides/choosing-metric.md) and [Concepts](concepts.md).

---

## `EvaluationResult.to_dict()` and warnings

Calling `.to_dict()` on the returned `EvaluationResult` returns a flat, JSON-friendly representation of the results, including evaluation metadata, statistics, and any active warnings:

```json
{
  "factor": "factor",
  "cell": {
    "scope": "individual",
    "density": "dense",
    "structure": "panel"
  },
  "forward_periods": 5,
  "n_periods": 494,
  "n_pairs": 49400,
  "n_assets": 100,
  "context": {},
  "metrics": {
    "ic": {
      "value": 0.0722,
      "p_value": 2.129e-40,
      "stat": 14.60,
      "n_obs": 494,
      "metadata": {
        "n_periods": 494,
        "p_value": 2.129e-40,
        "stat_type": "t",
        "h0": "mu=0",
        "method": "Newey-West HAC t-test on overlapping IC series",
        "newey_west_lags": 5,
        "forward_periods": 5,
        "tie_ratio": 0.0
      }
    }
  },
  "warnings": [],
  "plan": "1. compute_ic [batchable]\n2. ic [per-factor] requires=ic_df"
}
```

The most common warnings include:

- `UNRELIABLE_SE_SHORT_PERIODS` — $20 \le T < 30$; Newey-West (NW) HAC SE is unstable. (Falling below the metric's hard sample floor raises `InsufficientSampleError` under `strict=True`; the exact floor is metric-specific).
- `PERSISTENT_REGRESSOR` — factor augmented Dickey-Fuller (ADF) $p$-value > 0.10.
- `EVENT_WINDOW_OVERLAP` — event windows overlap on the same asset.
- `SERIAL_CORRELATION_DETECTED` — Ljung-Box $p$-value < 0.05 on residuals.

For the full enum and the trigger conditions for each `WarningCode`, see [Reference § Warning codes](../reference/warning-codes.md).

For exception classes (`InsufficientSampleError`, `IncompatibleAxisError`, `UserInputError`, ...) and their catch patterns, see [Errors](../api/errors.md).

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
