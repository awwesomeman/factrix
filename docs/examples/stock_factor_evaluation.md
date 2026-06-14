---
title: Stock factor evaluation
---

Evaluate whether a per-stock factor (one value per `(date, asset_id)`) carries cross-sectional return predictability.

Runnable notebook: [`examples/stock_factor_evaluation.ipynb`](https://github.com/awwesomeman/factrix/blob/main/examples/stock_factor_evaluation.ipynb).

## Factor type

This recipe uses the `ic` metric (with Newey-West HAC inference), which operates under `FactorScope.INDIVIDUAL`, `FactorDensity.DENSE`, and `DataStructure.PANEL`.

Procedure: per-date Spearman correlation between factor and forward return, aggregated to a Newey-West (NW) HAC t-statistic on the mean.

Null hypothesis $\mathbb{E}[\text{IC}] = 0$ — the factor has no rank-based predictive ordering of forward returns across assets, on average across dates.

## Use this when

- Factor varies across assets at each date (per-stock signal, e.g. momentum, value, quality).
- Cross-section is wide ($N \ge 30$ for clean inference).
- Time series is at least 30 periods ($T < 20$ is hard-blocked).

## 1. Setup

```python
import factrix as fx
from factrix.preprocess import compute_forward_return
from factrix.metrics import ic
```

## 2. Synthesise a cross-sectional panel

`make_cs_panel` produces a canonical panel with a target IC built in.

```python
raw = fx.datasets.make_cs_panel(
    n_assets=100,
    n_dates=500,
    ic_target=0.08,
    seed=2024,
)
panel = compute_forward_return(raw, forward_periods=5)
print(f"panel shape={panel.shape}  N={panel['asset_id'].n_unique()}")
```

## 3. Evaluate

```python
results = fx.evaluate(
    panel,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor"],
    forward_periods=5,
)

res = results["factor"]
ic_res = res.metrics["ic"]

print(f"factor       = {res.factor}")
print(f"cell         = {res.cell}")
print(f"ic_mean      = {ic_res.value:+.4f}")
print(f"ic_p_value   = {ic_res.p_value:.4g}")
```

Illustrative output:

```text
factor       = factor
cell         = (individual, dense, panel)
ic_mean      = +0.0722
ic_p_value   = 2.129e-40
```

## 4. Inspect the result dictionary

Call `.to_dict()` on the `EvaluationResult` to obtain a JSON-friendly nested representation of the results.

```python
import json

print(json.dumps(res.to_dict(), indent=2))
```

Illustrative output:

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
