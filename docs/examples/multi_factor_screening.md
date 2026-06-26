---
title: Multi-factor screening
---

Apply Benjamini-Hochberg-Yekutieli (BHY) false discovery rate (FDR) control to a batch of candidate factors.

Runnable notebook: [`examples/multi_factor_screening.ipynb`](https://github.com/awwesomeman/factrix/blob/main/examples/multi_factor_screening.ipynb).

## Factor type

This recipe uses `multi_factor.bhy(...)` over a list of `EvaluationResult` objects. Each result is produced by `evaluate()`.

The input list **is** the family. Each result must carry a unique `(factor, forward_periods)` identifier — the anti-shopping defense. The recommended path: name the factor column distinctly per candidate panel and pass it via `factor_cols=`; `evaluate` stamps the `factor` name onto the returned `EvaluationResult`.

Pass `expand_over=("forward_periods",)` to declare per-bucket independent step-ups (Benjamini & Bogomolov 2014 selective inference). Mixing `forward_periods` without `expand_over` emits a `RuntimeWarning`.

## Use this when

- You have $\ge 2$ candidate factors and want type-I error control under multiple testing.
- You prefer FDR control (BHY) over family-wise error (Bonferroni) — appropriate when discoveries can tolerate a known false-positive rate and power matters.

## 1. Setup

```python
import factrix as fx
import numpy as np
import polars as pl
from factrix.preprocess import compute_forward_return
from factrix.metrics import ic
```

## 2. Build a single-family batch

Five candidate factors, all evaluated under Newey-West IC with `forward_periods=5`.

We start from one ground-truth factor and add increasing noise to produce variants with varying signal strengths. Each variant is materialized under its own column name (`variant_0` … `variant_4`).

```python
raw = fx.datasets.make_cs_panel(
    n_assets=100,
    n_dates=500,
    ic_target=0.08,
    seed=2024,
)
panel = compute_forward_return(raw, forward_periods=5)

def variant_panel(
    base: pl.DataFrame, *, name: str, scale: float, seed: int
) -> pl.DataFrame:
    """Add IID noise on top of the ground-truth factor and store under a fresh column name."""
    rng = np.random.default_rng(seed)
    noisy = base["factor"].to_numpy() + scale * rng.standard_normal(base.height)
    return base.drop("factor").with_columns(pl.Series(name, noisy))

candidates = {
    f"variant_{i}": variant_panel(
        panel, name=f"variant_{i}", scale=0.5 + 0.3 * i, seed=100 + i
    )
    for i in range(5)
}

results = []
for name, p in candidates.items():
    res = fx.evaluate(
        p,
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
        factor_cols=[name],
        forward_periods=5,
    )
    results.extend(res.values())

for er in results:
    ic_res = er.metrics["ic"]
    print(f"  {er.factor:12s} p_value={ic_res.p_value:.4g}")
```

## 3. Apply BHY

The input list **is** the family. `bhy` runs one Benjamini-Yekutieli step-up over all results and returns a dict of `BhyResult` containers.

```python
fdr_results = fx.multi_factor.bhy(results, metrics=["ic"], q=0.05)
bhy_ic = fdr_results["ic"]

print(f"BHY survivors: {len(bhy_ic.survivors)} / {len(results)}")
for res, adj in zip(bhy_ic.survivors, bhy_ic.adj_p, strict=True):
    ic_res = res.metrics["ic"]
    print(f"  {res.factor:12s} p_value={ic_res.p_value:.4g}  adj_p={adj:.4g}")
```

Illustrative output:

```text
BHY survivors: 5 / 5
  variant_0    p_value=2.136e-39  adj_p=2.439e-38
  variant_1    p_value=4.052e-26  adj_p=2.313e-25
  variant_2    p_value=6.682e-22  adj_p=2.543e-21
  variant_3    p_value=3.987e-17  adj_p=1.138e-16
  variant_4    p_value=7.407e-14  adj_p=1.691e-13
```

## 4. Duplicate-identity defense

`evaluate()` requires unique factor columns across lists passed to `bhy()`. If two results carry the same factor name and horizon, `bhy()` raises `UserInputError` to prevent duplicate hypotheses from collapsing FDR controls.

```python
from factrix.metrics import fm_beta

# Collides because both evaluate on the same "factor" name
unstamped = []
unstamped.extend(fx.evaluate(panel, metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)}, factor_cols=["factor"], forward_periods=5).values())
unstamped.extend(fx.evaluate(panel, metrics={"fm": fm_beta()}, factor_cols=["factor"], forward_periods=5).values())

try:
    fx.multi_factor.bhy(unstamped, metrics=["ic"], q=0.05)
except fx.UserInputError as exc:
    print("UserInputError raised as expected:")
    print(str(exc))
```

## 5. Where to go next

For the broader structural details, see [Guides § Panel vs timeseries](../guides/panel-timeseries.md) and [Large-scale evaluation and memory protection](../guides/large-scale-evaluation.md).
