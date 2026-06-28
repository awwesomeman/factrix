---
title: Allocation-style experiments
---

# Allocation-style experiments

factrix answers the research question before portfolio construction:

> Does this factor carry predictive edge?

It does not optimize weights, model slippage, rebalance a live book, or
produce a full backtest. Those belong downstream. Still, it is often useful
to run an allocation-style proxy while screening factors: "if I rank assets by
this signal, does a simple long-short allocation earn a positive spread?"

This guide shows the boundary.

## Use factrix for the signal test

Start with the standard cross-sectional panel and evaluate the factor:

```python
import factrix as fx
from factrix.metrics import ic, quantile_spread
from factrix.preprocess import compute_forward_return

raw = fx.datasets.make_cs_panel(n_assets=50, n_dates=252, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

results = fx.evaluate(
    panel,
    metrics={
        "ic": ic(inference=fx.inference.NEWEY_WEST),
        "spread": quantile_spread(n_groups=5),
    },
    factor_cols=["factor"],
    forward_periods=5,
)

result = results["factor"]
print(result.metrics["ic"].value, result.metrics["ic"].p_value)
print(result.metrics["spread"].value, result.metrics["spread"].p_value)
```

`ic` asks whether the rank relation is statistically different from zero.
`quantile_spread` asks whether a simple top-minus-bottom portfolio spread is
positive on average. In small allocation universes, add `k_spread`: it keeps
each leg at a fixed name count instead of forcing unstable quantile buckets.

## Build a simple allocation proxy

For a custom composite signal, create the signal column upstream and use
`compute_spread_series` to inspect the per-rebalance long-short path:

```python
import numpy as np
import polars as pl
from factrix.metrics.quantile import compute_spread_series

panel = panel.with_columns(
    (
        0.5 * pl.col("factor")
        + 0.5 * pl.col("factor").rank().over("date")
    ).alias("composite_signal")
)

series = compute_spread_series(
    panel,
    factor_cols=["composite_signal"],
    forward_periods=5,
    n_groups=5,
)["composite_signal"]

spread = series["spread"].drop_nulls().to_numpy()
periods_per_year = 252 / 5

annualized_spread = float(np.mean(spread) * periods_per_year)
annualized_sharpe = float(
    np.mean(spread) / np.std(spread, ddof=1) * np.sqrt(periods_per_year)
)
hit_rate = float(np.mean(spread > 0))

print(annualized_spread, annualized_sharpe, hit_rate)
```

This is not a production backtest. It is a diagnostic allocation proxy:
equal-weight top bucket, equal-weight bottom bucket, no turnover costs, no
borrow constraints, no capacity model, and no weight optimization.

## Small universes

If the universe is small, reduce the number of groups or use the fixed-count
spread. With only five assets, `n_groups=5` means one asset per bucket, so the
spread is dominated by individual names. `n_groups=2` is the coarsest useful
long-short split, while `k_spread(k=1)` / `k_spread(k=2)` states the leg size
directly.

```python
from factrix.metrics import k_spread

series = compute_spread_series(
    panel,
    factor_cols=["factor"],
    forward_periods=5,
    n_groups=2,
)["factor"]

spread_metric = k_spread(panel, forward_periods=5, k=2)
print(spread_metric.value, spread_metric.p_value, spread_metric.metadata["method"])
```

When even `k=1` is too noisy, treat the result as a fragile
small-cross-section diagnostic rather than a portfolio claim. Pair it with
`directional_hit_rate` when the question is "does the signal get the direction
right?" rather than "does a long-short spread clear zero?"

```python
from factrix.metrics import directional_hit_rate

hit = directional_hit_rate(panel, forward_periods=5)
print(hit.value, hit.p_value, hit.metadata["p_expected"])
```

For 5-20 asset panels, read IC / FM output as the canonical inference layer when
available, but expect `WarningCode.FEW_ASSETS` on thin cross-sections. The
small-N spread and directional diagnostics are supplementary; they do not
replace the canonical p-value family for screening.

## Common macro factors and heterogeneous betas

Common factors such as growth surprises, inflation shocks, VIX, DXY, or policy
expectations are shared across assets on a date. A single average beta can hide
rotation value: equities may load positively while duration or gold loads
negatively, leaving `ts_beta` close to zero even though the factor separates the
asset classes.

Use the beta profile before reading the average-beta test:

```python
import polars as pl
from factrix.metrics import mean_r_squared, ts_beta, ts_beta_sign_consistency
from factrix.metrics import ts_quantile_spread
from factrix.metrics.ts_beta import compute_ts_betas

macro = (
    panel.select("date")
    .unique()
    .sort("date")
    .with_row_index("t")
    .with_columns((pl.col("t").cast(pl.Float64) / 20.0).sin().alias("macro_growth"))
    .select("date", "macro_growth")
)
macro_panel = panel.join(macro, on="date")
betas_df = compute_ts_betas(macro_panel, factor_cols=["macro_growth"])["macro_growth"]

avg_beta = ts_beta(betas_df)
r2 = mean_r_squared(betas_df)
signs = ts_beta_sign_consistency(betas_df)
spread = ts_quantile_spread(
    macro_panel,
    factor_col="macro_growth",
    forward_periods=5,
    n_groups=3,
)

print(avg_beta.value, avg_beta.p_value)
print(avg_beta.metadata["beta_std"], avg_beta.metadata["median_beta"])
print(r2.value, signs.value, signs.metadata["fraction_positive"])
print(spread.value, spread.p_value)
```

Read the pieces together:

| Question | Diagnostic |
|---|---|
| Is the average exposure different from zero? | `ts_beta.value`, `ts_beta.p_value` |
| Are asset betas dispersed enough for rotation? | `ts_beta.metadata["beta_std"]`, `compute_ts_betas(...)[factor]["beta"]` |
| Does one sign dominate, or do signs split by asset class? | `ts_beta_sign_consistency.value`, `metadata["fraction_positive"]` |
| Does the factor explain individual asset returns? | `mean_r_squared.value`, `metadata["median_r_squared"]` |
| Is the common factor nonlinear or extreme-state driven? | `ts_quantile_spread` |

This remains a factor diagnostic. If the beta vector suggests a long-equity /
short-duration rotation, convert that insight into weights, risk budgets,
rebalance rules, and transaction-cost assumptions in a downstream portfolio or
backtest layer.

## Multi-factor screens

After evaluating many candidate factors, pass the `EvaluationResult` values to
BHY false discovery rate control. Build a panel with several factor columns —
`make_multi_factor_panel` emits `factor_0000`, `factor_0001`, ...:

```python
multi = compute_forward_return(
    fx.datasets.make_multi_factor_panel(
        n_assets=50, n_dates=252, n_factors=3, seed=2024
    ),
    forward_periods=5,
)

results = fx.evaluate(
    multi,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor_0000", "factor_0001", "factor_0002"],
    forward_periods=5,
    strict=False,
)

bhy_ic = fx.multi_factor.bhy(list(results.values()), metrics=["ic"], q=0.05)["ic"]
print([r.factor for r in bhy_ic.survivors])
```

`strict=False` is useful for sensitivity grids where some small-universe cells
may be too thin for a metric. Data-shortage placeholders are kept in
`evaluate` output for inspection, while `bhy` excludes `insufficient_*`
placeholders from the tested family.

## Windows console output

Polars may render tables with box-drawing characters. On some Windows consoles
using legacy encodings, that can fail with a `UnicodeEncodeError`. If a script
prints Polars tables, set ASCII formatting near the top:

```python
import polars as pl

pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
```

This only affects display formatting; it does not change the data.
