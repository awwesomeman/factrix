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
positive on average.

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

If the universe is small, reduce the number of groups. With only five assets,
`n_groups=5` means one asset per bucket, so the spread is dominated by
individual names. `n_groups=2` is the coarsest useful long-short split.

```python
series = compute_spread_series(
    panel,
    factor_cols=["factor"],
    forward_periods=5,
    n_groups=2,
)["factor"]
```

When `n_groups=2` is still too thin, treat the result as a fragile
small-cross-section diagnostic rather than a portfolio claim.

## Multi-factor screens

After evaluating many candidate factors, pass the `EvaluationResult` values to
BHY false discovery rate control:

```python
results = fx.evaluate(
    panel,
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
