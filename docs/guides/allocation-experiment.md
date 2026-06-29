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
positive_rate = float(np.mean(spread > 0))

print(annualized_spread, annualized_sharpe, positive_rate)
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
right?" and with `directional_pair_accuracy` when the question is "does the
higher-scored asset usually beat the lower-scored asset on the same date?"

```python
from factrix.metrics import directional_hit_rate, directional_pair_accuracy

hit = directional_hit_rate(panel, forward_periods=5)
print(hit.value, hit.p_value, hit.metadata["p_expected"])

ordering = directional_pair_accuracy(panel, forward_periods=5)
print(ordering.value, ordering.p_value, ordering.metadata["n_pairs"])
```

For 5-20 asset panels, read IC / FM output as the canonical inference layer when
available, but expect `WarningCode.FEW_ASSETS` on thin cross-sections. The
small-N spread and directional diagnostics are supplementary; they do not
replace the canonical p-value family for screening.

## Country-level momentum

Country equity, rates, FX, or commodity momentum is usually an
asset-specific dense signal: each asset has its own score on each date. Keep it
in the `Individual × Dense` cell and read a small set of complementary
diagnostics:

```python
from factrix.metrics import directional_hit_rate, directional_pair_accuracy, fm_beta, k_spread

country_panel = panel.rename({"factor": "country_momentum"})

country = fx.evaluate(
    country_panel,
    metrics={
        "fm": fm_beta(),
        "spread": k_spread(k=2),
        "direction": directional_hit_rate(),
        "ordering": directional_pair_accuracy(),
    },
    factor_cols=["country_momentum"],
    forward_periods=5,
    strict=False,
)

result = country["country_momentum"]
print(result.metrics["fm"].value, result.metrics["fm"].p_value)
print(result.metrics["spread"].value, result.metrics["spread"].p_value)
print(result.metrics["direction"].value, result.metrics["direction"].p_value)
print(result.metrics["ordering"].value, result.metrics["ordering"].p_value)
```

`fm_beta` keeps the economic unit of the signal, `k_spread` gives a
fixed-count long-short read when the asset universe is small, and
`directional_hit_rate` checks whether the signal gets the sign right.
`directional_pair_accuracy` checks same-date rank ordering without turning
within-date pairs into a naive p-value. Treat the set as factor diagnostics, not
as a completed allocation rule.

## Common macro factors and heterogeneous betas

Common factors such as growth surprises, inflation shocks, VIX, DXY, or policy
expectations are shared across assets on a date. A single average beta can hide
rotation value: equities may load positively while duration or gold loads
negatively, leaving `common_beta` close to zero even though the factor separates the
asset classes.

Use the beta profile before reading the average-beta test:

```python
import polars as pl
from factrix.metrics import common_beta_r_squared, common_beta, common_beta_profile
from factrix.metrics import common_beta_sign_consistency
from factrix.metrics import common_quantile_spread
from factrix.metrics.common_beta import compute_common_betas

macro = (
    panel.select("date")
    .unique()
    .sort("date")
    .with_row_index("t")
    .with_columns((pl.col("t").cast(pl.Float64) / 20.0).sin().alias("macro_growth"))
    .select("date", "macro_growth")
)
macro_panel = panel.join(macro, on="date")
betas_df = compute_common_betas(macro_panel, factor_cols=["macro_growth"])["macro_growth"]

avg_beta = common_beta(betas_df)
profile = common_beta_profile(betas_df)
r2 = common_beta_r_squared(betas_df)
signs = common_beta_sign_consistency(betas_df)
spread = common_quantile_spread(
    macro_panel,
    factor_col="macro_growth",
    forward_periods=5,
    n_groups=3,
)

print(avg_beta.value, avg_beta.p_value)
print(profile.value, profile.metadata["n_positive_beta"], profile.metadata["n_negative_beta"])
print(r2.value, signs.value, signs.metadata["fraction_positive"])
print(spread.value, spread.p_value)
```

Read the pieces together:

| Question | Diagnostic |
|---|---|
| Is the average exposure different from zero? | `common_beta.value`, `common_beta.p_value` |
| Are asset betas dispersed enough for rotation? | `common_beta_profile.metadata["beta_std"]`, `compute_common_betas(...)[factor]["beta"]` |
| Does one sign dominate, or do signs split by asset class? | `common_beta_profile.metadata["n_positive_beta"]`, `metadata["n_negative_beta"]`, `common_beta_sign_consistency.value` |
| Does the factor explain individual asset returns? | `common_beta_r_squared.value`, `metadata["median_r_squared"]` |
| Is the common factor nonlinear or extreme-state driven? | `common_quantile_spread` |

When the asset-class split matters, join your own labels to the beta table and
summarize signs and dispersion by group. Keep missing labels explicit:

```python
asset_groups = pl.DataFrame(
    {
        "asset_id": ["A00", "A01", "A02"],
        "asset_group": ["equity", "duration", "commodity"],
    }
)

group_profile = (
    betas_df.join(asset_groups, on="asset_id", how="left")
    .with_columns(pl.col("asset_group").fill_null("__missing_group__"))
    .group_by("asset_group")
    .agg(
        pl.len().alias("n_assets"),
        (pl.col("beta") > 0).sum().alias("n_positive_beta"),
        (pl.col("beta") < 0).sum().alias("n_negative_beta"),
        pl.col("beta").std().alias("beta_std"),
        pl.col("beta").abs().mean().alias("abs_beta_mean"),
    )
)
```

This remains a factor diagnostic. If the beta vector suggests a long-equity /
short-duration rotation, convert that insight into weights, risk budgets,
rebalance rules, and transaction-cost assumptions in a downstream portfolio or
backtest layer.

## Mapped policy event factors

Sparse event factors use their non-zero sign as the expected return direction.
That is not always the raw event type. A central-bank hike can be bearish for
duration-sensitive assets, bullish for a currency basket, and neutral for some
commodities. Encode the factor after that mapping, then run the sparse metrics.

```python
from factrix.metrics import caar, clustering_hhi, event_hit_rate, profit_factor
from factrix.metrics.caar import compute_caar

dates = (
    panel.select("date")
    .unique()
    .sort("date")
    .with_row_index("t")
    .with_columns(
        pl.when(pl.col("t") % 5 == 0)
        .then(1.0)      # raw hike
        .when(pl.col("t") % 11 == 0)
        .then(-1.0)     # raw cut
        .otherwise(0.0)
        .alias("policy_event_type")
    )
    .select("date", "policy_event_type")
)
asset_map = (
    panel.select("asset_id")
    .unique()
    .sort("asset_id")
    .with_row_index("i")
    .with_columns(
        pl.when(pl.col("i") % 3 == 0)
        .then(-1.0)     # hike expected to hurt this asset group
        .otherwise(1.0)
        .alias("policy_expected_direction")
    )
    .select("asset_id", "policy_expected_direction")
)

policy_panel = (
    panel.join(dates, on="date")
    .join(asset_map, on="asset_id")
    .with_columns(
        (pl.col("policy_event_type") * pl.col("policy_expected_direction"))
        .alias("policy_signal")
    )
)

hit = event_hit_rate(policy_panel, factor_col="policy_signal")
pf = profit_factor(policy_panel, factor_col="policy_signal")
caar_series = compute_caar(policy_panel, factor_col="policy_signal")
caar_out = caar(caar_series, forward_periods=5)
crowding = clustering_hhi(policy_panel, factor_col="policy_signal")

print(hit.value, hit.p_value)
print(pf.value)
print(
    caar_out.metadata["total_events"],
    caar_out.metadata["n_event_periods"],
    caar_out.metadata["n_event_periods_sampled"],
)
print(crowding.value, crowding.metadata["hhi_normalized"])
```

Read the counts in that order:

| Count | Meaning |
|---|---|
| `total_events` | Non-zero `(date, asset_id)` event rows behind the study |
| `n_event_periods` | Distinct event dates after same-date events are collapsed into one CAAR observation |
| `n_event_periods_sampled` | Event dates kept by the non-overlap sampler used for the `caar` t-test |

`event_hit_rate` and `profit_factor` use `sign(policy_signal)`, so they answer
whether the mapped direction was right. `caar` preserves magnitude in
`forward_return * policy_signal`; use a clean `{0, -1, +1}` signal when you want
signed CAAR rather than magnitude-weighted CAAR.

## Regime workflow

factrix does not infer regimes. Attach labels upstream, then choose the
descriptive or inferential path deliberately:

```python
from factrix import by_slice, slice_period_pairwise_test

regime_labels = macro.with_columns(
    pl.when(pl.col("macro_growth") >= 0)
    .then(pl.lit("growth_up"))
    .otherwise(pl.lit("growth_down"))
    .alias("regime")
).select("date", "regime")

regime_panel = policy_panel.join(regime_labels, on="date")

per_regime = by_slice(
    regime_panel,
    caar(),
    by="regime",
    factor_col="policy_signal",
    strict=False,
)

pairs = slice_period_pairwise_test(
    regime_panel,
    caar(),
    by="regime",
    factor_col="policy_signal",
    rng_seed=7,
)
print(pairs)
```

Use `by_slice` for per-regime result tables. Use
`slice_period_pairwise_test` / `slice_period_joint_test` when regimes are
date-disjoint and you need a calibrated cross-regime contrast. Do not compare
two per-regime p-values and call that a regime difference.

Date-axis slicing can truncate history for metrics that look across dates
(`common_beta`, event windows, rolling/OOS diagnostics). `by_slice` emits
`WarningCode.SLICE_BOUNDARY_TRUNCATION` for those cases. For pure per-date
metrics such as IC / FM, a date-axis split is closer to a normal period
decomposition.

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
