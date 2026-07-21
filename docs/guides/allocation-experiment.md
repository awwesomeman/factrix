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

When a research sheet mixes asset-specific signals (for example trend or carry
scores that differ by asset) with common macro signals (for example the same
growth surprise value broadcast to every asset on a date), preflight and
evaluate them in separate batches. `inspect_data(..., factor_cols=[...])` bases
the displayed applicability table on one detected factor cell and warns when
the requested columns mix scope or density; the clean workflow is one batch for
`Individual x Dense` metrics such as `ic` / `fm_beta`, and a separate batch for
`Common x Dense` metrics such as `common_beta` and its diagnostics.

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

Add the built-in cost helpers only when their units match that proxy:

```python
from factrix.metrics import breakeven_cost, net_spread, notional_turnover

gross_spread = float(series["spread"].drop_nulls().mean())
turnover = notional_turnover(
    panel, factor_col="composite_signal", n_groups=5, forward_periods=5
)
breakeven = breakeven_cost(gross_spread, turnover.value, forward_periods=5)
after_cost = net_spread(
    gross_spread, turnover.value, estimated_cost_bps=10, forward_periods=5
)
```

`notional_turnover` measures membership churn in the same equal-weight
top/bottom construction; `breakeven_cost` and `net_spread` assume that turnover
and a single-leg cost. They do not price a long-only or custom-weight portfolio.
For those portfolios, compute turnover, slippage, market impact, and capacity
from the actual target weights downstream. `rank_turnover` measures rank
stability and is not a cost input. See [Tradability](../api/metrics/tradability.md)
for the formulas and units.

## Small universes

Small cross-sections reduce power; they do not change the research question.
Choose the first-pass metric by estimand, then add diagnostics:

| Question | Evidence | Selection role |
|---|---|---|
| Does the signal rank future returns? | `ic(inference=fx.inference.NEWEY_WEST)` | First-pass inference for a ranking strategy |
| Is return linear in exposure units? | `fm_beta()` | First-pass inference for an exposure-premium strategy |
| Does the signal predict absolute up/down direction? | `directional_hit_rate()` | Inferential only when sign prediction is the objective; not a substitute for rank IC |
| Does the higher-scored asset beat the lower-scored asset? | `directional_pair_accuracy()` | Descriptive; same-date pairs are dependent, so `p_value=None` |
| Is IC stable through time? | `ic_ir()` | Descriptive; use `ic` for mean-IC inference |
| Is the payoff shape economically coherent? | `k_spread()`, `quantile_spread()`, `monotonicity()` | Supplementary; tiny legs make spread and shape fragile |

For spread diagnostics, reduce `n_groups` or state the leg size directly with
`k_spread`. With five assets, five quantiles leave one name per bucket; that is
an individual-name diagnostic, not diversified portfolio evidence.

```python
from factrix.metrics import directional_hit_rate, directional_pair_accuracy, ic
from factrix.metrics import ic_ir, k_spread

small_assets = panel["asset_id"].unique().sort().head(6).to_list()
small_panel = panel.filter(pl.col("asset_id").is_in(small_assets))

small = fx.evaluate(
    small_panel,
    metrics={
        "ic": ic(inference=fx.inference.NEWEY_WEST),
        "ic_ir": ic_ir(),
        "spread": k_spread(k=2),
        "direction": directional_hit_rate(),
        "ordering": directional_pair_accuracy(),
    },
    factor_cols=["factor"],
    forward_periods=5,
    strict=False,
    expected_warnings=("few_assets",),
)

result = small["factor"]
print(result.metrics["ic"].p_value)
print(result.metrics["ic_ir"].p_value)       # None: descriptive
print(result.metrics["ordering"].p_value)    # None: descriptive
print(result.unexpected_warnings)             # alerts not declared above
```

`expected_warnings` marks matching records as expected and quiets their repeated
`UserWarning` echo. It does not remove records, alter p-values, or change an
estimator; inspect the full audit trail through `result.warnings`. Leave other
codes undeclared so `result.unexpected_warnings` remains an actionable alert
view.

After the primary screen, use [slice analysis](slice-analysis.md),
[`spanning_alpha`](../api/metrics/spanning.md), and
[`pooled_beta`](../api/metrics/fm_beta.md) as follow-up robustness evidence.
Do not turn several supplementary reads into an unregistered "any-pass" gate.

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
        (
            pl.col("beta").filter(pl.col("beta") > 0).mean()
            - pl.col("beta").filter(pl.col("beta") < 0).mean()
        ).alias("positive_minus_negative_beta_spread"),
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

## Multiple testing and horizons

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

Declare the family from the decision the research process can make:

| Research decision | Family declaration |
|---|---|
| Select the best factor or horizon from the full grid | Run `evaluate_horizons`, then call `bhy` without `expand_over`; all factor × horizon hypotheses stay pooled |
| Report predeclared horizon-specific screens without comparing them | `bhy(..., expand_over=("forward_periods",))` |
| Require a factor to pass at least k of m horizons | `partial_conjunction(..., min_pass=k, expand_over=("forward_periods",))` |

Do not split families by horizon and later choose the best bucket: that leaves
the cross-horizon selection uncorrected. Horizon suitability comes from the
effective sample, overlap, and warning records—not from a universal list of
allowed horizon numbers. See [Multi-horizon evaluation](../api/multi-horizon.md)
and [Multi-factor screening](../api/multi-factor.md) for the APIs.

`strict=False` is useful for sensitivity grids where some small-universe cells
may be too thin for a metric. Data-shortage placeholders remain in `evaluate`
output for inspection. `bhy` excludes outputs whose reason starts with
`insufficient_` from the active test count and leaves their adjusted p-value
empty; other missing or invalid p-values still fail loudly.

## Windows console output

Polars may render tables with box-drawing characters. On some Windows consoles
using legacy encodings, that can fail with a `UnicodeEncodeError`. If a script
prints Polars tables, set ASCII formatting near the top:

```python
import polars as pl

pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
```

This only affects display formatting; it does not change the data.
