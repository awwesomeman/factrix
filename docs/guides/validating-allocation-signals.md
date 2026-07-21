---
title: Validating allocation signals
---

# Validating allocation signals

factrix validates whether a signal carries statistical edge before portfolio
construction. It does not optimize weights, simulate execution, or produce a
production backtest. Use this guide to choose evidence for a small allocation
universe without crossing that boundary.

## Match the metric to the signal

Run `inspect_data` first and evaluate different cells in separate batches. An
asset-specific score, a macro value broadcast across assets, and an event flag
do not share an estimator.

| Signal and question | First-pass evidence | Follow-up diagnostics |
|---|---|---|
| Individual dense: does the score rank future returns? | `ic` | `ic_ir`, `directional_pair_accuracy`, `k_spread`, `monotonicity` |
| Individual dense: what is the premium per exposure unit? | `fm_beta` | `pooled_beta`, `fm_beta_sign_consistency` |
| Common dense: is average asset exposure different from zero? | `common_beta` | `common_beta_profile`, `common_beta_r_squared`, `common_beta_sign_consistency` |
| Sparse event: is the event-time effect different from zero? | `caar` | `bmp_z`, `event_hit_rate`, `profit_factor`, `clustering_hhi` |
| Dense sign forecast: does the signal predict absolute up/down direction? | `directional_hit_rate` | Sign-balance metadata and stability checks |

The research question determines the first-pass metric. In particular,
`directional_hit_rate` tests absolute sign prediction; it does not replace rank
IC when the allocation rule ranks assets against one another.

## Small-universe workflow

A small cross-section reduces power; it does not change the estimand. IC and FM
remain the first-pass inference for ranking and exposure-premium questions when
their sample requirements are met. Read the supplementary metrics by role:

| Metric | What it adds | Selection status |
|---|---|---|
| `ic_ir` | Time-series stability of IC | Descriptive; `p_value=None` |
| `directional_pair_accuracy` | Same-date pair ordering | Descriptive; dependent pairs are not given a naive p-value |
| `directional_hit_rate` | Absolute sign skill | Inferential only for a sign-prediction objective |
| `k_spread` / `quantile_spread` | Long-short economic magnitude | Supplementary; tiny legs are fragile |
| `monotonicity` | Shape between the tails | Supplementary; do not infer it from top-minus-bottom alone |

This example keeps the warning audit trail while quieting a sample regime that
is expected by design:

```python
import factrix as fx
from factrix.metrics import directional_hit_rate, directional_pair_accuracy, ic
from factrix.metrics import ic_ir, k_spread
from factrix.preprocess import compute_forward_return

raw = fx.datasets.make_cs_panel(n_assets=12, n_dates=252, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

inspection = fx.inspect_data(panel, factor_cols=["factor"])
print(inspection.properties)

results = fx.evaluate(
    panel,
    metrics={
        "ic": ic(inference=fx.inference.NEWEY_WEST),
        "ic_ir": ic_ir(),
        "spread": k_spread(k=2),
        "direction": directional_hit_rate(),
        "ordering": directional_pair_accuracy(),
    },
    factor_cols=["factor"],
    strict=False,
    expected_warnings=("few_assets",),
)

result = results["factor"]
print(result.metrics["ic"].p_value)
print(result.metrics["ic_ir"].p_value)       # None: descriptive
print(result.metrics["ordering"].p_value)    # None: descriptive
print(result.unexpected_warnings)             # alerts not declared above
```

`expected_warnings` marks matching records as expected and quiets their
repeated `UserWarning` echo. It does not remove records, alter p-values, or
change an estimator. Use `result.warnings` for the full audit trail and
`result.unexpected_warnings` for the alert view.

## Keep cost units aligned

The built-in tradability helpers describe one specific proxy:

| Output | Contract |
|---|---|
| `notional_turnover` | Membership churn in an equal-weight top/bottom quantile portfolio |
| `breakeven_cost` | Single-leg cost that reduces the matching gross spread to zero |
| `net_spread` | Matching gross spread after that cost assumption |
| `rank_turnover` | Rank stability only; not a position-turnover or cost input |

Use the same `n_groups` and `forward_periods` for the spread and turnover.
These helpers do not price a long-only or custom-weight allocation. Compute
turnover, slippage, market impact, borrow, and capacity from the actual target
weights downstream. See [Tradability](../api/metrics/tradability.md) for units
and the [stock factor example](../examples/stock_factor_evaluation.md) for an
executable gross-to-net workflow.

## Declare selection families

Apply multiple-testing control to the hypotheses the research process can
select, not to every descriptive column in a report.

| Research decision | Family declaration |
|---|---|
| Select the best factor or horizon from a grid | Run `evaluate_horizons`, then call `bhy` without `expand_over`; keep factor × horizon hypotheses pooled |
| Report predeclared horizon-specific screens without comparing them | `bhy(..., expand_over=("forward_periods",))` |
| Require a factor to pass at least k of m horizons | `partial_conjunction(..., min_pass=k, expand_over=("forward_periods",))` |
| Select factor × metric cells from one family | `bhy_across_metrics(...)`; the survivor unit remains a cell hypothesis |
| Require a factor to pass at least k of m metrics | `partial_conjunction_across_metrics(..., min_pass=k)` |

Do not deduplicate pooled cell survivors into factors and claim factor-level
FDR; an any-metric-pass factor promotion is a different procedure. Horizon
suitability comes from the effective sample, overlap, and warning records—not
a universal list of allowed horizon numbers.

With `strict=False`, data-shortage placeholders remain visible. `bhy` excludes
outputs whose reason starts with `insufficient_` from the active test count and
leaves their adjusted p-value empty; other missing or invalid p-values still
fail loudly. See [Multi-horizon evaluation](../api/multi-horizon.md) and
[Multi-factor screening](../api/multi-factor.md) for the APIs.

## Add robustness after the primary screen

Keep each follow-up tied to the question it answers:

| Question | Tool | Boundary |
|---|---|---|
| Does performance differ across precomputed regimes? | `by_slice`, `slice_period_pairwise_test`, `slice_period_joint_test` | Attach lookahead-safe labels upstream; two separate p-values do not test their difference |
| Does a candidate add information beyond a fixed baseline? | `spanning_alpha` | Supplementary fixed-base comparison, not stepwise post-selection inference |
| Is the slope robust to pooled panel dependence? | `pooled_beta` | Supports clustered or Driscoll-Kraay SE; it is not a two-way fixed-effects model |

See [Slice analysis](slice-analysis.md),
[`spanning_alpha`](../api/metrics/spanning.md), and
[`pooled_beta`](../api/metrics/fm_beta.md) for the complete contracts.

## Continue with executable examples

Guides explain what to choose; notebooks show one runnable research path:

- [Stock factor evaluation](../examples/stock_factor_evaluation.md):
  preprocessing, neutralization, coverage, and gross-to-net feasibility.
- [Multi-factor screening](../examples/multi_factor_screening.md): BHY screening,
  hypothesis identity, and redundancy checks.
