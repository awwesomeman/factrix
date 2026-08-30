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

## Separate promotion evidence from diagnostics

One entry point for the whole path: the research question, the evidence that
can promote a candidate, the diagnostics and robustness checks that cannot,
and what stays outside factrix. Every cell links to the page that owns the
contract; none of the derivations are restated here.

| Research question / stage | Primary evidence | Supplementary or robustness evidence | Do not infer |
|---|---|---|---|
| Cross-sectional asset ranking | [`ic`](../api/metrics/ic.md) or [`fm_beta`](../api/metrics/fm_beta.md) over a [declared hypothesis family](#declare-selection-families), read against the [persistent-series caveat](../reference/statistical-methods.md#persistent-per-period-series-no-hac-or-bootstrap-path-is-calibrated) | [`k_spread`](../api/metrics/k_spread.md), [`monotonicity`](../api/metrics/monotonicity.md), [`directional_pair_accuracy`](../api/metrics/directional_pair_accuracy.md), [`ic_ir`](../api/metrics/ic.md) stability | That one diagnostic passing promotes the candidate |
| Single-asset time-series predictability | [`predictive_beta`](../api/metrics/predictive_beta.md) | [Persistence](../reference/statistical-methods.md#4-persistence-diagnostics-under-near-unit-root-predictors) and sample diagnostics, `PERSISTENT_REGRESSOR` in [warning codes](../reference/warning-codes.md) | That the slope is cross-sectional ranking evidence |
| Regime robustness | [`by_slice`](../api/by-slice.md), [`slice_period_pairwise_test`](../api/slice-test.md) contrasts, effect sizes | [`slice_period_joint_test`](../api/slice-test.md) where it is calibrated — `K = 2`, or `K >= 3` on longer slices | That a declared `short_slice_joint_test` corrected the p-value, or that a short-slice joint p is an admission gate |
| Candidate redundancy | Correlation and fixed-base [`spanning_alpha`](../api/metrics/spanning.md) | Economic reading of the residual alpha | That `greedy_forward_selection` t-stats are post-selection inference |
| Broad adaptive search / winner selection | Held-out evaluation, or an external search-wide procedure | [BHY](../api/multi-factor.md) over the separately declared marginal family | That BHY has controlled the whole research search |

The boundaries the table compresses, each owned by the page it links to:

- `expected_warnings` affects presentation and the audit fields only — it marks
  matching records as expected and quiets their echo, and it does not change
  the estimator, the p-value, or the sample requirement
  ([warning contract](../api/slice-test.md#warning-contract-slice_period_joint_test)).
- A regime joint test on short slices with `K >= 3` carries a
  [disclosed over-rejection](../reference/statistical-methods.md#joint-period-test-on-short-slices-known-over-rejection),
  and declaring `short_slice_joint_test` as expected does not calibrate that
  p-value, so it stays robustness evidence rather than an admission gate.
- `spanning_alpha` is a fixed-base incremental-information / redundancy
  diagnostic, and `greedy_forward_selection`'s t-stats are
  selection-conditioned rather than
  [post-selection inference](../api/metrics/spanning.md).
- BHY controls the FDR of the declared hypothesis family, not the winner of an
  adaptive search across many candidates; for that use held-out evaluation or
  an external search-wide procedure — Hansen SPA / White Reality Check, which
  [factrix does not implement](../reference/statistical-methods.md#2-multiple-testing-under-dependence).
- An uneven evaluation grid is disclosed by `uneven_evaluation_grid`
  ([warning codes](../reference/warning-codes.md)); the series-mean paths are
  calibrated there and the regression HAC, persistence, event-window and
  adjacent-period paths are not — read the
  [scope of that disclosure](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns)
  instead of generalizing it.

The metric follows the research question rather than a global ranking of
metrics: `predictive_beta` is the primary evidence for single-asset
time-series predictability and is not cross-sectional ranking evidence, just as
rank IC says nothing about one asset's own series. Regime, redundancy and
search-wide questions come after a primary screen, and everything past a
promoted candidate — weights, execution, capacity — is downstream of factrix.

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

!!! warning "Lower the bucket counts before you run them"

    The bucketing defaults are calibrated for a ~2000-name equity universe:
    `monotonicity(n_groups=10)`, `notional_turnover(n_groups=5)`,
    `quantile_spread(n_groups=5)`, `k_spread(k=5)`. A metric whose bucket count
    exceeds the cross-section cannot fill its legs on any date, so it refuses on
    the **assets** axis (`InsufficientSampleError` under `strict=True`;
    `reason="insufficient_assets_for_quantile_groups"` and
    `THIN_QUANTILE_GROUPS` under `strict=False`) however long the panel. For
    `n_assets < 20` pass `n_groups=2..3` and `k=1..2` — around five names per
    leg is the point below which a leg mean is one or two names' idiosyncratic
    noise. `fx.inspect_data(panel)` pre-flights the **default** configuration,
    so check it against the arguments you actually intend to pass.

This example keeps the warning audit trail while quieting a sample regime that
is expected by design:

```python
import factrix as fx
from factrix.metrics import directional_hit_rate, directional_pair_accuracy, ic
from factrix.metrics import ic_ir, k_spread
from factrix.preprocess import compute_forward_return

raw = fx.datasets.make_cs_panel(n_assets=12, n_dates=252, rng=2024)
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

Use the same `n_groups` for the spread and the turnover. On an evaluation grid
built with `compute_forward_return(..., dates=)`, pass `rebalance_lag=1` to the
turnover metrics when that grid *is* the rebalance schedule, and give
`breakeven_cost` / `net_spread` a `holding_periods` in **underlying return
periods** — never the derived evaluation-grid `overlap_periods`.

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
