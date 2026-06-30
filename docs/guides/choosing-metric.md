---
title: Choosing a metric
---

This guide helps you choose the first metric to run, then points to the
reference tables that answer whether your data can run it and how to read the
output.

## First-pass metrics vs diagnostics

Start from the research question. The first-pass metric is the conventional
headline read for that question; diagnostics add shape, robustness, and
implementation context.

factrix uses "mainstream" and "supplementary" as a usage convention, not
as a registry tier. This mirrors the way quant research usually separates a
primary specification from robustness checks, decompositions, and diagnostics.
`evaluate()` runs exactly the metrics you pass, and `list_metrics()` lists
public metrics without marking one as mandatory.

You may run a diagnostic directly when it matches the research question. Treat
it as targeted evidence unless you deliberately make it the tested family. If
you screen many factors or many diagnostic variants, define that family up
front and apply multiple-testing control deliberately.

| Research question | Usually first-pass | Useful diagnostics | Why |
|---|---|---|---|
| Rank-ordering in a cross-section | [`ic`](../api/metrics/ic.md) | [`ic_ir`](../api/metrics/ic.md), [`monotonicity`](../api/metrics/monotonicity.md), [`k_spread`](../api/metrics/k_spread.md), [`directional_pair_accuracy`](../api/metrics/directional_pair_accuracy.md), [`directional_hit_rate`](../api/metrics/directional_hit_rate.md) | `ic` is the rank-based default for `Individual x Continuous`; diagnostics check concentration, spread shape, pairwise ordering, and sign skill. |
| Exposure premium in a cross-section | [`fm_beta`](../api/metrics/fm_beta.md) | [`pooled_beta`](../api/metrics/fm_beta.md), [`fm_beta_sign_consistency`](../api/metrics/fm_beta.md), [`k_spread`](../api/metrics/k_spread.md) | `fm_beta` keeps the economic unit of exposure; diagnostics show pooled fit, sign stability, and small-universe spread. |
| Sparse event effect | [`caar`](../api/metrics/caar.md) | [`bmp_z`](../api/metrics/caar.md), [`corrado_rank`](../api/metrics/corrado_rank.md), [`event_hit_rate`](../api/metrics/event_quality.md), [`profit_factor`](../api/metrics/event_quality.md), [`clustering_hhi`](../api/metrics/clustering_hhi.md) | `caar` is the event-time headline test; diagnostics check variance assumptions, ranks, hit quality, payoff asymmetry, and event clustering. |
| Common macro exposure | [`common_beta`](../api/metrics/common_beta.md) | [`common_beta_profile`](../api/metrics/common_beta.md), [`common_beta_r_squared`](../api/metrics/common_beta.md), [`common_beta_sign_consistency`](../api/metrics/common_beta.md), [`compute_common_betas(...)[factor]`](../api/metrics/common_beta.md) | `common_beta` tests average per-asset beta; diagnostics reveal whether offsetting positive and negative betas hide a rotation profile. |
| Single-asset dense prediction | [`predictive_beta`](../api/metrics/predictive_beta.md) | [`directional_hit_rate`](../api/metrics/directional_hit_rate.md), plus the [`predictive_beta` stability workflow](../api/metrics/predictive_beta.md#stability-workflow) | `predictive_beta` tests the HAC predictive-regression slope; diagnostics check sign prediction and whether pre-declared rolling betas keep the same economic direction. |
| Tradability / implementation pressure | No headline test | [`tradability`](../api/metrics/tradability.md), [`concentration`](../api/metrics/concentration.md), turnover / cost diagnostics | These are descriptive checks around implementation pressure, not a replacement for the factor's headline inference. |

Then use the cross-reference pages by task:

| Question | Reference |
|---|---|
| Can my data run this metric? | [Metric applicability](../reference/metric-applicability.md) |
| How is the metric computed? | [Metric pipelines](../reference/metric-pipelines.md) |
| Which `MetricResult` fields and metadata keys matter? | [Stat keys by metric](../reference/stat-keys-by-metric.md) |
| Which metrics apply to my specific panel? | [`inspect_data`](../api/inspect-data.md) |

## Information coefficient (IC) vs Fama-MacBeth (FM)

Both metrics evaluate individual, continuous factors (`FactorScope.INDIVIDUAL`
and `FactorDensity.DENSE` cells), but they answer different research questions:

| Feature | Information coefficient (IC) | Fama-MacBeth (FM) |
|---|---|---|
| Research question | Does the factor consistently rank-order future returns? | What is the return premium per unit of factor exposure? |
| Statistical method | Per-date Spearman rank correlation, then Newey-West HAC t-test on the mean IC. | Per-date cross-sectional OLS regression slope, then Newey-West HAC t-test on the mean slope. |
| Robustness | Rank-based and robust to outliers. | OLS-based and more sensitive to outliers and extreme values. |
| Economic interpretation | Rank-ordering capability. | Return premium per unit of factor exposure. |
| Sample sensitivity | Drops dates with fewer than 2 complete pairs; warns when retained dates have fewer than 10 assets. | Drops dates with fewer than 3 complete pairs; warns when retained dates have fewer than 10 assets. |

- Use `ic` when you are building a ranking-based stock selection strategy.
- Use `fm_beta` when you need to estimate risk premia or require an
  economically interpretable slope premium.

## Evaluating and comparing metrics

To evaluate both IC and Fama-MacBeth on a candidate factor panel, pass both metric instances to `fx.evaluate()`:

```python
import factrix as fx
from factrix.metrics import ic, fm_beta

raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=200, seed=42)
data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

results = fx.evaluate(
    data,
    metrics={
        "ic": ic(inference=fx.inference.NEWEY_WEST),
        "fm": fm_beta(),
    },
    factor_cols=["factor"],
    forward_periods=5,
)

# Compare metric values and p-values side by side
board = fx.compare(list(results.values()), metrics=["ic", "fm"], sort_by="ic")
print(board.to_dicts())
```

For the full metric catalog, see [Metrics](../api/metrics/index.md).

## Single-asset dense stability

For `n_assets == 1` dense panels, use `predictive_beta` as the single canonical
inference read. If you need stability evidence, derive a rolling or expanding
beta series with pre-declared `window_periods` / `step_periods` and summarize it
descriptively: sign consistency, recent beta, median beta, and min/max beta.

Those windowed betas are robustness diagnostics, not a new alpha-selection
family. Overlapping windows share observations, so their p-values are not
independent hypotheses to rank, optimize over, or pass to `bhy`. If the profile
suggests a structural break, treat the break as a separate regime research
design rather than letting the stability helper pick the regime for you.
