---
title: Choosing a metric
---

This guide helps you choose the first metric to run, then points to the
reference tables that answer whether your data can run it and how to read the
output.

## Start from the research question

| If you want to know... | Start with | Why |
|---|---|---|
| Whether an equity-style factor rank-orders future returns | [`ic`](../api/metrics/ic.md) | Rank-based, robust default for `Individual × Continuous` factors. |
| The return premium per unit of factor exposure | [`fm_beta`](../api/metrics/fm_beta.md) | Per-date cross-sectional slope with an economic unit. |
| Whether an event signal produces abnormal returns | [`caar`](../api/metrics/caar.md) | Mainstream event-time significance test for sparse factors. |
| Whether event-study inference is sensitive to variance or ranking assumptions | [`bmp_z`](../api/metrics/caar.md), [`corrado_rank`](../api/metrics/corrado_rank.md) | Robustness checks for event-induced variance and non-normal returns. |
| Whether a market-wide time-series factor is priced across assets | [`common_beta`](../api/metrics/common_beta.md) | Cross-asset test on per-asset beta estimates; requires `n_assets >= 2`. |
| Whether a single-asset dense signal predicts forward returns | [`predictive_beta`](../api/metrics/predictive_beta.md) | Direct `forward_return ~ factor` slope with Newey-West HAC inference; requires `n_assets == 1`. |
| Whether a common macro factor separates assets with opposite beta signs | [`common_beta`](../api/metrics/common_beta.md), [`common_beta_sign_consistency`](../api/metrics/common_beta.md#factrix.metrics.common_beta.common_beta_sign_consistency) | Read the beta profile, sign split, and explanatory power before turning it into an allocation rule. |
| Whether a small allocation universe has a fixed-count long-short edge | [`k_spread`](../api/metrics/k_spread.md), [`directional_hit_rate`](../api/metrics/directional_hit_rate.md) | Supplementary small-N diagnostics when quantile buckets are too thin; not a replacement for the canonical IC/FM p-value family. |
| Whether a signal is tradable after turnover / cost pressure | [`tradability`](../api/metrics/tradability.md), [`concentration`](../api/metrics/concentration.md) | Descriptive diagnostics around implementation pressure. |
| Whether a result decays, trends, or keeps the right sign over time | [`oos_decay`](../api/metrics/oos_decay.md), [`ic_trend`](../api/metrics/trend.md), [`positive_rate`](../api/metrics/positive_rate.md) | Series diagnostics layered on top of a cell's primary result. |

Then use the cross-reference pages by task:

| Question | Reference |
|---|---|
| Can my data run this metric? | [Metric applicability](../reference/metric-applicability.md) |
| How is the metric computed? | [Metric pipelines](../reference/metric-pipelines.md) |
| Which `MetricResult` fields and metadata keys matter? | [Stat keys by metric](../reference/stat-keys-by-metric.md) |
| Which metrics apply to my specific panel? | [`inspect_data`](../api/inspect-data.md) |

## First-pass metrics vs diagnostics

factrix uses "mainstream" and "supplementary" as a usage convention, not
as a registry tier. This mirrors the way quant research usually separates a
primary specification from robustness checks, decompositions, and diagnostics.
`evaluate()` runs exactly the metrics you pass, and `list_metrics()` lists
public metrics without marking one as mandatory.

You may run a diagnostic directly when it matches the research question. Treat
it as targeted evidence unless you deliberately make it the tested family. If
you screen many factors or many diagnostic variants, define that family up
front and apply multiple-testing control deliberately.

| Research question | Usually first-pass | Usually diagnostic |
|---|---|---|
| Rank-ordering in a cross-section | [`ic`](../api/metrics/ic.md) | `ic_ir`, `monotonicity`, [`k_spread`](../api/metrics/k_spread.md), [`directional_hit_rate`](../api/metrics/directional_hit_rate.md) |
| Exposure premium in a cross-section | [`fm_beta`](../api/metrics/fm_beta.md) | `pooled_beta`, `fm_beta_sign_consistency`, [`k_spread`](../api/metrics/k_spread.md) |
| Sparse event effect | [`caar`](../api/metrics/caar.md) | `bmp_z`, `corrado_rank`, `event_hit_rate`, `profit_factor`, `clustering_hhi` |
| Common macro exposure | [`common_beta`](../api/metrics/common_beta.md) | `common_beta_r_squared`, `common_beta_sign_consistency`, beta-profile diagnostics |
| Single-asset dense prediction | [`predictive_beta`](../api/metrics/predictive_beta.md) | [`directional_hit_rate`](../api/metrics/directional_hit_rate.md), plus trend / out-of-sample diagnostics on derived `(date, value)` series |

## Information coefficient (IC) vs Fama-MacBeth (FM)

Both metrics evaluate individual, continuous factors (`FactorScope.INDIVIDUAL` and `FactorDensity.DENSE` cells), but they answer different research questions:

| Feature | Information Coefficient (IC) | Fama-MacBeth (FM) |
|---|---|---|
| **Research Question** | Does the factor consistently rank-order future returns? | What is the return premium per unit of factor exposure? |
| **Statistical Method** | Per-date Spearman rank correlation $\rho_t$ → Newey-West HAC $t$-test on $\mathbb{E}[\rho_t]$. | Per-date cross-sectional OLS regression slope $\lambda_t$ → Newey-West HAC $t$-test on $\mathbb{E}[\lambda_t]$. |
| **Robustness** | Extremely robust to outliers (rank-based). | Sensitive to outliers and extreme values (OLS-based). |
| **Economic Interpretation** | Rank-ordering capability (signal quality). | Return premium per unit of factor exposure. |
| **Sample Sensitivity** | Drops dates with fewer than 2 complete pairs; warns when retained dates have fewer than 10 assets. | Drops dates with fewer than 3 complete pairs; warns when retained dates have fewer than 10 assets. |

- **Use IC** (`ic`) when you are building a ranking-based stock selection strategy.
- **Use FM** (`fm_beta`) when you need to estimate risk premia or require an economically interpretable slope premium.

## Allocation panels

For common macro factors, `common_beta` is the average exposure test. Inspect
`metadata["beta_std"]`, `metadata["median_beta"]`,
`common_beta_sign_consistency`, and `common_beta_r_squared` when assets may load with
opposite signs. For one-asset dense data, use `predictive_beta` for the
HAC-corrected magnitude slope and `directional_hit_rate` for sign prediction.
For small individual dense panels, use `k_spread` and `directional_hit_rate`
as supplementary reads alongside IC / FM, not as a new screening family.

---

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
