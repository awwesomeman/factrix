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
| Whether a market-wide time-series factor is priced across assets | [`ts_beta`](../api/metrics/ts_beta.md) | Cross-asset test on per-asset beta estimates; requires `N >= 2`. |
| Whether a signal is tradable after turnover / cost pressure | [`tradability`](../api/metrics/tradability.md), [`concentration`](../api/metrics/concentration.md) | Descriptive diagnostics around implementation pressure. |
| Whether a result decays, trends, or keeps the right sign over time | [`oos_decay`](../api/metrics/oos_decay.md), [`ic_trend`](../api/metrics/trend.md), [`hit_rate`](../api/metrics/hit_rate.md) | Series diagnostics layered on top of a cell's primary result. |

Then use the cross-reference pages by task:

| Question | Reference |
|---|---|
| Can my data run this metric? | [Metric applicability](../reference/metric-applicability.md) |
| How is the metric computed? | [Metric pipelines](../reference/metric-pipelines.md) |
| Which `MetricResult` fields and metadata keys matter? | [Stat keys by metric](../reference/stat-keys-by-metric.md) |
| Which metrics apply to my specific panel? | [`inspect_data`](../api/inspect-data.md) |

## Information coefficient (IC) vs Fama-MacBeth (FM)

Both metrics evaluate individual, continuous factors (`FactorScope.INDIVIDUAL` and `FactorDensity.DENSE` cells), but they answer different research questions:

| Feature | Information Coefficient (IC) | Fama-MacBeth (FM) |
|---|---|---|
| **Research Question** | Does the factor consistently rank-order future returns? | What is the return premium per unit of factor exposure? |
| **Statistical Method** | Per-date Spearman rank correlation $\rho_t$ → Newey-West HAC $t$-test on $\mathbb{E}[\rho_t]$. | Per-date cross-sectional OLS regression slope $\lambda_t$ → Newey-West HAC $t$-test on $\mathbb{E}[\lambda_t]$. |
| **Robustness** | Extremely robust to outliers (rank-based). | Sensitive to outliers and extreme values (OLS-based). |
| **Economic Interpretation** | Rank-ordering capability (signal quality). | Return premium per unit of factor exposure. |
| **Sample Sensitivity** | Drops dates with fewer than 10 assets. | Requires at least 3 assets per date but can be unstable at low $N$. |

- **Use IC** (`ic`) when you are building a ranking-based stock selection strategy.
- **Use FM** (`fm_beta`) when you need to estimate risk premia or require an economically interpretable slope premium.

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
