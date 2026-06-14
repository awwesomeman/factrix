---
title: Information coefficient vs Fama-MacBeth
---

This guide helps you choose the correct metric and execution parameters for your factor research.

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
board = fx.compare(results, metrics=["ic", "fm"], sort_by="ic")
print(board)
```

For a full list of available metrics and their mathematical descriptions, see the [Metrics applicability table](../reference/metric-applicability.md) and the [API Reference](../api/index.md).
