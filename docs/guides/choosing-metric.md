---
title: Information coefficient vs Fama-MacBeth
---

This page is task-oriented: assume you already know factrix exists and you
need to pick a factory. For the canonical 5-scenario table (research
question → factory → procedure → literature) see
[Concepts § Five analysis scenarios](../getting-started/concepts.md#five-analysis-scenarios).

Factory methods are type-safe constructors. Unsupported combinations (e.g.
`metric=IC` on a sparse signal) are caught by the IDE before runtime — no
need to memorise the legal axis triples.

## Information coefficient (IC) vs FM

Both apply to `(INDIVIDUAL, CONTINUOUS)`. Choose by research question:

| | IC | FM |
|--|----|----|
| Question | Predictive rank ordering? | Unit-exposure return premium? |
| Method | Spearman ρ per date → Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t on E[information coefficient (IC)] | ordinary least squares (OLS) slope λ per date → NW HAC t on E[λ] |
| Robust to | Outliers (rank-based) | Proportional exposure differences |
| Economic interpretation | Directional signal quality | Premium per unit of factor exposure |
| `n_assets` sensitivity | Drops dates with < 10 assets | Runs at N ≥ 3 but unstable at low N |

Use IC when you care about rank ordering (stock selection). Use FM when you need an economically interpretable premium estimate (risk premia, factor pricing).

For the lookup table — which metrics are supported under which `(scope, signal)` cell, with sample-size floors and warning codes — see
[Reference § Metric applicability](../reference/metric-applicability.md).

## Standalone metrics vs `evaluate()`

[`evaluate()`][factrix.evaluate] runs the canonical inference procedure for a cell. Standalone metrics in `factrix.metrics` provide supplementary diagnostics without a formal PASS/FAIL outcome:

| When to use `evaluate()` | When to use standalone metrics |
|---|---|
| Canonical signal validity inference | Diagnose shape, asymmetry, regime splits |
| Benjamini-Yekutieli (BHY) family input (needs [`FactorProfile`][factrix.FactorProfile]) | Multi-statistic decomposition |
| Primary screening gate | out-of-sample (OOS) decay, tradability, concentration |

See [Metric pipelines](../reference/metric-pipelines.md) for the full module list.
