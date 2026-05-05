# Choosing a Metric

This page is task-oriented: assume you already know factrix exists and you
need to pick a factory. For the canonical 5-scenario table (research
question → factory → procedure → literature) see
[Concepts § Five analysis scenarios](../getting-started/concepts.md#five-analysis-scenarios).

Factory methods are type-safe constructors. Unsupported combinations (e.g.
`metric=IC` on a sparse signal) are caught by the IDE before runtime — no
need to memorise the legal axis triples.

## IC vs FM

Both apply to `(INDIVIDUAL, CONTINUOUS)`. Choose by research question:

| | IC | FM |
|--|----|----|
| Question | Predictive rank ordering? | Unit-exposure return premium? |
| Method | Spearman ρ per date → NW HAC t on E[IC] | OLS slope λ per date → NW HAC t on E[λ] |
| Robust to | Outliers (rank-based) | Proportional exposure differences |
| Economic interpretation | Directional signal quality | Premium per unit of factor exposure |
| `n_assets` sensitivity | Drops dates with < 10 assets | Runs at N ≥ 3 but unstable at low N |

Use IC when you care about rank ordering (stock selection). Use FM when you need an economically interpretable premium estimate (risk premia, factor pricing).

## Standalone metrics vs `evaluate()`

`evaluate()` runs the canonical PASS/FAIL procedure for a cell. Standalone metrics in `factrix.metrics` provide supplementary diagnostics without a formal PASS/FAIL verdict:

| When to use `evaluate()` | When to use standalone metrics |
|---|---|
| Canonical signal validity verdict | Diagnose shape, asymmetry, regime splits |
| BHY family input (needs `FactorProfile`) | Multi-statistic decomposition |
| Primary screening gate | OOS decay, tradability, concentration |

See [Standalone metrics](../reference/standalone-metrics.md) for the full module list.
