---
title: factrix.list_estimators
---

::: factrix.list_estimators

No-arg programmatic discovery of every registered inference method
(`Estimator` instance) — the inference-side twin of
[`list_metrics`](list-metrics.md). Returns the whole registry
regardless of cell context (#255, mirroring the `list_metrics`
overview design in #498).

Two contracts to keep in mind:

- **Cell → procedure is 1:1** (per #148). A cell's primary p-value is produced by exactly one estimator at `evaluate` time.
- **Family functions accept an `estimator=` override.** `bhy` / `bhy_hierarchical` / `partial_conjunction` swap which already-computed p-value drives the multiplicity step. `list_estimators` returns every candidate for that override; per-cell applicability is each `Estimator`'s own `applicable_to(scope, density)` contract, not a filter on this call.

## No axes are inputs

`list_estimators` takes no `scope` / `density` / `DataStructure`
arguments — it is a flat overview of the registry. Whether a given
estimator applies to a specific cell is `Estimator.applicable_to`'s
job, checked at `evaluate` time, not here.
See the docstring Examples block above for the canonical text-list and
JSON-form calls.

---

## See also

- [Estimator alternatives](estimator-alternatives.md) — cookbook for
  using `estimator=` on screening functions
- [`stats`](stats.md) — full estimator catalogue and StatCode pairings
- [`list_metrics`](list-metrics.md) — descriptive metric counterpart
