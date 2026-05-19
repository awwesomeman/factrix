---
title: factrix.list_estimators
---

::: factrix.list_estimators

Programmatic discovery of inference methods (`Estimator` instances)
applicable to a given `(scope, signal)` cell — the inference-side twin
of [`list_metrics`](list-metrics.md).

Two contracts to keep in mind:

- **Cell → procedure is 1:1** (per #148). A cell's primary p-value is produced by exactly one estimator at `evaluate` time.
- **Family functions accept an `estimator=` override.** `bhy` / `bhy_hierarchical` / `partial_conjunction` swap which already-computed p-value drives the multiplicity step. `list_estimators` returns the candidates for that override, filtered to those applicable to the cell.

## PanelMode axis is not an input

`PanelMode` is intentionally not a parameter — estimator applicability is
cell-axis-dependent (scope × signal), not panel-shape-dependent.
See the docstring Examples block above for the canonical text-list and
JSON-form calls.

---

## See also

- [Estimator alternatives](estimator-alternatives.md) — cookbook for
  using `estimator=` on screening functions
- [`stats`](stats.md) — full estimator catalogue and StatCode pairings
- [`list_metrics`](list-metrics.md) — descriptive metric counterpart
