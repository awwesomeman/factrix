# list_estimators

Programmatic discovery of inference methods (`Estimator` instances)
applicable to a given `(scope, signal)` cell — the inference-side twin
of [`list_metrics`](list-metrics.md).

Two contracts to keep in mind:

- **Cell → procedure is 1:1** (per #148). A cell's primary p-value is produced by exactly one estimator at `evaluate` time.
- **Family verbs accept an `estimator=` override.** `bhy` / `bhy_hierarchical` / `partial_conjunction` swap which already-computed p-value drives the multiplicity step. `list_estimators` returns the candidates for that override, filtered to those applicable to the cell.

## Call shape

```python
import factrix as fx

# Text list (default) — pretty names for CLI / REPL inspection
fx.list_estimators(fx.FactorScope.INDIVIDUAL, fx.Signal.CONTINUOUS)
# → ['NeweyWest', 'HansenHodrick', 'BlockBootstrap', ...]

# Structured records — programmatic filtering / import paths
fx.list_estimators(fx.FactorScope.INDIVIDUAL, fx.Signal.CONTINUOUS,
                   format='json', with_import=True)
# → [{'name': 'NeweyWest', 'import': 'from factrix.stats import NeweyWest', ...}, ...]
```

`Mode` is intentionally not an input — estimator applicability is
cell-axis-dependent (scope × signal), not panel-shape-dependent.

---

## See also

- [Estimator alternatives](estimator-alternatives.md) — cookbook for
  using `estimator=` on family verbs
- [`stats`](stats.md) — full estimator catalogue and StatCode pairings
- [`list_metrics`](list-metrics.md) — descriptive metric counterpart

::: factrix.list_estimators
