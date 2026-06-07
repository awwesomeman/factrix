---
title: factrix.list_metrics
---

::: factrix.list_metrics

Programmatic discovery of the standalone metrics under
[`factrix.metrics`](metrics/index.md). `list_metrics()` takes **no
arguments** and returns a family-grouped catalog —
a `dict[str, list[MetricSpec]]` keyed by concept family (the declaring
module stem), values being that family's public specs:

```python
import factrix as fx

overview = fx.list_metrics()
overview["ic"]       # [MetricSpec(name="ic", ...), MetricSpec(name="ic_ir", ...), ...]
sorted(overview)     # ['caar', 'fm_beta', 'ic', ...] — the concept families
```

The catalog is a *reference*, not a runnable input. Wire concrete
callables up from `factrix.metrics` and pass them to
[`evaluate`](evaluate.md):

```python
from factrix.metrics import ic, fm_beta, breakeven_cost
```

## Per-cell applicability → `inspect_data`

The former `list_metrics(scope, density)` cell filter — together with
its `format="json"` and `with_import` knobs — is **retired**. To
learn which metrics actually run on a given panel, and which are
degraded or blocked by sample floors, inspect a real panel with
`inspect_data`:

```python
info = fx.inspect_data(panel)
[m.name for m in info.usable]     # production-safe metrics for this panel
[m.name for m in info.degraded]   # run, but inference degraded
[m.name for m in info.unusable]   # blocked (cell mismatch / sample floor)
```

`inspect_data` subsumes the old cell filter with a structure-aware,
sample-floor-aware verdict: each `MetricApplicability` carries the
metric `name`, its callable class, and any `blockers` / `warnings`.
Unlike the static cell filter, it accounts for the actual panel shape
(`n_periods` / `n_assets` / `n_pairs`).

## Source of truth

The catalog is built from the registered `@metric` classes in each
`factrix/metrics/*.py` module (`factrix._metric_index`). Adding a metric
makes it appear in the overview automatically.
