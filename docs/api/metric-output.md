---
title: factrix.MetricResult
---

::: factrix.MetricResult

The unified return type produced by every metric in
[`factrix.metrics`](metrics/index.md). A single dataclass carrying
the canonical scalar (`value`), the two-sided p-value (`p`, promoted
from `metadata["p_value"]` to a typed field; `None` for descriptive
metrics), the sample size the estimator saw (`n_obs`), the test
statistic (`stat`), a back-pointer to the declaring spec (`spec`), and
a `metadata` dict for everything else (method label, short-circuit
reason, secondary diagnostics). All metrics — primary or auxiliary —
return this shape so downstream code can treat every metric uniformly.
`n_obs` shares its name with `FactorProfile.n_obs` but a different
scope: per-metric single-stage count vs. the final-stage test
denominator at the dispatched-cell level.

## Resolving a metric name to a docs page
[](){ #name-index }

A `MetricResult` produced through the DAG executor (`fx.evaluate`)
carries its declaring spec, so the metric name is `result.spec.name`.

!!! note "Direct primitive calls return `spec=None`"

    Calling a metric directly — `fx.metrics.ic(panel)`,
    `fx.metrics.caar(events)` — returns a `MetricResult` with
    `spec is None` (the registry stamps `spec` only on the executor
    path). On that path you already hold the metric you called, so the
    name is the function you invoked; there is no `result.name`
    attribute (removed in favour of `spec.name`). Reach for the index
    below only when resolving a name you received second-hand.

The metric name maps back to its API page two ways:

- **Programmatic** — the docs-build hooks resolve each metric's page
  anchor through `docs_anchor_for` in `factrix._metric_index`, which
  follows the `DOCS_ANCHOR_FMT` convention (docs-root-relative path +
  mkdocstrings symbol fragment), resolvable without scraping.
- **Static** — the auto-generated table below maps every emitted
  metric name to the function that produced it and the API page
  anchor. Regenerated from the same `Matrix-row:` SSOT on every docs
  build.

For most metrics the emitted name equals the function name; a small
set of historical exceptions (e.g. `fama_macbeth` emits `fm_beta`,
`corrado_rank_test` emits `corrado_rank`) emit a different label. The
first column is what you receive at runtime; the second column is the
import-path callable.

--8<-- "docs/reference/_generated_metric_name_index.md"
