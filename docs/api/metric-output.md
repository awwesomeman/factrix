# MetricOutput

The unified return type produced by every metric in
[`factrix.metrics`](metrics/index.md). A single dataclass carrying
the canonical scalar (`value`), the sample size the estimator saw
(`n_obs`), the test statistic (`stat`), the significance marker, and
a `metadata` dict for everything else (p-value, method label,
short-circuit reason, secondary diagnostics). All metrics — primary
or auxiliary — return this shape so downstream code can treat every
metric uniformly. `n_obs` shares its name with `FactorProfile.n_obs`
but a different scope: per-metric single-stage count vs. the
final-stage test denominator at the dispatched-cell level.

::: factrix.MetricOutput

## Resolving `MetricOutput.name` to a docs page
[](){ #name-index }

Downstream consumers holding a `MetricOutput` can map `output.name`
back to the metric's API page two ways:

- **Programmatic** —
  [`list_metrics(..., format="json")`](list-metrics.md) rows carry an
  `emitted_name` field (the literal `MetricOutput.name` value) and a
  `docs_anchor` field following the `DOCS_ANCHOR_FMT` convention
  defined in `factrix._metric_index` (docs-root-relative path +
  mkdocstrings symbol fragment), resolvable without scraping.
- **Static** — the auto-generated table below maps every emitted
  `MetricOutput.name` value to the function that produced it and the
  API page anchor. Regenerated from the same `Matrix-row:` SSOT on
  every docs build.

For most metrics `MetricOutput.name` equals the function name; a
small set of historical exceptions (e.g. `fama_macbeth` emits
`fm_beta`, `corrado_rank_test` emits `corrado_rank`) emit a
different label. The first column is what you receive at runtime;
the second column is the import-path callable.

--8<-- "docs/reference/_generated_metric_name_index.md"
