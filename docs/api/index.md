# API

Reference for every public symbol exported from `factrix`.

## Entry points

| Page | What it is | When to read |
|---|---|---|
| [`AnalysisConfig`](analysis-config.md) | Three-axis frozen dataclass selecting the dispatch cell. Four factory methods (`individual_continuous`, `individual_sparse`, `common_continuous`, `common_sparse`). | Picking the analysis cell. |
| [`evaluate`](evaluate.md) | Single dispatch entry — runs the registered procedure for a `(config, panel)` pair and returns a `FactorProfile`. | Running an analysis. |
| [`FactorProfile`](factor-profile.md) | Frozen result object: `primary_p`, `verdict()`, `diagnose()`, `stats`, `warnings`, `info_notes`, `mode`, `n_obs`, `n_assets`. | Reading the result. |
| [`multi_factor`](multi-factor.md) | `bhy(...)` for per-family BHY FDR screening across a list of `FactorProfile`s. | Multi-factor screening. |
| [`stats`](stats.md) | Estimator catalogue (`NeweyWest` / `HansenHodrick` / `WaldNWCluster` / `WaldTwoWayCluster` / `BlockBootstrap`), StatCode pairs, FDR / bootstrap utilities. | Picking inference method for `bhy(estimator=…)` or cross-slice tests. |
| [`list_metrics`](list-metrics.md) | Programmatic discovery of standalone `factrix.metrics.*` callables applicable to a given `(scope, signal)` cell. | Picking a follow-up metric after `evaluate()`. |
| [`Metrics`](metrics/index.md) | Per-module reference for every public function under `factrix.metrics`. | Calling a standalone metric directly on a `FactorProfile` / panel. |

## Supporting surface

| Page | What it is |
|---|---|
| [`MetricOutput`](metric-output.md) | Common wrapper returned by every standalone metric — `value`, `p_value`, `stats`, `metadata`. |
| [`datasets`](datasets.md) | Synthetic panels (`make_cs_panel`, `make_event_panel`) for smoke tests and docs examples. |

`describe_analysis_modes` and `suggest_config` are introspection shims,
documented inline on [`AnalysisConfig`](analysis-config.md) and
[Concepts](../getting-started/concepts.md).

## Naming convention

Sidebar entries mirror the actual Python identifier — the case
distinction is intentional, not inconsistent:

| Sidebar entry | Identifier kind | Example call |
|---|---|---|
| `AnalysisConfig`, `FactorProfile`, `MetricOutput` | Class | `fx.AnalysisConfig.individual_continuous(...)` |
| `evaluate`, `list_metrics` | Function | `fx.evaluate(panel, cfg)` |
| `multi_factor`, `datasets`, `Metrics` (and submodules) | Module | `fx.multi_factor.bhy(profiles)` |
