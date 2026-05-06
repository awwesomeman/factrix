# API

Reference for every public symbol exported from `factrix`. The pages
below mirror the user surface in dependency order — start at the
top if you are new.

## Entry points

| Page | What it is | When to read |
|---|---|---|
| [`AnalysisConfig`](analysis-config.md) | Three-axis frozen dataclass selecting the dispatch cell. Four factory methods (`individual_continuous`, `individual_sparse`, `common_continuous`, `common_sparse`). | Picking the analysis cell. |
| [`evaluate`](evaluate.md) | Single dispatch entry — runs the registered procedure for a `(config, panel)` pair and returns a `FactorProfile`. | Running an analysis. |
| [`FactorProfile`](factor-profile.md) | Frozen result object: `primary_p`, `verdict()`, `diagnose()`, `stats`, `warnings`, `info_notes`, `mode`, `n_obs`, `n_assets`. | Reading the result. |
| [`list_metrics`](list-metrics.md) | Programmatic discovery of standalone `factrix.metrics.*` callables applicable to a given `(scope, signal)` cell. | Picking a follow-up metric after `evaluate()`. |
| [`Metrics`](metrics/index.md) | Per-module reference for every public function under `factrix.metrics`. | Calling a standalone metric directly on a `FactorProfile` / panel. |

## Supporting surface

| Page | What it is |
|---|---|
| [`MetricOutput`](metric-output.md) | Common wrapper returned by every standalone metric — `value`, `p_value`, `stats`, `metadata`. |
| [`multi_factor`](multi-factor.md) | `bhy(...)` for per-family BHY FDR screening across a list of `FactorProfile`s. |
| [`datasets`](datasets.md) | Synthetic panels (`make_cs_panel`, `make_event_panel`) for smoke tests and docs examples. |

The two helpers `describe_analysis_modes` and `suggest_config` are
documented inline on
[`AnalysisConfig`](analysis-config.md) and the
[Concepts](../getting-started/concepts.md) page; both are introspection
shims, not part of the dispatch surface.

## Naming convention

Sidebar entries mirror the actual Python identifier — the case
distinction is intentional, not inconsistent:

| Sidebar entry | Identifier kind | Example call |
|---|---|---|
| `AnalysisConfig`, `FactorProfile`, `MetricOutput` | Class | `fl.AnalysisConfig.individual_continuous(...)` |
| `evaluate`, `list_metrics` | Function | `fl.evaluate(panel, cfg)` |
| `multi_factor`, `datasets`, `Metrics` (and submodules) | Module | `fl.multi_factor.bhy(profiles)` |

Reading the sidebar gives you the import path verbatim — `fl.evaluate`
is a function, `fl.AnalysisConfig` is a class, `fl.multi_factor` is a
module you reach into for `bhy`.
