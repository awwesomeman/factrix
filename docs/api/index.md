---
title: API
---

Reference for every public symbol exported from `factrix`.

## Function flow

```mermaid
flowchart LR
    P[data]

    subgraph PerFactor["Inference · per factor"]
        EV[evaluate]
        EVH[evaluate_horizons]
    end

    subgraph Screening["Screening · FDR across factors"]
        BHY[bhy]
        PC[partial_conjunction]
        BHYH[bhy_hierarchical]
    end

    subgraph Slices["Inference · across slices (no FDR)"]
        ST["slice_pairwise_test / slice_joint_test<br/>slice_period_pairwise_test / slice_period_joint_test"]
    end

    subgraph View["Descriptive view"]
        BS[by_slice]
        CMP[compare]
    end

    subgraph Introspection
        LM[list_metrics]
        ID[inspect_data]
    end

    P ==> EV
    P ==> EVH
    P -.-> BS
    P -.-> ST
    P -.-> ID
    EV ==>|results| BHY
    EV ==>|results| PC
    EV ==>|results| BHYH
    EV ==>|results| CMP
    EVH ==>|results| BHY
    EVH ==>|results| CMP
    BHY ==>|survivors| CMP
    PC ==>|survivors| CMP
    BHYH ==>|survivors| CMP
    LM -.->|metric names| EV

    click EV "evaluate/" "evaluate API"
    click EVH "evaluate/#factrix.evaluate_horizons" "evaluate_horizons API"
    click BS "by-slice/" "by_slice API"
    click ST "slice-test/" "slice tests — cross-sectional + period-disjoint API"
    click BHY "multi-factor/" "bhy API"
    click PC "partial-conjunction/" "partial_conjunction API"
    click BHYH "bhy-hierarchical/" "bhy_hierarchical API"
    click LM "metrics/#factrix.list_metrics" "list_metrics API"
    click CMP "compare/" "compare API"
    click ID "inspect-data/" "inspect_data API"
```

Click any node to jump to its API page.

**Edge convention:**

- **Solid `==>` — hard signature dependency.** The target's call signature accepts the source object.
    - `evaluate(data, metrics=...)` consumes `P` (data).
    - `bhy` / `partial_conjunction` / `bhy_hierarchical` consume `evaluate` outputs as `list[EvaluationResult]` (via `list(results.values())`).
- **Dashed `-.->` — suggested workflow.** The source is data-derived, but the target's signature differs in shape.

**Inference vs screening** (the two test-bearing stages above):

- **Inference** produces the primary statistical test — a *p*-value — for a **single** hypothesis. `evaluate` / `evaluate_horizons` do this **per factor** ("tests one factor"); the `slice_*_test` family does it **across slices** of one factor (sector, size, market regime). Neither corrects for testing more than one thing.
- **Screening** is what you add when you test **many** factors at once: `bhy` / `partial_conjunction` / `bhy_hierarchical` take a `list[EvaluationResult]` and control the false-discovery rate across the family ("screens a thousand"). Screening consumes inference output; it does not re-run the test.
- `by_slice` / `compare` are **descriptive** — they arrange or rank results into a view, without adding a test of their own.

---

## Typical patterns

| Goal | Pipeline |
|---|---|
| Single-factor/multi-factor inference | `evaluate(data, metrics=...)` → `dict[str, EvaluationResult]` |
| Multi-horizon sweep | `evaluate_horizons(data, metrics=..., forward_periods=[...])` → `list[EvaluationResult]` |
| Slice exploration (single axis) | `by_slice(data, metric, by="...", factor_col="...")` → `dict[str, EvaluationResult]` |
| Slice statistical test (date-aligned) | `slice_pairwise_test(df, metric, by="...")` or `slice_joint_test(...)` → pairwise / omnibus test result |
| Slice statistical test (date-disjoint) | `slice_period_pairwise_test(...)` or `slice_period_joint_test(...)` → pairwise / omnibus across regimes / calendar periods |
| Metric catalog discovery | `list_metrics()` → family-grouped `dict` of specs |
| Per-panel applicability | `inspect_data(data)` → `.usable` / `.degraded` / `.unusable` |
| Multi-factor screening with FDR | `evaluate(...)` → `multi_factor.bhy(list(results.values()), metrics=[...])` |
| Cross-factor leaderboard | `compare(list(results.values()), metrics=[...])` → `pl.DataFrame` |

See the [Slice analysis guide](../guides/slice-analysis.md) for the slice surface end-to-end.

---

## Entry points

| Page | Category | What it is | When to read |
|---|---|---|---|
| [`evaluate`](evaluate.md) | Inference (per factor) | Single dispatch entry — runs the registered metrics on a panel and returns the evaluation results. | Running an analysis. |
| [`evaluate_horizons`](evaluate.md#factrix.evaluate_horizons) | Inference (per factor) | Sweep `evaluate` across several overlap horizons of one raw panel. | Multi-horizon analysis / sweeping. |
| [`by_slice`](by-slice.md) | Descriptive view | Partition a panel on a column and run `evaluate` per slice; returns `dict[str, EvaluationResult]`. | Per-slice metric exploration. |
| [`slice_*_test` family](slice-test.md) | Inference (across slices) | `slice_pairwise_test` / `slice_joint_test` (date-aligned) and `slice_period_pairwise_test` / `slice_period_joint_test` (date-disjoint regimes): pairwise / omnibus tests over slice families. | Testing whether slice means differ. |
| [`multi_factor`](multi-factor.md) | Screening (FDR) | Module-level overview of collection-level FDR functions. | Multi-factor FDR screening overview. |
| [`bhy`](bhy.md) | Screening (FDR) | Benjamini-Hochberg-Yekutieli step-up FDR. | Screening candidate factors. |
| [`partial_conjunction`](partial-conjunction.md) | Screening (FDR) | k-of-m partial conjunction screening. | "Factor passes in ≥ k of m contexts." |
| [`bhy_hierarchical`](bhy-hierarchical.md) | Screening (FDR) | Two-stage hierarchical BHY FDR. | Grouped / nested-context screening. |
| [`compare`](compare.md) | Descriptive view | Cross-factor leaderboard — stacks evaluation results into a `pl.DataFrame`. | Ranking candidate factors. |
| [`list_metrics`](metrics/index.md#factrix.list_metrics) | Introspection | Family-grouped catalog of standalone public metrics. | Browsing the metric catalog. |
| [`inspect_data`](inspect-data.md) | Introspection | Inspects a panel's applicability metrics. | Pre-flight check on data dimensions. |
| [`Metrics`](metrics/index.md) | Catalogue | Per-module reference for every public function under `factrix.metrics`. | Calling a standalone metric directly. |
| [`stats`](stats.md) | Catalogue | Statistical estimators and HAC/bootstrap utilities. | Under-the-hood statistical details. |

---

## Supporting surface

| Page | What it is |
|---|---|
| [Panel schema](panel-schema.md) | The four-column input contract every panel-consuming function depends on. |
| [`EvaluationResult`](evaluation-results.md) | The bundle result returned by `evaluate`. Includes groups, metric results, and warnings. |
| [`datasets`](datasets.md) | Synthetic panels for testing and examples. |
| [`preprocess`](preprocess.md) | Helper functions for preprocessing (e.g. computing forward returns). |

## Naming convention

Sidebar entries mirror the actual Python identifier:

| Sidebar entry | Identifier kind | Example call |
|---|---|---|
| `EvaluationResult` | Class | `fx.EvaluationResult` |
| `evaluate`, `inspect_data` | Function | `fx.evaluate(data, metrics=...)` |
| `multi_factor`, `datasets`, `Metrics` | Module | `fx.multi_factor.bhy(list(results.values()), metrics=[...])` |
