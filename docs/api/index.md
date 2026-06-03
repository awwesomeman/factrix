---
title: API
---

Reference for every public symbol exported from `factrix`.

## Function flow

```mermaid
flowchart LR
    P[panel + cfg]

    subgraph Compute
        EV[evaluate]
    end

    subgraph Screening["Screening (FDR)"]
        BHY[bhy]
        PC[partial_conjunction]
        BHYH[bhy_hierarchical]
    end

    subgraph Inference["Inference (no FDR)"]
        ST["slice_pairwise_test<br/>slice_joint_test"]
    end

    subgraph View["Descriptive view"]
        BS[by_slice]
        CMP[compare]
    end

    subgraph Introspection
        LM[list_metrics]
        LE[list_estimators]
        SC[suggest_config]
    end

    P ==> EV
    P -.-> BS
    P -.-> ST
    EV ==>|profiles| BHY
    EV ==>|profiles| PC
    EV ==>|profiles| BHYH
    EV ==>|profiles| CMP
    BHY ==>|survivors| CMP
    PC ==>|survivors| CMP
    BHYH ==>|survivors| CMP
    LM -.->|metric names| EV
    LE -.->|estimator names| BHY
    SC -.->|inferred cfg| EV

    click EV "evaluate/" "evaluate API"
    click BS "by-slice/" "by_slice API"
    click ST "slice-test/" "slice_pairwise_test / slice_joint_test API"
    click BHY "multi-factor/" "bhy API"
    click PC "partial-conjunction/" "partial_conjunction API"
    click BHYH "bhy-hierarchical/" "bhy_hierarchical API"
    click LM "list-metrics/" "list_metrics API"
    click LE "list-estimators/" "list_estimators API"
    click SC "suggest-config/" "suggest_config API"
    click CMP "compare/" "compare API"
```

Click any node to jump to its API page. Nodes are grouped into category subgraphs; the sixth category, **Compare-sensitivity** (`by_estimator`, #178), is not drawn — it lands in v0.14 once #263 unblocks HACEstimator parameter config + catalog expansion.

**Edge convention:**

- **Solid `==>` — hard signature dependency.** The target's call signature literally accepts the source object.
    - `evaluate(panel, cfg)` consumes `P` (the panel).
    - `bhy` / `partial_conjunction` / `bhy_hierarchical` consume `evaluate` outputs.
- **Dashed `-.->` — suggested workflow.** The source is panel-derived, but the target's signature differs in shape.
    - `by_slice` / `slice_pairwise_test` / `slice_joint_test` accept `(metric, metric_df, label=…)`, where `metric_df` is a per-date frame the caller builds from the panel (e.g. `compute_ic(panel)`).
    - `list_metrics` returns candidate names the caller forwards to `evaluate(metrics=[…])`.

---

## Typical patterns

| Goal | Pipeline |
|---|---|
| Single-factor inference | `evaluate(panel, cfg)` → read `primary_p` |
| Slice exploration (single axis) | `by_slice(metric, df, label="...")` → `SliceResult` |
| Slice statistical test | `slice_pairwise_test(metric, df, label="...")` or `slice_joint_test(...)` → pairwise / omnibus test result |
| Cell metric discovery | `list_metrics(scope, signal)` → names → `evaluate(metrics=[...])` |
| Multi-factor screening with false discovery rate (FDR) | `[evaluate(panel, cfg_i) for cfg_i in cfgs]` → `multi_factor.bhy(profiles)` |
| Cross-factor leaderboard | `compare(profiles)` / `compare(survivors)` → `pl.DataFrame` |

See the [Slice analysis guide](../guides/slice-analysis.md) for the slice surface end-to-end, and the [Batch screening with Benjamini-Hochberg-Yekutieli (BHY)](../guides/batch-screening.md) guide for the multi-factor screening workflow.

---

## Entry points

| Page | Category | What it is | When to read |
|---|---|---|---|
| [`evaluate`](evaluate.md) | Compute | Single dispatch entry — runs the registered procedure for a `(config, panel)` pair and returns the evaluation result. | Running an analysis. |
| [`by_slice`](by-slice.md) | Descriptive view | Slice a metric over a label column; returns a `SliceResult` with `.to_frame()` rendering. | Per-slice metric exploration. |
| [`slice_pairwise_test` / `slice_joint_test`](slice-test.md) | Inference (no FDR) | Statistical tests over slice families (pairwise / omnibus) with family-internal MTC. No cell-level FDR claim. | Testing whether slice means differ. |
| [`multi_factor`](multi-factor.md) | Screening (FDR) | `bhy(...)` for BHY FDR screening across a `Profile[]`; `expand_over=` opens hypothesis-dimension expansion. | Multi-factor FDR screening. |
| [`partial_conjunction`](partial-conjunction.md) | Screening (FDR) | k-of-m partial conjunction p-values (Benjamini-Heller 2008) → BHY. | "Factor X passes in ≥ k of m contexts." |
| [`bhy_hierarchical`](bhy-hierarchical.md) | Screening (FDR) | Hierarchical FDR (Yekutieli 2008) — outer BHY on group representatives, inner BHY within passing groups. | Factor families / nested-context structure. |
| [`compare`](compare.md) | Descriptive view | Cross-factor leaderboard — stacks evaluation / `Survivors` artifacts into a `pl.DataFrame` (no recompute). | Ranking N candidate factors. |
| [`list_metrics`](list-metrics.md) | Introspection | Programmatic discovery of standalone `factrix.metrics.*` callables applicable to a `(scope, signal)` cell. | Picking a follow-up metric after `evaluate()`. |
| [`list_estimators`](list-estimators.md) | Introspection | Estimators applicable to a cell — inference-side twin of `list_metrics`. | Picking `estimator=` for screening functions. |
| [`Metrics`](metrics/index.md) | Catalogue | Per-module reference for every public function under `factrix.metrics`. | Calling a standalone metric directly. |
| [`stats`](stats.md) | Catalogue | Estimator catalogue (`NeweyWest` / `HansenHodrick` / `WaldNWCluster` / `WaldTwoWayCluster` / `BlockBootstrap`), StatCode pairs, FDR / bootstrap utilities. | Picking an inference method to pass through `estimator=`. |

---

## Supporting surface

| Page | What it is |
|---|---|
| [Panel schema](panel-schema.md) | The four-column input contract every panel-consuming function depends on. |
| [`MetricResult`](metric-output.md) | Common wrapper returned by every standalone metric — `value`, `p`, `stat`, `metadata`. |
| [`datasets`](datasets.md) | Synthetic panels (`make_cs_panel`, `make_event_panel`) for smoke tests and docs examples. |

## Naming convention

Sidebar entries mirror the actual Python identifier — the case
distinction is intentional, not inconsistent:

| Sidebar entry | Identifier kind | Example call |
|---|---|---|
| `MetricResult` | Class | `fx.MetricResult` |
| `evaluate`, `list_metrics` | Function | `fx.evaluate(panel, cfg)` |
| `multi_factor`, `datasets`, `Metrics` (and submodules) | Module | `fx.multi_factor.bhy(profiles)` |
