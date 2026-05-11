# factrix

**Does this factor possess predictive edge?** factrix is the first Polars-native Python toolkit that picks the right statistical test for each factor type. Cross-sectional, event, common factor — each gets the tests that fit its data-generating process.

[![GitHub](https://img.shields.io/badge/GitHub-factrix-blue?logo=github)](https://github.com/awwesomeman/factrix)

## Verb map

```mermaid
flowchart LR
    P[panel + cfg]
    EV[evaluate]
    RM[run_metrics]
    BS[by_slice]
    ST["slice_pairwise_test<br/>slice_joint_test"]
    BHY{{multi_factor.bhy}}
    LM[/list_metrics/]

    P ==> EV
    P ==> RM
    P -.-> BS
    P -.-> ST
    EV ==>|profiles| BHY
    LM -.->|metric names| RM

    classDef compute fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef decision fill:#fce4ec,stroke:#c2185b,color:#000
    classDef view fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef introspect fill:#fff9c4,stroke:#f9a825,color:#000
    class EV,RM compute
    class BHY decision
    class BS,ST view
    class LM introspect

    click EV "api/evaluate/" "evaluate API"
    click RM "api/run-metrics/" "run_metrics API"
    click BS "api/by-slice/" "by_slice API"
    click ST "api/slice-test/" "slice_pairwise_test / slice_joint_test API"
    click BHY "api/multi-factor/" "multi_factor.bhy API"
    click LM "api/list-metrics/" "list_metrics API"
```

The seven shipped verbs, coloured by category — **compute** (blue, produce primary artefacts), **decision** (pink, FDR), **view** (purple, slice surface), **introspection** (yellow). Solid arrow = hard signature dependency, dashed = suggested workflow. Click any node to jump to its API page. Full edge convention and the four future-design verbs (`compare`, `robustness`, `bhy_hierarchical`, `partial_conjunction` per #148) are described on the [API reference landing](api/index.md).

## Quick links

| If you want | Go to |
|---|---|
| **Install and run a smoke test** | [Installation](getting-started/install.md) · [Quickstart](getting-started/quickstart.md) |
| **Understand the three-axis design** (scope / signal / metric) | [Concepts](getting-started/concepts.md) |
| **Compare factrix against alphalens / qlib / peers** | [Where factrix fits](where-factrix-fits.md) |
| **Screen a batch of factors with BHY** | [Batch screening](guides/batch-screening.md) |
| **Slice any metric by regime / universe / sector** | [Slice analysis](guides/slice-analysis.md) |
| **Look up formulas and applicability** | [Metric applicability](reference/metric-applicability.md) |
| **Read every public symbol** | [API reference](api/index.md) |
| **Browse runnable notebooks** | [Examples](examples/index.md) |
| **Read the internal architecture** | [Architecture](development/architecture.md) |
