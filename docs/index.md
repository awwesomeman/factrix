---
title: factrix
---

**Does this factor possess predictive edge?**

factrix is the first Polars-native Python toolkit that picks the right statistical test for each factor type. Cross-sectional, event, common factor — each gets the tests that fit its data-generating process.

[![GitHub](https://img.shields.io/badge/GitHub-factrix-blue?logo=github)](https://github.com/awwesomeman/factrix)

## What ships

factrix exposes a small set of functions organised by category:

- **Compute** — `evaluate`
- **Screening (false discovery rate (FDR))** — `bhy`, `partial_conjunction`, `bhy_hierarchical`
- **Inference (no FDR)** — `slice_pairwise_test`, `slice_joint_test`
- **Descriptive view** — `by_slice`, `compare`
- **Introspection** — `list_metrics`, `inspect_data`

See the [API reference landing](api/index.md) for the function-flow graph, edge conventions, and the full entry-point table with category, signature summary, and when-to-reach-for guidance.

---

## Quick links

| If you want | Go to |
|---|---|
| **Install and run a smoke test** | [Installation](getting-started/install.md) · [Quickstart](getting-started/quickstart.md) |
| **Understand the three-axis design** (scope / signal / metric) | [Concepts](getting-started/concepts.md) |
| **Compare factrix against alphalens / qlib / peers** | [Where factrix fits](where-factrix-fits.md) |
| **Screen a batch of factors with Benjamini-Hochberg-Yekutieli (BHY)** | [BHY FDR screening](api/bhy.md) |
| **Slice any metric by regime / universe / sector** | [Slice analysis](guides/slice-analysis.md) |
| **Look up formulas and applicability** | [Metric applicability](reference/metric-applicability.md) |
| **Read every public symbol** | [API reference](api/index.md) |
| **Browse runnable notebooks** | [Examples](examples/index.md) |
| **Read the internal architecture** | [Architecture](development/architecture.md) |
