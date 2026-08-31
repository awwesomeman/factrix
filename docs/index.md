---
title: factrix
---

<h3 align="center"><b>Tests one factor. Screens a thousand.</b></h3>

**Does this factor have predictive value?**

factrix is a Polars-native toolkit for factor inference and batch screening.
It provides methods for cross-sectional, event, and common factors while
keeping estimates, p-values, diagnostics, and warnings explicit.

[![GitHub](https://img.shields.io/badge/GitHub-factrix-blue?logo=github)](https://github.com/awwesomeman/factrix)

## Start here

| If you want | Go to |
|---|---|
| Install and run a complete example | [Installation](getting-started/install.md) · [Quickstart](getting-started/quickstart.md) |
| Prepare and validate a panel | [Preparing data](guides/preparing-data.md) |
| Match a research question to a metric | [Choosing a metric](guides/choosing-metric.md) |
| Understand factor types and dispatch | [Concepts](getting-started/concepts.md) · [Panel vs timeseries](guides/panel-timeseries.md) |
| Read estimates, metadata, and warnings | [Reading results](guides/reading-results.md) |
| Screen many factors with FDR control | [Multi-factor screening](examples/multi_factor_screening.md) |
| Look up a metric or public symbol | [Metric applicability](reference/metric-applicability.md) · [API reference](api/index.md) |
| Decide whether factrix fits your workflow | [Where factrix fits](where-factrix-fits.md) |

## Workflow

1. Prepare a long Polars frame on a consistent period grid.
2. Choose metrics that match the factor scope, density, and data structure.
3. Run `evaluate()` and inspect each metric's estimate, sample metadata, and warnings.
4. When screening a family of candidates, apply the multi-factor FDR helpers.

The [API reference](api/index.md) lists every entry point. Contributors should
start with the [development guide](development/contributing.md) and
[architecture](development/architecture.md).
