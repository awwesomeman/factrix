# factrix

Polars-native **factor signal validation toolkit** for quantitative finance.

factrix answers one question: **「這個因子在統計上真的有效嗎？」** It produces a single `primary_p` per factor cell from a NW HAC-corrected canonical procedure (IC / FM-λ / CAAR / TS-β), then applies Benjamini-Yekutieli FDR correction across factor families. It is a *signal validator*, not a backtest engine — feed surviving factors into downstream portfolio constructors.

[![GitHub](https://img.shields.io/badge/GitHub-factrix-blue?logo=github)](https://github.com/awwesomeman/factrix)

## Quick links

| Want to | Go to |
|---------|-------|
| Install and run a smoke test | [Get Started](getting-started/index.md) |
| Understand the three-axis design | [Concepts](getting-started/concepts.md) |
| Screen a batch of factors with BHY | [Batch screening](guides/batch-screening.md) |
| See API formulas and applicability | [Reference](reference/methodology.md) |
| Browse an end-to-end notebook | [Examples](examples/demo.ipynb) |
| Understand the internal architecture | [Architecture](development/architecture.md) |
