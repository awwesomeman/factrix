---
title: Individual continuous
---

Metrics for the `Individual × Continuous` cell — one continuous factor
score per `(date, asset_id)`, evaluated cross-sectionally per date and
aggregated over time.

| Metric | Page |
|---|---|
| Spearman rank correlation, the canonical signal-quality summary | [`ic`](ic.md) |
| Per-date ordinary least squares (OLS) slope $\lambda$, Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) $t$ on its mean | [`fm_beta`](fm_beta.md) |
| Long-short quintile spread (equal- and value-weighted) | [`quantile`](quantile.md) |
| Quintile-return monotonicity rank test | [`monotonicity`](monotonicity.md) |
| HHI-style top-bucket concentration | [`concentration`](concentration.md) |
| Turnover, breakeven cost, net spread | [`tradability`](tradability.md) |
| Spanning $\alpha$ vs an existing factor pool | [`spanning`](spanning.md) |

`ic` and `fm_beta` are the two **inferential** entry points (each
carries a $p$-value); the rest are **descriptive / risk** diagnostics
that profile the same factor without contributing to the inferential
test.

For when each applies see
[Reference § Metric applicability](../../reference/metric-applicability.md).
The discipline behind the inferential SEs (NW HAC bandwidth, Shanken EIV)
is consolidated in
[Reference § Statistical methods](../../reference/statistical-methods.md).
