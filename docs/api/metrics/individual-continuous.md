# Individual continuous

Metrics for the `Individual × Continuous` cell — one continuous factor
score per `(date, asset_id)`, evaluated cross-sectionally per date and
aggregated over time.

| Metric | Role | Page |
|---|---|---|
| Spearman rank correlation, the canonical signal-quality summary | Primary | [`ic`](ic.md) |
| Per-date OLS slope $\lambda$, NW HAC $t$ on its mean | Primary | [`fama_macbeth`](fama_macbeth.md) |
| Long-short quintile spread (equal- and value-weighted) | Profile | [`quantile`](quantile.md) |
| Quintile-return monotonicity rank test | Profile | [`monotonicity`](monotonicity.md) |
| HHI-style top-bucket concentration | Profile | [`concentration`](concentration.md) |
| Turnover, breakeven cost, net spread | Profile | [`tradability`](tradability.md) |
| Spanning $\alpha$ vs an existing factor pool | Profile | [`spanning`](spanning.md) |

`ic` and `fama_macbeth` are the two **inferential** entry points; the
rest are **profile / risk** diagnostics that describe the same factor
without contributing to the verdict.

For when each applies see
[Reference § Metric applicability](../../reference/metric-applicability.md).
The discipline behind the inferential SEs (NW HAC bandwidth, Shanken EIV)
is consolidated in
[Reference § Statistical methods](../../reference/statistical-methods.md).
