---
title: Common continuous
---

Metrics for the `Common × Continuous` cell — one time-series factor
broadcast across `N` assets (a market-wide signal: VIX, USD index, oil,
sentiment). Aggregation is **time-series first**: per-asset ordinary least squares (OLS) $\beta$
over all dates, then a cross-asset $t$ on the mean of the per-asset
betas.

| Metric | Page |
|---|---|
| Cross-asset $t$ on per-asset $\beta_i$ (BJS aggregation) | [`ts_beta`](ts_beta.md) |
| Spread between top- and bottom-bucket assets sorted by $\beta_i$ | [`ts_quantile`](ts_quantile.md) |
| Sign-asymmetric slopes (positive vs negative regimes) | [`ts_asymmetry`](ts_asymmetry.md) |

`ts_beta` carries the cross-asset significance test; `ts_quantile` and
`ts_asymmetry` are descriptive profile diagnostics.

At `N == 1` the cross-asset $t$ degenerates; factrix auto-routes to a
TIMESERIES single-series test (null: $\beta = 0$, **not**
$\mathbb{E}[\beta] = 0$). The statistic lands in the same
`MetricResult.stat` field in both modes, but the null differs —
`metadata["method"]` records which test ran.

The `Common × Sparse` cell swaps this continuous regressor for a
`{0, R}` broadcast event dummy (`R` unrestricted; `{0, 1}` for a pure
event flag is the simplest form). It does **not** reuse this
time-series-first OLS-$\beta$ flow — a broadcast sparse dummy is
evaluated through the event-time metrics under
[Common sparse](common-sparse.md).
