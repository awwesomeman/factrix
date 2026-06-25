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

These metrics test the **cross-asset** distribution of per-asset
$\beta_i$, so they need $N \ge 2$ assets. At `N == 1` there is no
cross-section: the cell is `PANEL`, so `evaluate` raises
`IncompatibleAxisError` (or returns NaN + `structure_mismatch` under
`strict=False`) rather than running. For single-asset time-series
questions use the scope-agnostic TIMESERIES metrics (`hit_rate`,
`oos_decay`, `ic_trend`, `directional_hit_rate`) or `ic` on the series.

The `Common × Sparse` cell swaps this continuous regressor for a
`{0, R}` broadcast event dummy (`R` unrestricted; `{0, 1}` for a pure
event flag is the simplest form). It does **not** reuse this
time-series-first OLS-$\beta$ flow — a broadcast sparse dummy is
evaluated through the event-time metrics under
[Common sparse](common-sparse.md).
