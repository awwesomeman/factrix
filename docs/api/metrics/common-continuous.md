---
title: Common continuous
---

Metrics for the `Common × Continuous` cell — one time-series factor
broadcast across `n_assets` assets (a market-wide signal: VIX, USD index, oil,
sentiment). Aggregation is **time-series first**: per-asset ordinary least squares (OLS) $\beta$
over all dates, then a cross-asset $t$ on the mean of the per-asset
betas.

| Metric | Page |
|---|---|
| Cross-asset $t$ on per-asset $\beta_i$ (BJS aggregation) | [`common_beta`](common_beta.md) |
| Positive / negative / neutral profile of per-asset $\beta_i$ | [`common_beta`](common_beta.md) |
| Spread between top- and bottom-bucket assets sorted by $\beta_i$ | [`common_quantile`](common_quantile.md) |
| Sign-asymmetric slopes (positive vs negative regimes) | [`common_asymmetry`](common_asymmetry.md) |

`common_beta` carries the cross-asset significance test; `common_quantile` and
`common_asymmetry` are descriptive profile diagnostics.

These metrics test the **cross-asset** distribution of per-asset
$\beta_i$, so they need `n_assets >= 2`. At `n_assets == 1` there is no
cross-section: the cell is `PANEL`, so `evaluate` raises
`IncompatibleAxisError` (or returns NaN + `structure_mismatch` under
`strict=False`) rather than running. For single-asset time-series
questions use [`predictive_beta`](predictive_beta.md) for the direct
dense predictive-regression slope, the panel-input `directional_hit_rate`
on `(date, asset_id, factor, forward_return)` for directional skill, or
the two-column series diagnostics (`positive_rate`, `oos_decay`, `ic_trend`) on
an explicit `(date, value)` series.

The `Common × Sparse` cell swaps this continuous regressor for a
`{0, R}` broadcast event dummy (`R` unrestricted; `{0, 1}` for a pure
event flag is the simplest form). It does **not** reuse this
time-series-first OLS-$\beta$ flow — a broadcast sparse dummy is
evaluated through the event-time metrics under
[Common sparse](common-sparse.md).
