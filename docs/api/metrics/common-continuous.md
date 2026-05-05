# Common continuous

Metrics for the `Common × Continuous` cell — one time-series factor
broadcast across `N` assets (a market-wide signal: VIX, USD index, oil,
sentiment). Aggregation is **time-series first**: per-asset OLS $\beta$
over all dates, then a cross-asset $t$ on the mean of the per-asset
betas.

| Metric | Role | Page |
|---|---|---|
| Cross-asset $t$ on per-asset $\beta_i$ (BJS aggregation) | Primary | [`ts_beta`](ts_beta.md) |
| Spread between top- and bottom-bucket assets sorted by $\beta_i$ | Profile | [`ts_quantile`](ts_quantile.md) |
| Sign-asymmetric slopes (positive vs negative regimes) | Profile | [`ts_asymmetry`](ts_asymmetry.md) |

At `N == 1` the cross-asset $t$ degenerates; factrix auto-routes to a
TIMESERIES single-series test (null: $\beta = 0$, **not**
$\mathbb{E}[\beta] = 0$). Same `StatCode.TS_BETA` identifier, different
statistical meaning — see
[`profile.mode`](../factor-profile.md) for which path ran.

The `Common × Sparse` cell shares the time-series-first aggregation
shape but with a `{-1, 0, +1}` broadcast dummy in place of the
continuous regressor; metrics live under
[Individual sparse](individual-sparse.md).
