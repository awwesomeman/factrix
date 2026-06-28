---
title: Series diagnostics
---

Axis-agnostic diagnostics that operate on any `(date, value)` series produced by
an upstream cell metric — information coefficient (IC) time series, $\beta$ time
series, CAAR time series, an external factor return, etc. They do not care which
cell produced the series.

!!! warning "Not the same as `DataStructure.TIMESERIES`"
    `DataStructure.TIMESERIES` is the dispatch structure for `n_assets == 1` (set on `EvaluationResult.cell`); the metrics on this page run on a `(date, value)` series regardless of which structure produced it.

For the single-asset dense `evaluate()` path, use
[`predictive_beta`](predictive_beta.md). It consumes the long panel directly and
tests the predictive-regression slope with Newey-West HAC inference.

| Metric | Page |
|---|---|
| Out-of-sample decay across multiple splits | [`oos`](oos_decay.md) |
| Theil-Sen monotonic trend with augmented Dickey-Fuller (ADF) persistence flag | [`trend`](trend.md) |
| Sign-consistency / hit rate of the series | [`hit_rate`](hit_rate.md) |

All three are descriptive robustness diagnostics. Use them against the IC series from [`ic.compute_ic`](ic.md), the CAAR series from [`caar.compute_caar`](caar.md), or any externally constructed `(date, value)` DataFrame to layer decay, drift, and sign-stability checks on top of the cell's inferential test.

[`directional_hit_rate`](directional_hit_rate.md) sits next to these sign
diagnostics in the navigation, but it is **not** a `(date, value)` helper: it
consumes the long panel `(date, asset_id, factor, forward_return)` and tests
whether `sign(factor)` predicts `sign(forward_return)`.
