# Series tools

Axis-agnostic metrics that operate on any `(date, value)` series
produced by an upstream cell metric — IC time series, $\beta$ time
series, CAAR time series, an external factor return, etc. They do not
care which cell produced the series.

| Metric | Role | Page |
|---|---|---|
| Out-of-sample decay across multiple splits | Robustness | [`oos`](oos.md) |
| Theil-Sen monotonic trend with ADF persistence flag | Diagnostic | [`trend`](trend.md) |
| Sign-consistency / hit rate of the series | Diagnostic | [`hit_rate`](hit_rate.md) |

Use these against the IC series from [`ic.compute_ic`](ic.md), the
CAAR series from [`caar.compute_caar`](caar.md), or any externally
constructed `(date, value)` DataFrame to layer decay, drift, and
sign-stability checks on top of the cell's primary inferential test.
