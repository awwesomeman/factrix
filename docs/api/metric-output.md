# MetricOutput

The unified return type produced by every metric in
[`factrix.metrics`](metrics/index.md). A single dataclass carrying
the canonical scalar (`value`), the test statistic (`stat`), the
significance marker, and a `metadata` dict for everything else
(p-value, method label, sample-size diagnostics, short-circuit
reason). All metrics — cell-canonical or auxiliary — return this
shape so downstream code can treat every metric uniformly.

::: factrix.MetricOutput
