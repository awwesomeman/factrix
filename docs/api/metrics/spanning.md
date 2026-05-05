# spanning

Spanning regression — single-factor alpha against base factors, and
greedy forward selection over a pool. Operates on factor return time
series (quantile spread series), not IC.

::: factrix.metrics.spanning
    options:
      members:
        - spanning_alpha
        - greedy_forward_selection
        - SpanningResult
        - ForwardSelectionResult

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
