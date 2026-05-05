# quantile

Quantile-bucket long-short spread on a panel: per-date
top-minus-bottom return → spread series, then non-overlapping *t* on
its mean. Equal-weight and value-weight variants.

::: factrix.metrics.quantile
    options:
      members:
        - quantile_spread
        - quantile_spread_vw
        - compute_spread_series
        - compute_group_returns

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
