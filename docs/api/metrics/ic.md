# IC

Information Coefficient (Spearman rank correlation between factor and
forward return) and its variants. Cross-sectional first; significance
via NW HAC or non-overlapping cross-asset *t*.

::: factrix.metrics.ic
    options:
      members:
        - ic
        - ic_newey_west
        - ic_ir
        - regime_ic
        - multi_horizon_ic
        - compute_ic

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
