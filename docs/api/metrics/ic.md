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

!!! note "Regime analysis: two layers"
    `regime_ic` is the **Layer B** wrapper that bundles the IC-specific
    cross-regime test (BHY adjustment + min-\|t\| + direction
    consistency). For a generic per-regime dispatcher with no
    second-layer inference, see [`by_regime`](../by-regime.md) or the
    [Regime analysis guide](../../guides/regime-analysis.md).

## See also

- [`by_regime`](../by-regime.md) — Layer A dispatcher for any registered metric.
- [Regime analysis guide](../../guides/regime-analysis.md) — when to use Layer A vs Layer B.
- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
