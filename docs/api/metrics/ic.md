# IC

Information Coefficient (Spearman rank correlation between factor and
forward return) and its variants. Cross-sectional first; significance
via NW HAC or non-overlapping cross-asset *t*.

→ Formula and references:
[NW HAC](../../reference/statistical-methods.md#nw-hac)
(`ic_newey_west`),
[BHY across regimes](../../reference/statistical-methods.md#bhy)
(`regime_ic`).

::: factrix.metrics.ic
    options:
      members:
        - ic
        - ic_newey_west
        - ic_ir
        - regime_ic
        - compute_ic

!!! note "Regime analysis: two roles"
    `regime_ic` is the **curated wrapper** that bundles the IC-specific
    cross-regime test (BHY adjustment + min-\|t\| + direction
    consistency). For a generic per-regime dispatcher with no
    second-layer inference, see [`by_regime`](../by-regime.md) or the
    [Regime analysis guide](../../guides/regime-analysis.md).
    (Older docs called these roles "Layer A" / "Layer B"; renamed in
    [#157](https://github.com/awwesomeman/factrix/issues/157).)

## See also

- [`by_regime`](../by-regime.md) — dispatcher for any registered metric.
- [Regime analysis guide](../../guides/regime-analysis.md) — when to use the dispatcher vs a curated wrapper.
- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
