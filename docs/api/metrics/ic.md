# IC

Information Coefficient (Spearman rank correlation between factor and
forward return) and its variants. Cross-sectional first; significance
via NW HAC or non-overlapping cross-asset *t*.

→ Formula and references:
[NW HAC](../../reference/statistical-methods.md#nw-hac)
(`ic_newey_west`).

::: factrix.metrics.ic
    options:
      members:
        - ic
        - ic_newey_west
        - ic_ir
        - compute_ic

!!! note "Cross-regime IC analysis"
    For per-regime IC summaries, use [`by_slice`](../by-slice.md) on an
    IC frame joined with regime labels. For inferential contrasts
    (pairwise Wald χ² + Holm / Romano-Wolf adjusted p), use
    [`slice_pairwise_test`](../slice-test.md). The metric-specific
    `regime_ic` callable and `by_regime` dispatcher were removed in
    v0.12.0; see the [Regime analysis guide](../../guides/regime-analysis.md).

## See also

- [`by_slice`](../by-slice.md) — axis-agnostic slice dispatcher.
- [`slice_pairwise_test` / `slice_joint_test`](../slice-test.md) — cross-slice inference verb pair.
- [Regime analysis guide](../../guides/regime-analysis.md) — slicing and cross-slice inference end-to-end.
- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
