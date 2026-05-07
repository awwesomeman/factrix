# fama_macbeth

Per-date cross-sectional OLS slope λ then NW HAC *t* on the λ series
(Fama-MacBeth 1973). Pooled OLS variant with date-clustered SE; β-sign
consistency check.

→ Formula and references: stage-2 HAC SE on E[λ] —
[NW HAC](../../reference/statistical-methods.md#nw-hac)
(includes the Kan-Zhang / Shanken EIV correction invoked by
`is_estimated_factor=True`).

::: factrix.metrics.fama_macbeth
    options:
      members:
        - fama_macbeth
        - pooled_ols
        - beta_sign_consistency
        - compute_fm_betas

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
