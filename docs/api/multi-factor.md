# multi_factor

Collection-level FDR control across factor profiles. Use after
`evaluate` has produced one profile per candidate factor — `bhy`
adjusts the per-factor $p$-values for multiple testing under arbitrary
dependence (factor pools are dependent by construction: 200 momentum
variants on the same return panel correlate, and a Bonferroni step
that assumes independence over-corrects).

## Call shape

```python
profiles = [fl.evaluate(panel_i, cfg) for panel_i in candidates]
adjusted = fl.multi_factor.bhy(profiles)
```

Returns the collection with `p_value_bhy_adjusted` populated on each
profile; per-factor `verdict()` reads the adjusted $p$ when present.

`bhy` implements the Benjamini-Yekutieli (2001) procedure with the
$c(m) = \sum_{i=1}^{m} 1/i$ dependence correction, valid under
arbitrary positive or negative dependence at the cost of a $1/\ln m$
shrinkage relative to plain BH. Plain Benjamini-Hochberg (1995) is
**not** offered: typical factor-pool dependence violates its PRDS
assumption.

For the design rationale (BHY rather than Bayesian or
reality-check / SPA bootstraps) see
[Reference § Statistical methods § Multiple-testing](../reference/statistical-methods.md#2-multiple-testing-under-dependence)
and [Development § Design notes § BHY](../development/design-notes.md#5-bhy-rather-than-bayesian-multiple-testing).

::: factrix.multi_factor.bhy
