---
title: factrix.multi_factor.partial_conjunction_across_metrics
---

::: factrix.multi_factor.partial_conjunction_across_metrics

<hr>

## When to use it

Use this procedure for a predeclared factor-level claim such as "the factor has
signal on at least two of IC, beta, and spread." Metric labels are the fixed
condition axis: each factor identity receives one k-of-m partial-conjunction
p-value, followed by BHY across identities.

```python
screen = fx.multi_factor.partial_conjunction_across_metrics(
    results,
    metrics=["ic", "beta", "spread"],
    min_pass=2,
    q=0.05,
)

screen.to_frame()
# factor | pc_p | adj_p | survived | active
#        | n_tests | n_active | n_passed_uncorr
```

This differs from [`bhy_across_metrics`](bhy-across-metrics.md): pooled BHY
selects factor × metric cells, while partial conjunction returns factor
identities supported by at least `k` endpoints.

## Fixed-m data-shortage rule

The declared metric list fixes `m`. An `insufficient_*` endpoint is retained in
`hypotheses` and conservatively enters the k-of-m calculation as `p=1`; it is
never deleted in a way that would lower the confirmation bar. If fewer than
`min_pass` endpoints are active, that identity remains visible with
`active=False` and empty PC/adjusted p-values, and does not enter the outer BHY
family.

Descriptive endpoints and other invalid p-values fail loudly. The function does
not implement `min_pass=1` any-metric promotion.

## Result fields

| Field | Meaning |
|---|---|
| `entries` | Every tested factor identity, in input order |
| `hypotheses` | Underlying result × metric cells retained for audit |
| `pc_p_all` | Raw k-of-m p-value per identity; NaN when fewer than `k` endpoints are active |
| `adj_p_all` | BHY-adjusted PC p-value per identity |
| `survivors` / `adj_p` | Passing factor identities and their adjusted p-values |
| `metrics` / `min_pass` | Declared m endpoints and required k |
| `n_tests` | Fixed condition count m per identity |
| `n_active` | Computable endpoint count per identity |
| `n_identities` | Identities entering the outer BHY family |

::: factrix.multi_factor.CrossMetricPartialConjunctionResult
    options:
      show_root_toc_entry: false
      heading_level: 3
