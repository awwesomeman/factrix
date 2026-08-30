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

```python title="Illustrative"
screen = fx.multi_factor.partial_conjunction_across_metrics(
    results,
    metrics=["ic", "beta", "spread"],
    min_pass=2,
    q=0.05,
)

screen.to_frame()
# factor | pc_p | adj_p | survived | active
#        | n_tests | n_passed_uncorr
```

This differs from [`bhy_across_metrics`](bhy-across-metrics.md): pooled BHY
selects factor × metric cells, while partial conjunction returns factor
identities supported by at least `k` endpoints.

## Placeholder endpoints leave `m`

The declared metric list fixes the endpoints you may claim on; `m` is how many
of them actually ran a test. A placeholder endpoint (`insufficient_*`
short-circuit or `degenerate_variance`) never ran one, so it is excluded from
the k-of-m denominator rather than entering it at `p=1` — the same policy every
other screening function applies, so the same degenerate cell costs the same
whichever verb sees it. See [the module-level
policy](multi-factor.md#placeholder-hypotheses).

The endpoint stays in `hypotheses` for audit, and `n_hypotheses_inactive`
reports how many were dropped. If fewer than `min_pass` real endpoints remain,
that identity cannot support the claim: it stays visible with `active=False`
and empty PC/adjusted p-values, and does not enter the outer BHY family.

`min_pass` must be declared before the p-values are seen — `(m - k + 1) *
p_(k)` is not monotone in `k` (see
[`partial_conjunction`](partial-conjunction.md#declare-min_pass-before-you-look-at-the-p-values)).

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
| `n_tests` | Count of real endpoints m per identity — the k-of-m denominator |
| `n_hypotheses_inactive` | Placeholder endpoints excluded before adjustment |
| `n_identities` | Identities entering the outer BHY family |

::: factrix.multi_factor.CrossMetricPartialConjunctionResult
    options:
      show_root_toc_entry: false
      heading_level: 3
