---
title: factrix.multi_factor
---

# factrix.multi_factor

Collection-level false-discovery-rate (FDR) control across a list of
`EvaluationResult` objects. Use after `evaluate`
has produced results for candidate factors (or per factor × params
combinations): the functions in this module adjust traceable
factor/context/metric hypotheses for multiple testing under the dependence
structure that factor pools exhibit by construction.

This page is a module-level index. Each function has its own page
covering call shape, parameters, the result containers,
and design rationale.

## Choosing a function

| Question you are asking | Function | Page |
|---|---|---|
| "Which factors in this candidate pool survive FDR ≤ q under arbitrary dependence?" | `bhy` | [api/bhy](bhy.md) |
| "Which factor × metric cells survive one pooled FDR family?" | `bhy_across_metrics` | [api/bhy-across-metrics](bhy-across-metrics.md) |
| "Which factors are significant in at least `k` of `m` replication conditions?" | `partial_conjunction` | [api/partial-conjunction](partial-conjunction.md) |
| "Which factors have signal on at least `k` of `m` predeclared metrics?" | `partial_conjunction_across_metrics` | [api/partial-conjunction-across-metrics](partial-conjunction-across-metrics.md) |
| "Which factor *families* carry signal, and which factors within each surviving family survive?" | `bhy_hierarchical` | [api/bhy-hierarchical](bhy-hierarchical.md) |

Start with `bhy` when one metric defines the screen. Use a cross-metric function
only when the predeclared selection rule may choose among metric labels or
requires confirmation on at least `k` endpoints; use the hierarchical function
only for a predeclared group structure.

## Inactive candidates

A metric output that never ran a test is *inactive*. Two shapes qualify: a
data-shortage short-circuit (`reason` starting `insufficient_`) and a
`degenerate_variance` result (observations but no dispersion, so no statistic
exists). An inactive candidate can never be rejected — the only open question
is whether it still counts toward `m`.

Every function on this page answers that with the same explicit kwarg,
`inactive_policy`, and the same default.

| `inactive_policy` | What `m` (and `G`, and a k-of-m denominator) counts | What it assumes |
|---|---|---|
| `"count"` (default) | Every declared candidate. The inactive ones participate at an inert `p = 1`. | Nothing beyond the family declaration itself. |
| `"exclude"` | Only the candidates that produced a statistic. | That the activity filter is **independent of the p-values or pre-specified** — an assertion you make, not one factrix can check. |

### Why `"count"` is the default

BHY is monotone in its p-vector, so an inert `p = 1` can only raise — never
lower — another candidate's adjusted p. Counting inactive candidates is
therefore the conservative reading of a declared family, and it keeps the FDR
statement free of any claim about *why* a candidate came back inactive.
Missingness and degeneracy are commonly factor- and data-dependent: the
candidates that ran out of observations are not a random subset of the ones you
declared.

### The size of the difference

One active hypothesis at $p = 0.01$ submitted alongside nine
`insufficient_ic_periods` candidates:

| Policy | `m` | Adjusted $p$ for the live hypothesis | Survives at $q = 0.05$? |
|---|---|---|---|
| `"count"` | 10 | **0.293** | No |
| `"exclude"` | 1 | **0.010** | Yes |

The BY correction over ten hypotheses is
$0.01 \times 10 \times c(10) = 0.01 \times 10 \times 2.929 = 0.293$; over one
it is $0.01$ itself. Same input, opposite verdict — which is why the choice is
a declaration rather than a silent default. Both numbers are computed, not
quoted: see `test_the_documented_numeric_example_is_the_one_the_code_produces`
in `tests/test_inactive_policy.py`.

### What every result discloses

Under either policy the inactive candidates stay in `entries` for audit, and
every screening result carries a `family`
([`FamilyAccounting`][factrix.multi_factor.FamilyAccounting]) reporting the
four counts and the policy that produced them:

| Field | Meaning |
|---|---|
| `n_candidates_declared` | Candidate cells submitted — the declared family. |
| `n_tests_computed` | Candidates that produced a test statistic. |
| `n_inactive` | Candidates that never ran a test. |
| `n_tests_adjusted` | Candidates that actually entered an adjustment. |
| `policy` | The `inactive_policy` the call ran under. |

They also appear in the `repr` / notebook rendering, so a screen never hides
the gap between the family you declared and the family that was adjusted:

```text title="Illustrative"
BhyResult(metric=ic, n=0, q=0.05, family(declared=10, computed=1, inactive=9, adjusted=10, policy='count'))
```

The per-bucket / per-identity mapping beside it is named `family_size`, not
`n_tests`: under `"count"` not every member of it is a test that ran.

::: factrix.multi_factor.FamilyAccounting
    options:
      show_root_toc_entry: false
      heading_level: 3

## See also

<div class="grid cards" markdown>

-   **Large-scale evaluation**

    ---

    How to structure factor screens using a user-side batched loop with Polars LazyFrames.

    [guides/large-scale-evaluation →](../guides/large-scale-evaluation.md)

-   **Statistical methods — multiple testing**

    ---

    Why Benjamini-Hochberg-Yekutieli (BHY) rather than Bayesian or reality-check / SPA bootstraps;
    positive regression dependence on a subset (PRDS) and the harmonic dependence correction.

    [reference/statistical-methods →](../reference/statistical-methods.md#2-multiple-testing-under-dependence)

</div>
