---
title: factrix.multi_factor
---

# factrix.multi_factor

Collection-level false-discovery-rate (FDR) control across a list of
[`FactorProfile`][factrix.FactorProfile] objects. Use after `evaluate`
has produced one profile per candidate factor (or per factor × context
combination): the functions in this module adjust per-factor p-values
for multiple testing under the dependence structure that factor pools
exhibit by construction.

This page is a module-level index. Each function has its own page
covering call shape, parameters, the `Survivors` return container,
and design rationale.

## Choosing a function

| Question you are asking | Function | Page |
|---|---|---|
| "Which factors in this candidate pool survive FDR ≤ q under arbitrary dependence?" | `bhy` | [api/bhy](bhy.md) |
| "Which factors are significant in at least `k` of `m` replication conditions?" | `partial_conjunction` | [api/partial-conjunction](partial-conjunction.md) |
| "Which factor *families* carry signal, and which factors within each surviving family survive?" | `bhy_hierarchical` | [api/bhy-hierarchical](bhy-hierarchical.md) |

`bhy` is the canonical entry point — start there unless you have an
explicit reason to claim partial-conjunction or hierarchical group
structure.

## See also

<div class="grid cards" markdown>

-   __Batch screening guide__

    ---

    End-to-end recipe wiring `evaluate` into the multi-factor FDR
    pipeline, including how to preserve `identity` / `context` across
    a candidate pool and how to choose between the three functions.

    [guides/batch-screening →](../guides/batch-screening.md)

-   __Statistical methods — multiple testing__

    ---

    Why Benjamini-Hochberg-Yekutieli (BHY) rather than Bayesian or reality-check / SPA bootstraps;
    positive regression dependence on a subset (PRDS) and the harmonic dependence correction.

    [reference/statistical-methods →](../reference/statistical-methods.md#2-multiple-testing-under-dependence)

</div>
