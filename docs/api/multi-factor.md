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
