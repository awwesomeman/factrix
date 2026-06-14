---
title: factrix.multi_factor.bhy
---

::: factrix.multi_factor.bhy

<hr>

## Use cases

<div class="grid cards" markdown>

-   **Screening a candidate pool with false discovery rate (FDR) control**

    ---

    Run `evaluate` over `m` candidate signals on the same return panel,
    feed the resulting list of `EvaluationResult` to `bhy`, and read the surviving subset off the
    [`BhyResult`][factrix.multi_factor.BhyResult] container. Controls
    FDR ≤ `q` under arbitrary dependence — the regime that matches a
    correlated factor pool (e.g. 200 momentum variants on one panel).

-   **Splitting into independent families**

    ---

    Pass `expand_over=(<context key>,)` to run one step-up per distinct
    bucket — for instance per `regime_id` or `universe_id` — under the
    Benjamini & Bogomolov (2014) selective-inference framing. Each
    bucket's `m`, threshold, and survivors stay self-contained.

-   **Auditing the family boundary**

    ---

    `BhyResult.expand_over` / `BhyResult.n_tests` / `BhyResult.q`
    record the family declared and the `m` fed into each bucket's
    step-up, so the FDR claim is self-contained in the return object.

</div>

## BhyResult attributes

The returned dictionary maps each mainstream metric label to a `BhyResult` container, which exposes:

| Attribute | Type | Meaning |
|---|---|---|
| `metric_name` | `str` | Name of the metric driving the screen. |
| `survivors` | `list[EvaluationResult]` | Input order, surviving subset. |
| `adj_p` | `np.ndarray` | Bucket-local Benjamini-Hochberg-Yekutieli (BHY)-adjusted p-value, index-aligned with `survivors`. |
| `q` | `float` | The nominal target FDR you passed. |
| `expand_over` | `tuple[str, ...]` | `()` for a single family; `("regime_id",)` etc. otherwise. |
| `n_tests` | `Mapping[tuple, int]` | `{(): N}` or `{bucket_key: m_per_bucket}`. |

Jupyter rendering surfaces a three-column text / HTML table of
`factor | adj_p`, plus an `expand_over_values` column
when buckets are declared.

::: factrix.multi_factor.BhyResult
    options:
      show_root_toc_entry: false
      heading_level: 3

## Parameters

| Kwarg | Default | Meaning |
|-------|---------|---------|
| `metrics` | (required) | `list[str]` of metric labels to run the FDR screen for. |
| `expand_over` | `()` | Context keys whose distinct value tuples split the input into independent step-ups. Names must live in `EvaluationResult.context` (except for the built-in `"forward_periods"`). |
| `q` | `0.05` | Nominal false discovery rate target. The Benjamini–Yekutieli $c(m)$ correction is applied internally — pass the level you actually want; do not pre-divide. |

### Identity vs context (anti-shopping defense)

Identity = `(factor, forward_periods)` — names *which hypothesis*.
Context = mutable dict of slicing conditions (`universe_id`,
`regime_id`, …) — names *which slice* of the data the hypothesis was
tested on. `expand_over` may only name context keys (or the built-in `"forward_periods"`), never the factor name.

Concretely: if `expand_over=["factor"]` were allowed, every factor
would land in its own size-1 family and pass its own step-up trivially
— FDR control across the screening universe collapses.

### Sample restriction vs hypothesis dimension

A context key (`universe_id`, `regime_id`, …) can play one of two roles
in a screening run:

- **Sample restriction** — you have *already* committed to a single slice
  (e.g. "this study runs on `tw50` only"). The context value is a
  pre-registered scope, not a tested dimension. Filter the input list
  upstream and call `bhy` without naming that key. FDR is controlled over
  the implied family.
- **Hypothesis dimension** — you want to ask "across these
  contexts, which factor / context combinations survive?" The context
  value is part of the hypothesis identity at the family level. Pass it
  via `expand_over=(<key>,)`; one independent step-up runs per distinct
  value tuple.

| User intent | API call | Family scope per step-up |
|---|---|---|
| "Run BHY on the `tw50` universe only" | `bhy([r for r in results if r.context.get("universe_id") == "tw50"], metrics=["ic"])` | `factor × forward_periods` |
| "Test the same factors on `tw50` *and* `tw100`, treating universe as a tested dimension" | `bhy(results, metrics=["ic"], expand_over=("universe_id",))` | `factor × forward_periods × universe_id`, one step-up per universe |

## See also

<div class="grid cards" markdown>

-   **Batch evaluation**

    ---

    Evaluate multiple candidate factors and multiple metrics in a single execution DAG.

    [api/evaluate →](evaluate.md)

-   **`partial_conjunction`**

    ---

    The "significant in `k` of `m` conditions" claim — use when you
    want a per-factor verdict across replications rather than a
    per-condition step-up.

    [api/partial-conjunction →](partial-conjunction.md)

-   **`bhy_hierarchical`**

    ---

    Two-stage FDR over group structure: gate at the group level, then run `bhy` within each rejected group.

    [api/bhy-hierarchical →](bhy-hierarchical.md)

-   **`multi_factor` overview**

    ---

    Module-level entry point listing all collection-level FDR functions.

    [api/multi-factor →](multi-factor.md)

</div>
