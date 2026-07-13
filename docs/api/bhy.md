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

    Pass `expand_over=(<params key>,)` to run one step-up per distinct
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
| `entries` | `list[EvaluationResult]` | Every tested factor, input order — survivors and eliminated alike. |
| `adj_p_all` | `np.ndarray` | Bucket-local Benjamini-Hochberg-Yekutieli (BHY)-adjusted p-value, index-aligned with `entries`; `NaN` for an entry dropped before the family formed. |
| `survivors` | `list[EvaluationResult]` | Surviving subset (derived: `adj_p_all <= q`). |
| `adj_p` | `np.ndarray` | Adjusted p-value for the survivors, aligned with `survivors` (derived). |
| `q` | `float` | The nominal target FDR you passed. |
| `expand_over` | `tuple[str, ...]` | `()` for a single family; `("regime_id",)` etc. otherwise. |
| `n_tests` | `Mapping[tuple, int]` | `{(): N}` or `{bucket_key: m_per_bucket}`. |

Call `result.to_frame()` for a `factor | adj_p | survived` DataFrame over
**all** tested factors — so a screen of N factors passing 2 still shows how
far the eliminated N-2 sat from the threshold, rather than discarding them.

Jupyter rendering surfaces a three-column text / HTML table of
`factor | adj_p` for the survivors, plus an `expand_over_values` column
when buckets are declared.

::: factrix.multi_factor.BhyResult
    options:
      show_root_toc_entry: false
      heading_level: 3

## Parameters

| Kwarg | Default | Meaning |
|-------|---------|---------|
| `metrics` | (required) | `list[str]` of metric labels to run the FDR screen for. |
| `expand_over` | `()` | Params keys whose distinct value tuples split the input into independent step-ups. Names must live in `EvaluationResult.params` (except for the built-in `"forward_periods"`). Naming a `metadata` key is rejected — bookkeeping does not define a family. |
| `q` | `0.05` | Nominal false discovery rate target. The Benjamini–Yekutieli $c(m)$ correction is applied internally — pass the level you actually want; do not pre-divide. |

### Identity vs partition (anti-shopping defense)

Two separate concerns, on two separate knobs:

- **Identity** — `(factor, forward_periods, *params)`. It names *which
  hypothesis* this is. Every `EvaluationResult.params` entry joins it
  automatically, so a swept knob (`base_tf`, `universe_id`, …) never has to be
  encoded into the factor name to stay unique.
- **Family partition** — `expand_over` alone. It is a purely statistical
  declaration about which hypotheses compete with each other. It may name
  `params` keys or the built-in `"forward_periods"`, never the factor name.

`EvaluationResult.metadata` is bookkeeping (`run_id`, data vintage). It joins
neither. Two results differing only in `metadata` are the *same* hypothesis and
raise as a duplicate — that check exists to catch a hypothesis submitted twice,
and a bookkeeping label must not defeat it.

Concretely: if `expand_over=["factor"]` were allowed, every factor would land in
its own size-1 family and pass its own step-up trivially — FDR control across the
screening universe collapses.

With `expand_over=()`, all hypotheses enter one step-up. Pooling makes the family
larger and the per-rank threshold *stricter*, so it is the correct — and
conservative — choice whenever the research process may select the winner across
the swept knobs. Use `expand_over=(…)` only when the buckets were predeclared and
will be selected and reported separately; separate buckets provide no global
control over later shopping across them.

### The three roles a knob can play

| User intent | Where the knob lives | API call | Family scope per step-up |
|---|---|---|---|
| "This study runs on `tw50` only" — a pre-registered scope, not a tested dimension | filter upstream; the knob need not be stamped at all | `bhy([r for r in results if …], metrics=["ic"])` | `factor × forward_periods` |
| "Sweep `base_tf` ∈ {1h, 4h, 1d} and take the best" — a tested dimension, winner selected across it | `params` | `bhy(results, metrics=["ic"])` | `factor × forward_periods × base_tf`, **one** step-up over all of them |
| "Report predeclared `tw50` and `tw100` screens separately, with no cross-universe winner selection" | `params` + `expand_over` | `bhy(results, metrics=["ic"], expand_over=("universe_id",))` | one step-up per universe |
| "Record which pipeline run produced this" — never affects inference | `metadata` | — | not applicable |

The middle row is the one that used to require encoding the knob into the factor
name. It no longer does: stamping `base_tf` on `params` makes each hypothesis
uniquely identified while leaving them all in a single family.

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
