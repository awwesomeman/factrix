---
title: factrix.multi_factor.bhy
---

::: factrix.multi_factor.bhy

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Screening a candidate pool with false discovery rate (FDR) control__

    ---

    Run `evaluate` over `m` candidate signals on the same return panel,
    feed the resulting list of [`FactorProfile`][factrix.FactorProfile]
    to `bhy`, and read the surviving subset off the
    [`Survivors`][factrix.multi_factor.Survivors] container. Controls
    FDR ≤ `q` under arbitrary dependence — the regime that matches a
    correlated factor pool (e.g. 200 momentum variants on one panel).

-   __Splitting into independent families__

    ---

    Pass `expand_over=[<context key>]` to run one step-up per distinct
    bucket — for instance per `regime_id` or `universe_id` — under the
    Benjamini & Bogomolov (2014) selective-inference framing. Each
    bucket's `m`, threshold, and survivors stay self-contained.

-   __Swapping the inference path__

    ---

    Pass `estimator=NeweyWest()` (or another `Estimator`) to feed a
    specific heteroskedasticity-and-autocorrelation-consistent (HAC) variant's p-value into the step-up math rather than
    `primary_p`. The procedure dispatches via `Estimator.emits_for` to
    a `StatCode` on `profile.stats`.

-   __Auditing the family boundary__

    ---

    `Survivors.expand_over` / `Survivors.n_tests` / `Survivors.q`
    record the family declared and the `m` fed into each bucket's
    step-up, so the FDR claim is self-contained in the return object.

</div>

## Survivors attributes

See the docstring Examples block above for the canonical
multi-factor call. The returned `Survivors` container exposes:

| Attribute | Type | Meaning |
|---|---|---|
| `survivors.profiles` | `list[FactorProfile]` | input order, surviving subset |
| `survivors.adj_p` | `np.ndarray` | bucket-local Benjamini-Hochberg-Yekutieli (BHY)-adjusted p-value, index-aligned with `profiles` |
| `survivors.q` | `float` | the nominal target you passed |
| `survivors.expand_over` | `tuple[str, ...]` | `()` for a single family; `("regime_id",)` etc. otherwise |
| `survivors.n_tests` | `dict[tuple, int]` | `{(): N}` or `{bucket_key: m_per_bucket}` |

Jupyter rendering surfaces a three-column text / HTML table of
`identity | primary_p | adj_p`, plus an `expand_over_values` column
when buckets are declared.

The input list **is** the family. `bhy` runs one Benjamini–Yekutieli
step-up over all profiles by default, returning the surviving subset
wrapped in a `Survivors` container with rich Jupyter rendering
(three-column text / HTML table of `identity | primary_p | adj_p`,
plus an `expand_over_values` column when buckets are declared).
Each panel carries its factor under a distinct column name and
`evaluate(..., factor_col=name)` auto-stamps `factor_id` from that
name; this is the canonical multi-factor pattern. When you cannot
rename the column (e.g. an upstream loader fixes it), reach for
`dataclasses.replace(profile, factor_id=...)` as an escape hatch.

Set `expand_over=[<context key>]` to declare per-bucket independent
families (Benjamini & Bogomolov 2014 selective inference); for
example, `expand_over=["regime_id"]` runs one step-up per regime.

`bhy` implements the Benjamini–Yekutieli (2001) procedure with the
harmonic dependence correction $c(m) = \sum_{i=1}^{m} 1/i$ baked into
the threshold. **Pass your nominal `q` directly — no manual division.**
The procedure controls FDR ≤ q under arbitrary positive or negative
dependence at the cost of a $1/\ln m$ power loss relative to plain Benjamini-Hochberg (BH).
Plain Benjamini–Hochberg (1995) is **not** offered: typical factor-pool
dependence violates its Positive Regression Dependency on a Subset
(PRDS) assumption.

## Parameters

| Kwarg | Default | Meaning |
|-------|---------|---------|
| `expand_over` | `None` | Context keys whose distinct value tuples split the input into independent step-ups. Names must live in `FactorProfile.context` (never identity). |
| `estimator` | `None` (= `primary_p`) | An `Estimator` instance (e.g. `NeweyWest()`) selecting which inference method's p-value to feed into the step-up math. Dispatches via `Estimator.emits_for` to a `StatCode` key on `profile.stats` (see #170, #187). |
| `q` | `0.05` | Nominal false discovery rate target. The Benjamini–Yekutieli $c(m)$ correction is applied internally — pass the level you actually want; do not pre-divide. |

### Identity vs context (anti-shopping defense)

Identity = `(factor_id, forward_periods)` — names *which hypothesis*.
Context = mutable dict of slicing conditions (`universe_id`,
`regime_id`, …) — names *which slice* of the data the hypothesis was
tested on. `expand_over` may only name context keys, never identity.

Concretely: if `expand_over=["factor_id"]` were allowed, every factor
would land in its own size-1 family and pass its own step-up trivially
— FDR control across the screening universe collapses. Same logic for
`forward_periods`: per-horizon families would let a candidate "shop"
the horizon at which its noise happens to clear the threshold. Forcing
`expand_over` to live in `context` keeps the family axis orthogonal to
the hypothesis being tested. See [Development § Architecture § Family
functions](../development/architecture.md#_resolve_family-four-invariants)
for the full invariant list.

### Sample restriction vs hypothesis dimension

A context key (`universe_id`, `regime_id`, …) can play one of two roles
in a screening run, and the role you intend dictates whether to
**pre-filter** or to pass it through **`expand_over`**:

- **Sample restriction** — you have *already* committed to a single slice
  (e.g. "this study runs on `tw50` only"). The context value is a
  pre-registered scope, not a tested dimension. Filter the input list
  upstream and call `bhy` without naming that key. FDR is controlled over
  the implied family `factor_id × forward_periods`.
- **Hypothesis dimension** — you genuinely want to ask "across these
  contexts, which factor / context combinations survive?" The context
  value is part of the hypothesis identity at the family level. Pass it
  via `expand_over=[<key>]`; one independent step-up runs per distinct
  value tuple and FDR is controlled within each bucket
  (Benjamini & Bogomolov 2014).

| User intent | API call | Family scope per step-up |
|---|---|---|
| "Run BHY on the `tw50` universe only" | `bhy([p for p in profiles if p.context["universe_id"] == "tw50"])` | `factor_id × forward_periods` |
| "Test the same factors on `tw50` *and* `tw100`, treating universe as a tested dimension" | `bhy(profiles, expand_over=["universe_id"])` | `factor_id × forward_periods × universe_id`, one step-up per universe |
| "Run BHY only in `bull` regime months" | `bhy([p for p in profiles if p.context["regime_id"] == "bull"])` | `factor_id × forward_periods` |
| "Test across `bull` and `bear` regimes as separate hypotheses" | `bhy(profiles, expand_over=["regime_id"])` | one step-up per regime |
| "Test universe × regime as a joint hypothesis grid" | `bhy(profiles, expand_over=["universe_id", "regime_id"])` | one step-up per `(universe, regime)` cell |

`bhy` deliberately offers no implicit default for ambiguous contexts: if
the same `(factor_id, forward_periods)` appears more than once with no
`expand_over`, the duplicate-partition check raises with both fixes
(canonical `factor_id` rename and `expand_over=[<key>]`) called out in
the error. This forces an explicit commitment to the family boundary
before any step-up runs.

A context key can graduate from sample restriction to hypothesis
dimension when the study scope widens (extending a `tw50`-only run to
also cover `tw100` once `tw100` profiles exist). What the API refuses
to make easy is the reverse path: re-running with `expand_over=`
toggled to whichever shape produces more survivors. That is p-hacking
on the family boundary itself — the family must be declared before
inspecting the adjusted p-values it produces.

## Return type: `Survivors` (#171)

`bhy` returns a `Survivors` container, not a bare list. The container
is procedure-agnostic — `adj_p` carries the function's procedure-canonical
adjusted p-value (BHY here; future Holm / Bonferroni / Romano-Wolf
functions populate the same shape via their own `*_adjusted_p`).

`Survivors` carries only the kept rows. The construction rule **inside
`bhy`** is: compute `adj_p_all` over the full bucket-local input via
`bhy_adjusted_p`, then keep `{i : adj_p_all[i] <= q}` and slice both
`profiles` and `adj_p` to that index set. The survivor mask is derived
from the same adjusted p-values that downstream code reads — not a
separate rejection-mask path — so tie / boundary edge cases where two
parallel step-up implementations could disagree are eliminated by
construction.

`adj_p[i]` is computed within `profiles[i]`'s **own** `expand_over`
bucket — bucket-local `n` and `p_array`, never pooled across buckets.
This is the per-family adjustment that selective-inference theory
(Benjamini & Bogomolov 2014) requires; `n_tests[bucket_key]` records
the `m` fed into each step-up so the audit trail is self-contained.

Migration from the v0.10 list-of-profiles return:

```python
# before
survivors = bhy(profiles, q=0.05)
for s in survivors:
    print(s.factor_id)

# after
survivors = bhy(profiles, q=0.05)
for s in survivors.profiles:
    print(s.factor_id)
# or just `repr(survivors)` in Jupyter for the table view
```

::: factrix.multi_factor.Survivors
    options:
      show_root_toc_entry: false
      heading_level: 3

## Behaviour change (#161)

| Before | After |
|--------|-------|
| `bhy(profiles, threshold=0.05)` | `bhy(profiles, q=0.05)` (`threshold=` removed in v0.12.0; raises `TypeError`) |
| `bhy(profiles, gate=StatCode.X)` | `bhy(profiles, p_stat=StatCode.X)` (`gate=` removed in v0.11.0) |
| auto-partition by dispatch cell × horizon | caller declares the family; mixed `forward_periods` without `expand_over` emits `RuntimeWarning`. **Fix:** split the call per horizon, or pass `expand_over=[<context key>]` if profiles legitimately co-exist as one family across horizons. |
| same `factor_id` across cells silently auto-split | raises [`UserInputError`][factrix.UserInputError] (duplicate identity). **Fix (canonical):** name each panel's factor column distinctly and pass `evaluate(..., factor_col=name)`. **Fix (escape hatch):** post-hoc stamp via `dataclasses.replace(profile, factor_id=...)`. **Or:** use `expand_over` if profiles really do share identity but belong in separate test buckets. |

## Design rationale

For why BHY (rather than Bayesian or reality-check / SPA bootstraps)
see [Reference § Statistical methods § Multiple-testing](../reference/statistical-methods.md#2-multiple-testing-under-dependence)
and [Development § Design notes § BHY](../development/design-notes.md#5-bhy-rather-than-bayesian-multiple-testing).
For the architectural place of `_resolve_family` and the closed-form
vs resampling-based function classification see [Development §
Architecture § Family functions](../development/architecture.md#family-functions-and-the-resolution-layer).

## See also

<div class="grid cards" markdown>

-   __Batch screening guide__

    ---

    End-to-end recipe: loop `evaluate` over candidates, preserve
    identity / context, and feed the list into `bhy`.

    [guides/batch-screening →](../guides/batch-screening.md)

-   __`partial_conjunction`__

    ---

    The "significant in `k` of `m` conditions" claim — use when you
    want a per-factor verdict across replications rather than a
    per-condition step-up.

    [api/partial-conjunction →](partial-conjunction.md)

-   __`bhy_hierarchical`__

    ---

    Two-stage FDR over group structure (factor families, regime
    blocks): gate at the group level, then run `bhy` within each
    rejected group.

    [api/bhy-hierarchical →](bhy-hierarchical.md)

-   __`multi_factor` overview__

    ---

    Module-level entry point listing all collection-level FDR
    functions and when to reach for each.

    [api/multi-factor →](multi-factor.md)

</div>

