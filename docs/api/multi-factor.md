# multi_factor

Collection-level FDR control across factor profiles. Use after
`evaluate` has produced one profile per candidate factor — `bhy`
adjusts the per-factor $p$-values for multiple testing under arbitrary
dependence (factor pools are dependent by construction: 200 momentum
variants on the same return panel correlate, and a Bonferroni step
that assumes independence over-corrects).

## Call shape

```python
profiles = [
    fx.evaluate(panel_per_lookback[lookback], cfg, factor_col=f"momentum_{lookback}")
    for lookback, cfg in candidates
]
survivors = fx.multi_factor.bhy(profiles, q=0.05)

survivors.profiles   # list[FactorProfile] in input order
survivors.adj_q      # numpy array — bucket-local BHY-adjusted p-value, aligned
survivors.q          # 0.05 — the nominal target you passed
survivors.expand_over     # () for a single family; ("regime_id",) etc. otherwise
survivors.n_total    # {(): N} or {bucket_key: m_per_bucket}
```

The input list **is** the family. `bhy` runs one Benjamini–Yekutieli
step-up over all profiles by default, returning the surviving subset
wrapped in a `Survivors` container with rich Jupyter rendering
(three-column text / HTML table of `identity | primary_p | adj_q`,
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
dependence at the cost of a $1/\ln m$ power loss relative to plain BH.
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
verbs](../development/architecture.md#_resolve_family-four-invariants)
for the full invariant list.

## Return type: `Survivors` (#171)

`bhy` returns a `Survivors` container, not a bare list. The container
is procedure-agnostic — `adj_q` carries the verb's procedure-canonical
adjusted p-value (BHY here; future Holm / Bonferroni / Romano-Wolf
verbs populate the same shape via their own `*_adjusted_p`).

`Survivors` carries only the kept rows. The construction rule **inside
`bhy`** is: compute `adj_q_all` over the full bucket-local input via
`bhy_adjusted_p`, then keep `{i : adj_q_all[i] <= q}` and slice both
`profiles` and `adj_q` to that index set. The survivor mask is derived
from the same adjusted p-values that downstream code reads — not a
separate rejection-mask path — so tie / boundary edge cases where two
parallel step-up implementations could disagree are eliminated by
construction.

`adj_q[i]` is computed within `profiles[i]`'s **own** `expand_over`
bucket — bucket-local `n` and `p_array`, never pooled across buckets.
This is the per-family adjustment that selective-inference theory
(Benjamini & Bogomolov 2014) requires; `n_total[bucket_key]` records
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

## Behaviour change (#161)

| Before | After |
|--------|-------|
| `bhy(profiles, threshold=0.05)` | `bhy(profiles, q=0.05)` (`threshold=` still accepted with `DeprecationWarning`) |
| `bhy(profiles, gate=StatCode.X)` | `bhy(profiles, p_stat=StatCode.X)` (`gate=` still accepted with `DeprecationWarning`) |
| auto-partition by dispatch cell × horizon | caller declares the family; mixed `forward_periods` without `expand_over` emits `RuntimeWarning`. **Fix:** split the call per horizon, or pass `expand_over=[<context key>]` if profiles legitimately co-exist as one family across horizons. |
| same `factor_id` across cells silently auto-split | raises `UserInputError` (duplicate identity). **Fix (canonical):** name each panel's factor column distinctly and pass `evaluate(..., factor_col=name)`. **Fix (escape hatch):** post-hoc stamp via `dataclasses.replace(profile, factor_id=...)`. **Or:** use `expand_over` if profiles really do share identity but belong in separate test buckets. |

## Design rationale

For why BHY (rather than Bayesian or reality-check / SPA bootstraps)
see [Reference § Statistical methods § Multiple-testing](../reference/statistical-methods.md#2-multiple-testing-under-dependence)
and [Development § Design notes § BHY](../development/design-notes.md#5-bhy-rather-than-bayesian-multiple-testing).
For the architectural place of `_resolve_family` and the closed-form
vs resampling-based verb classification see [Development §
Architecture § Family verbs](../development/architecture.md#family-verbs-and-the-resolution-layer).

::: factrix.multi_factor.bhy
