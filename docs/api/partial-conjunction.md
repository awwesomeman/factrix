---
title: factrix.multi_factor.partial_conjunction
---

::: factrix.multi_factor.partial_conjunction

Contract-bearing screening for the "factor X is significant in $k$ of
$m$ conditions" claim. Replaces the notebook idiom
`set(survivors_a) & set(survivors_b)`, which does **not** preserve false discovery rate (FDR)
([Benjamini & Bogomolov 2014](https://academic.oup.com/jrsssb/article/76/1/297/7075880)),
with the partial conjunction test of
[Benjamini & Heller (2008)](https://onlinelibrary.wiley.com/doi/10.1111/j.1541-0420.2008.00984.x).

```python
import dataclasses

import factrix as fx
from factrix.metrics import ic

# "Momentum is significant in BOTH large-cap AND small-cap universes"
# evaluate() returns dict[str, EvaluationResult]; pull the single result
# and stamp the universe label onto it with dataclasses.replace so
# expand_over can split the family on context["universe_id"].
def profile(panel, factor_col, **context):
    res = fx.evaluate(panel, metrics={"ic": ic()}, factor_cols=[factor_col])[factor_col]
    return dataclasses.replace(res, context=context)

profiles = [
    profile(panel_large, "mom", universe_id="large_cap"),
    profile(panel_small, "mom", universe_id="small_cap"),
    # ... + value, quality, etc. one profile per (factor, universe) cell
]
survivors = fx.multi_factor.partial_conjunction(
    profiles,
    metrics=["ic"],
    min_pass=2,
    n_conditions=2,
    expand_over=("universe_id",),
    q=0.05,
)
```

## Versus `bhy(expand_over=...)` — same data, different question

Both functions accept `expand_over=`, but the **survivor unit** and the
**question answered** differ. This is the single most common source of
confusion; pick the row that matches your claim.

| Function | Survivor unit | Question | Example claim |
|---|---|---|---|
| `bhy(profiles, expand_over=["universe_id"])` | `(factor, universe)` pair | "Where is this factor significant?" | "Momentum is significant in `large_cap`; value is significant in `small_cap`" |
| `partial_conjunction(profiles, min_pass=2, expand_over=["universe_id"])` | `factor` identity | "Which factors are significant across all conditions?" | "Momentum is significant across both universes" |

In other words: `bhy` treats each universe as **its own hypothesis** and
expands the family; `partial_conjunction` treats each universe as a
**condition the factor must pass jointly** and aggregates back to one
hypothesis per factor.

## When *not* to reach for `partial_conjunction`

| Real intent | Reach for | Why |
|---|---|---|
| "At least *any* condition is significant" | `bhy(profiles, expand_over=[...])` | `min_pass=1` is union semantics — FDR inflates to ~2q. `partial_conjunction` raises rather than implement this. |
| Rank candidates (no FDR control) | [`compare`](compare.md) | `compare` is a view, not a filter. |
| Sensitivity to estimator / sample choice | a dedicated robustness sweep | Conditions there are *methods*, not data slices. |
| Cross-slice metric difference (descriptive) | [`by_slice`](by-slice.md) | Returns per-slice metric values; no inference. |
| Cross-slice metric difference (inferential, slice-pairs) | [`slice_pairwise_test` / `slice_joint_test`](slice-test.md) | Tests whether the slices' metric series differ, not whether the factor is jointly significant. |

## Strict vs lenient mode

`n_conditions` is the contract knob.

| DataStructure | When | Behavior |
|---|---|---|
| **Strict** (`n_conditions=int`) | Paper-grade; you know the design (e.g. exactly 2 universes, exactly 4 horizons) | Identity with any condition count other than `n_conditions` raises. Data gaps surface fail-loud. |
| **Lenient** (`n_conditions=None`) | EDA / prototyping; condition count varies by identity | `m` inferred per identity from the data; only requires `m >= min_pass`. |

```python
# Strict: 2 universes required for every factor; missing one raises.
fx.multi_factor.partial_conjunction(
    profiles, min_pass=2, n_conditions=2, expand_over=["universe_id"]
)

# Lenient: "at least 3 of however many horizons each factor has".
fx.multi_factor.partial_conjunction(
    profiles, min_pass=3, expand_over=["fwd_period"]
)
```

## How the math works

Per identity, the $m$ per-condition $p$-values are reduced to a single
**PC $p$-value** (Bonferroni-style, BH2008):

$$
p_{\text{PC}}^{(k/m)} = \min\bigl(1,\; (m - k + 1) \cdot p_{(k)}\bigr)
$$

where $p_{(k)}$ is the $k$-th smallest of the $m$ $p$-values and
$k = \texttt{min\_pass}$. Two corner cases worth knowing:

- $k = m$ (full conjunction) → $p_{\text{PC}} = \max(p)$. Reject only
  when even the worst condition is significant.
- $k = 1$ (union) → $p_{\text{PC}} = m \cdot \min(p)$. Bonferroni-corrected
  minimum. **Forbidden here** — the surface raises with a pointer to
  `bhy(expand_over=...)`, where the family-level FDR inflation is
  explicit rather than hidden in a "robust across" claim.

The PC $p$-values are then fed to a standard Benjamini-Hochberg-Yekutieli (BHY) step-up across
identities, controlling group-level FDR ≤ `q`. The harmonic dependence
correction $c(m) = \sum 1/i$ is applied because PC $p$-values across
identities are not generally positive regression dependence on a subset (PRDS) — sharing underlying panels makes the
joint distribution unknown, so the conservative choice is the default.

BH2008 also presents a Simes-style PC combiner, which is less
conservative under PRDS. factrix ships only the Bonferroni-style — it
is uniformly valid without dependence assumptions; a Simes path may
be added later if a use case demands it.

## Result output

`partial_conjunction` returns a
[`PartialConjunctionResult`][factrix.multi_factor.PartialConjunctionResult]
per metric — the same `_FdrResultBase` shape as `bhy`'s
[`BhyResult`][factrix.multi_factor.BhyResult] (`entries` / `survivors` /
`adj_p` / `q` / `n_tests`), plus PC-specific fields:

| Field | Meaning |
|---|---|
| `entries` | One representative profile per tested identity (first profile of that identity, input order) — every identity, not just survivors |
| `adj_p_all` | BHY-adjusted PC $p$-value, aligned with `entries`; identity survives iff `adj_p_all <= q` |
| `pc_p_all` | Raw PC $p$-value (pre-BHY), aligned with `entries` |
| `survivors` / `adj_p` | Surviving subset and its adjusted p-value (derived from `adj_p_all <= q`) |
| `min_pass` | The $k$ you passed |
| `n_tests` | Keyed by the single-element identity tuple `(factor,)` → condition count $m$ for that identity |
| `n_passed_uncorr_all` | Per-identity count of raw $p < q$, aligned with `entries`. Descriptive — flags borderline (`n_passed_uncorr_all == min_pass`) and data-gap cases at a glance. **Cutoff is your `q`**, so the count moves with `q` — using it to override `adj_p` survivor selection is the anti-shopping failure mode this function exists to prevent. |

`to_frame()` gives a `factor | adj_p | survived` DataFrame over every tested
identity, eliminated ones included.

::: factrix.multi_factor.PartialConjunctionResult
    options:
      show_root_toc_entry: false
      heading_level: 3

## Validation summary

| Trigger | Outcome |
|---|---|
| `min_pass < 2` | [`UserInputError`][factrix.UserInputError]. `min_pass == 1` additionally points at `bhy(expand_over=...)`. |
| `expand_over` empty / `None` | [`UserInputError`][factrix.UserInputError] — the function is undefined without a condition axis. |
| `expand_over` names an identity field (`factor_id` / `forward_periods`) | [`UserInputError`][factrix.UserInputError] (anti-shopping defense — same as `bhy`). |
| `n_conditions < min_pass` | [`UserInputError`][factrix.UserInputError] (unsatisfiable). |
| Strict mode: identity's condition count $\neq$ `n_conditions` | [`UserInputError`][factrix.UserInputError] — surfaces missing-universe / missing-horizon data gaps. |
| Identity with condition count $<$ `min_pass` (lenient) | [`UserInputError`][factrix.UserInputError]. |
| Duplicate `(identity, expand_over_values)` partition key | [`UserInputError`][factrix.UserInputError] (family-resolution invariant). |

## References

- **[BH2008]** Benjamini, Y. & Heller, R. (2008). Screening for partial
  conjunction hypotheses. *Biometrics*, 64(4), 1215–1222.
- **[BB2014]** Benjamini, Y. & Bogomolov, M. (2014). Selective inference
  on multiple families of hypotheses. *JRSS-B*, 76(1).
- **[HY2014]** Heller, R. & Yekutieli, D. (2014). Replicability analysis
  for genome-wide association studies. *AOAS*, 8(1).
