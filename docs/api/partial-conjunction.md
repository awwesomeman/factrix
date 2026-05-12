# partial_conjunction

Contract-bearing screening for the "factor X is significant in $k$ of
$m$ conditions" claim. Replaces the notebook idiom
`set(survivors_a) & set(survivors_b)`, which does **not** preserve FDR
([Benjamini & Bogomolov 2014](https://academic.oup.com/jrsssb/article/76/1/297/7075880)),
with the partial conjunction test of
[Benjamini & Heller (2008)](https://onlinelibrary.wiley.com/doi/10.1111/j.1541-0420.2008.00984.x).

```python
import factrix as fx

# "Momentum is significant in BOTH large-cap AND small-cap universes"
profiles = [
    fx.evaluate(panel_large, cfg, factor_col="mom"),
    fx.evaluate(panel_small, cfg, factor_col="mom"),
    # ... + value, quality, etc. one profile per (factor, universe) cell
]
survivors = fx.multi_factor.partial_conjunction(
    profiles,
    min_pass=2,
    n_conditions=2,
    expand_over=["universe_id"],
    q=0.05,
)
```

## Versus `bhy(expand_over=...)` — same data, different question

Both verbs accept `expand_over=`, but the **survivor unit** and the
**question answered** differ. This is the single most common source of
confusion; pick the row that matches your claim.

| Verb | Survivor unit | Question | Example claim |
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
| Sensitivity to estimator / sample choice | `robustness` (#178) | Conditions there are *methods*, not data slices. |
| Cross-slice metric difference (descriptive) | [`by_slice`](by-slice.md) | Returns per-slice metric values; no inference. |
| Cross-slice metric difference (inferential, slice-pairs) | [`slice_pairwise_test` / `slice_joint_test`](slice-test.md) | Tests whether the slices' metric series differ, not whether the factor is jointly significant. |

## Strict vs lenient mode

`n_conditions` is the contract knob.

| Mode | When | Behavior |
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

The PC $p$-values are then fed to a standard BHY step-up across
identities, controlling group-level FDR ≤ `q`. The harmonic dependence
correction $c(m) = \sum 1/i$ is applied because PC $p$-values across
identities are not generally PRDS — sharing underlying panels makes the
joint distribution unknown, so the conservative choice is the default.

BH2008 also presents a Simes-style PC combiner, which is less
conservative under PRDS. factrix ships only the Bonferroni-style — it
is uniformly valid without dependence assumptions; a Simes path may
be added later if a use case demands it.

## Survivors output

`partial_conjunction` returns the same [`Survivors`](multi-factor.md)
container as `bhy`, populated with PC-specific metadata:

| Field | Meaning |
|---|---|
| `profiles` | One representative profile per surviving identity (the first profile of that identity in input order) |
| `adj_p` | BHY-adjusted PC $p$-value; survivor iff `adj_p <= q` |
| `pc_p` | Raw PC $p$-value (pre-BHY) |
| `min_pass` | The $k$ you passed |
| `n_total` | Keyed by identity tuple `(factor_id, forward_periods)` → actual $m$ used |
| `n_passed_uncorr` | Per-identity count of raw $p < q$. Descriptive — flags borderline (`n_passed_uncorr == min_pass`) and data-gap cases at a glance. **Cutoff is your `q`**, so the count moves with `q` — using it to override `adj_p` survivor selection is the anti-shopping failure mode this verb exists to prevent. |

## Validation summary

| Trigger | Outcome |
|---|---|
| `min_pass < 2` | `UserInputError`. `min_pass == 1` additionally points at `bhy(expand_over=...)`. |
| `expand_over` empty / `None` | `UserInputError` — the verb is undefined without a condition axis. |
| `expand_over` names an identity field (`factor_id` / `forward_periods`) | `UserInputError` (#160 anti-shopping defense — same as `bhy`). |
| `n_conditions < min_pass` | `UserInputError` (unsatisfiable). |
| Strict mode: identity's condition count $\neq$ `n_conditions` | `UserInputError` — surfaces missing-universe / missing-horizon data gaps. |
| Identity with condition count $<$ `min_pass` (lenient) | `UserInputError`. |
| Duplicate `(identity, expand_over_values)` partition key | `UserInputError` (family-resolution invariant). |

## References

- **[BH2008]** Benjamini, Y. & Heller, R. (2008). Screening for partial
  conjunction hypotheses. *Biometrics*, 64(4), 1215–1222.
- **[BB2014]** Benjamini, Y. & Bogomolov, M. (2014). Selective inference
  on multiple families of hypotheses. *JRSS-B*, 76(1).
- **[HY2014]** Heller, R. & Yekutieli, D. (2014). Replicability analysis
  for genome-wide association studies. *AOAS*, 8(1).

::: factrix.multi_factor.partial_conjunction
