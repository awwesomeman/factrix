# bhy_hierarchical

Two-stage FDR for factor sets with natural group structure (factor
families, regions, sectors). Outer Benjamini-Yekutieli on
[Simes (1986)](https://academic.oup.com/biomet/article/73/3/751/277681)
group representatives + inner BHY within each passing group, per
[Yekutieli (2008)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2007.ap06035).

```python
import factrix as fx

# "Which factor families have signal, and within those, which factors?"
profiles = [
    fx.evaluate(panel_mom_1m, cfg, factor_col="mom_1m",
                context={"family": "momentum"}),
    fx.evaluate(panel_mom_12m, cfg, factor_col="mom_12m",
                context={"family": "momentum"}),
    fx.evaluate(panel_pb, cfg, factor_col="pb",
                context={"family": "value"}),
    fx.evaluate(panel_pe, cfg, factor_col="pe",
                context={"family": "value"}),
    # ... + quality, low-vol, etc.
]
survivors = fx.multi_factor.bhy_hierarchical(profiles, group="family", q=0.05)
```

## Which function fits this question?

Same input shape (one profile per (factor, condition)), three different
claims:

| Claim | Survivor unit | Function |
|---|---|---|
| "Factor X significant in *each* condition / universe" | `(factor, condition)` pair | [`bhy(expand_over=)`](multi-factor.md) |
| "Factor X significant in $\ge k$ of $m$ conditions" | factor identity | [`partial_conjunction`](partial-conjunction.md) |
| "Which families have signal, and within those, which factors?" | factor identity (group-then-within) | `bhy_hierarchical` |

`bhy_hierarchical` is the only one of the three that keeps the
**family-level answer** first-class — readers learn both "5 of 8
families showed signal" and "within those, factors A / B / C survived"
from a single Survivors container.

## How the math works

Per group $g$ with $m_g$ member p-values:

1. Compute the group representative

    $$
    p_{\text{Simes},g} = \min_{k=1,\ldots,m_g} \frac{m_g}{k} \cdot p_{(k),g}
    $$

   where $p_{(k),g}$ is the $k$-th smallest p-value in group $g$. Simes
   dominates the Bonferroni representative $m_g \cdot \min(p)$ and is
   the Yekutieli 2008 recommended choice.

2. Outer BHY across the $G$ group representatives gives
   $p_{\text{outer},g}^{\text{adj}}$.

3. Inner BHY within each group gives $p_{\text{inner},i}^{\text{adj}}$
   for member $i$ of group $g(i)$.

4. The cell-level adjusted p is the max-of-layers fold

    $$
    p_i^{\text{adj}} = \max\bigl(p_{\text{outer},g(i)}^{\text{adj}},\; p_{\text{inner},i}^{\text{adj}}\bigr)
    $$

   This preserves the universal Survivors duality
   `survivor[i] iff adj_p[i] <= q` while encoding the two-layer logic:
   a cell can fail because its group failed outer, because the cell
   itself failed inner, or both.

## Survivors output

| Field | Meaning |
|---|---|
| `profiles` | Surviving profiles in input order |
| `adj_p` | Max-of-layers $\text{adj}_p$; survivor iff `adj_p <= q` |
| `q` | The `q` you passed (single target, both layers) |
| `expand_over` | `(group,)` — single-element tuple |
| `n_tests` | Mapping `(group_value,) -> m_group` for **every** input group (covers dead families too, so "N of M families survived" claims are computable directly). Counter to `partial_conjunction`, which keeps surviving identities only. |

Per-survivor group label: `profile.context[group]`.

## When *not* to reach for `bhy_hierarchical`

| Real intent | Reach for | Why |
|---|---|---|
| No natural group structure | [`bhy`](multi-factor.md) | The grouping is real or it isn't; faking a group axis trivializes the procedure. |
| "Factor X passes in *every* condition" | [`partial_conjunction`](partial-conjunction.md) with `min_pass == m` | Hierarchical is "group-then-within", not "joint across conditions". |
| Flat BHY split by family for display only | [`bhy(expand_over=["family"])`](multi-factor.md) | Independent step-ups per bucket, no group-level inference. Use when you do not need a "this family has signal" answer. |
| Mixed-sign factors in one bucket | Split the bucket / pre-orthogonalize | Within-group Simes assumes PRDS; structurally opposite factors (e.g. momentum + reversal in one group) can violate it. |

## Validation summary

| Trigger | Outcome |
|---|---|
| `group` shadows an identity field (`factor_id` / `forward_periods`) | `UserInputError`. |
| `group` key missing from a profile's `context` | `UserInputError`. |
| Only one distinct group value across input | `UserInputError` — points at [`bhy`](multi-factor.md). |
| Every profile is its own group at $n \ge 3$ (group axis near-unique) | `UserInputError` — pick a coarser categorical. |
| Duplicate `(identity, group_value)` partition key | `UserInputError`. |
| More than half of input groups contain a single profile | `RuntimeWarning` — inner BHY on $n=1$ is a raw cutoff. |

## Caveats

- **Simes outer representative**: not exposed as a kwarg. Dominates
  Bonferroni-min under PRDS; Edgington-style mean-p has no valid null
  distribution and is rejected.
- **PRDS within group**: Simes is valid under positive regression
  dependence — typical for factors within one family that share style
  exposure. If a group mixes structurally opposite factors (e.g.
  momentum + reversal in one bucket), the within-group PRDS assumption
  can fail; split the group or pre-orthogonalize.
- **Pre-filtered input**: `bhy_hierarchical` assumes the input *is* the
  candidate family. If profiles came from upstream pre-filtering
  (e.g. top-50 of 500 candidates), the FDR claim does not cover the
  full screening pipeline — track $K$ per the experiment-log discipline.
- **Composed FDR is approximate at exact $q$**: Yekutieli 2008 bounds
  group-level FDR $\le q$ and within-group FDR $\le q$ conditional on
  group passing; the *composed* per-hypothesis FDR under PRDS is bounded
  but not exactly $q$. Researcher claims should be "FDR-controlled at
  $q$ in each layer", not "joint FDR $= q$".

## References

- **[S1986]** Simes, R. J. (1986). An improved Bonferroni procedure
  for multiple tests of significance. *Biometrika*, 73(3), 751–754.
- **[Y2008]** Yekutieli, D. (2008). Hierarchical false discovery
  rate-controlling methodology. *JASA*, 103(481), 309–316.
- **[NBER34050]** NBER WP 34050 (2025). Hierarchical Multiple Testing
  in Empirical Asset Pricing.

::: factrix.multi_factor.bhy_hierarchical
