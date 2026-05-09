# Identity vs context

Every `FactorProfile` carries two structured fields that describe
"what hypothesis was tested" separately from "under what sample
conditions":

| Field | Type | Meaning |
|---|---|---|
| `identity` | `tuple[str, int]` | `(factor_id, forward_periods)` — the hypothesis tuple. |
| `context` | `Mapping[str, Any]` | Sample restriction / conditioning dimensions (`universe_id`, `regime_id`, future axes). |

Convenience accessors on the profile:

- `profile.factor_id` → `identity[0]`
- `profile.forward_periods` → `identity[1]`

## Why split them?

The split is the v1 anti-shopping defense for multi-horizon and
multi-universe factor research. Putting the wrong axis in `identity`
silently changes which hypotheses are considered "the same family" by
multiple-testing correction — and lets researchers walk the family
boundary until something looks significant.

| Spec-search variant | API-level guard | Status |
|---|---|---|
| Estimator shopping (swap SE method until significant) | Cell → procedure 1:1 (registry SSOT) | shipped |
| Stat shopping (swap `p_stat` per factor) | Study-level `p_stat=` only; per-factor not allowed | shipped |
| Universe shopping (swap large-cap → small-cap) | `context["universe_id"]` is a sample restriction, not a hypothesis dimension; promotion to family member requires explicit `expand_over=` | partial — context split shipped; `expand_over=` lands in [#161](https://github.com/awwesomeman/factrix/issues/161) |
| Family-scope shopping (multiplicative → per-slice) | No implicit default — `expand_over=` must be explicit | [#161](https://github.com/awwesomeman/factrix/issues/161) |
| **Horizon shopping** (run every `forward_periods` ∈ {1d, 5d, 1m, 3m, 6m, 12m}, report the smallest p) | `forward_periods` is part of `identity`; `bhy(profiles)` over a horizon sweep auto-forms the full family | shipped (BHY family already partitions on `forward_periods`) |

The path of least resistance — `[evaluate(panel, cfg) for cfg in
horizon_grid]` followed by `bhy(profiles)` — is also the statistically
correct one. Shopping has to actively shrink the profile list, which is
visible in code review.

## How identity is populated

`evaluate()` stamps `identity` from two sources:

```python
profile = evaluate(panel, cfg, factor_col="momentum_12_1")
profile.identity         # ("momentum_12_1", cfg.forward_periods)
profile.factor_id        # "momentum_12_1"
profile.forward_periods  # cfg.forward_periods
```

- `factor_id` ← the `factor_col` argument (the column name on `panel`)
- `forward_periods` ← `cfg.forward_periods`

Procedures themselves stay schema-agnostic; the stamp happens once at
the dispatch boundary inside `_evaluate`. `factor_id` and
`forward_periods` are read-only properties that proxy `identity[0]` /
`identity[1]` — `dataclasses.replace(p, identity=(new_id, fwd))` is
the way to override them; `replace(p, factor_id=...)` does not work.

## How context is populated

`context` ships empty by default. Higher-level verbs that operate on a
filtered or sliced panel populate it via `dataclasses.replace`:

```python
import dataclasses

p = evaluate(panel_large_cap, cfg, factor_col="momentum_12_1")
p = dataclasses.replace(p, context={"universe_id": "us_large_cap"})
```

The `by_slice` / `by_regime` consumers and the upcoming `run_metrics`
verb populate `context` automatically — manual `replace` is the
escape hatch for callers who run their own slicing.

## Querying context as a sample restriction

Treating universe / regime as sample restriction (the common case) is a
plain comprehension before the family verb:

```python
import factrix as fl

profiles = [
    evaluate(panel, cfg, factor_col=name)
    for name in factor_cols
]
large_cap = [p for p in profiles if p.context.get("universe_id") == "us_large_cap"]
fl.multi_factor.bhy(large_cap, threshold=0.05)
```

When the universe / regime axis IS a hypothesis dimension (e.g., "is
this factor significant in *some* universe?"), promote it via
`expand_over=` (see [`multi_factor.bhy`](multi-factor.md)). Mixing the
two paths is the single most common screening-loop bug; the split
makes the choice explicit at the call site.

## Reading the rendered profile

`repr(profile)` lists identity, mode, primary_p, sample sizes, and
omits `context` / `warnings` when empty:

```
FactorProfile(factor_id='momentum_12_1', forward_periods=5, mode=panel,
primary_p=0.0312, n_obs=240, n_assets=500)
```

In Jupyter, `_repr_html_` renders the same fields as a table and
unfolds non-empty `context` entries as `context.<key>` rows so
universe / regime restrictions are visible without calling `diagnose()`.
