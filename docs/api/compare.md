---
title: factrix.compare
---

::: factrix.compare

Leaderboard renderer that stacks N artifacts side by side as a
[polars `DataFrame`](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html).
Pure projection — no metric is recomputed; `Survivors.adj_p` is read
straight through so Benjamini-Hochberg-Yekutieli (BHY) survivor tables keep their adjusted p-values
without manual re-attach.

```python
import factrix as fx

profiles = [fx.evaluate(panel, cfg, factor_col=c) for c in candidates]
fx.compare(profiles, sort_by="primary_p")
```

## When to reach for `compare`

| Use case | Function | Notes |
|---|---|---|
| Rank N `evaluate` results | `compare(list[FactorProfile])` | Identity + context + `primary_stat` / `primary_stat_name` / `primary_p` |
| Rank N `run_metrics` results | `compare(list[MetricsBundle])` | Identity + context + one column per standalone metric (`MetricOutput.value`) |
| Rank BHY survivors | `compare(Survivors)` | Profile schema plus `adj_p` (read from `Survivors.adj_p`) |
| Re-run inference under perturbations | [`robustness`](../api/index.md) (#178) | `compare` is a pure view; `robustness` recomputes |
| Test factor across slices | [`slice_pairwise_test` / `slice_joint_test`](slice-test.md) | Re-runs inference per slice; `compare` does not |

If you need fresh statistics, you want a *re-compute* function. `compare`
is strictly read-through.

## Input dispatch

Single entrypoint, input-type dispatch — no `compare_profiles` /
`compare_bundles` split, so the call site does not branch on artifact
shape.

```python
compare(
    artifacts: list[FactorProfile] | list[MetricsBundle] | Survivors,
    *,
    sort_by: str | None = None,
) -> pl.DataFrame
```

### `list[FactorProfile]`

```
┌───────────────┬─────────────────┬─────────────┬──────────────┬───────────────────┬──────────┐
│ factor_id     │ forward_periods │ universe_id │ primary_stat │ primary_stat_name │ primary_p│
│ str           │ i64             │ str         │ f64          │ str               │ f64      │
╞═══════════════╪═════════════════╪═════════════╪══════════════╪═══════════════════╪══════════╡
│ quality_roe   │ 1               │ large_cap   │ 3.21         │ t_nw              │ 0.0013   │
│ momentum_12_1 │ 1               │ large_cap   │ 2.84         │ t_nw              │ 0.0046   │
│ value_btm     │ 1               │ large_cap   │ 1.92         │ t_nw              │ 0.0550   │
└───────────────┴─────────────────┴─────────────┴──────────────┴───────────────────┴──────────┘
```

`primary_stat_name` looks redundant when every entry shares one
procedure, but it is the only disambiguation for mixed lists — for
example a Newey-West t-stat alongside a block-bootstrap p-only entry.
The column carries the `StatCode.value` slug (`"t_nw"` / `"wald_nwcl"`
/ `"p_boot"` / …).

### `list[MetricsBundle]`

```
┌───────────────┬─────────────────┬─────────────┬───────┬───────┬───────────┬──────────┐
│ factor_id     │ forward_periods │ universe_id │ ic    │ ic_ir │ fm_lambda │ hit_rate │
│ str           │ i64             │ str         │ f64   │ f64   │ f64       │ f64      │
╞═══════════════╪═════════════════╪═════════════╪═══════╪═══════╪═══════════╪══════════╡
│ quality_roe   │ 1               │ large_cap   │ 0.051 │ 1.83  │ 0.0042    │ 0.561    │
│ momentum_12_1 │ 1               │ large_cap   │ 0.042 │ 1.52  │ 0.0031    │ 0.547    │
│ value_btm     │ 1               │ large_cap   │ 0.028 │ 0.89  │ 0.0019    │ 0.521    │
└───────────────┴─────────────────┴─────────────┴───────┴───────┴───────────┴──────────┘
```

One column per metric, projected from `MetricOutput.value`. Per-cell
`n_obs` (first-class on [`MetricOutput`](metric-output.md)) is *not*
flattened — for 4 metrics that would double the table to 8 columns
and drown the leaderboard. When you need sample-size honesty for a
specific cell, look up `bundle[metric].n_obs` directly.

### `Survivors`

```python
survivors = fx.multi_factor.bhy(profiles, q=0.05)
fx.compare(survivors, sort_by="adj_p")
```

```
┌───────────────┬─────────────────┬─────────────┬──────────────┬───────────────────┬──────────┬────────┐
│ factor_id     │ forward_periods │ universe_id │ primary_stat │ primary_stat_name │ primary_p│ adj_p  │
│ str           │ i64             │ str         │ f64          │ str               │ f64      │ f64    │
╞═══════════════╪═════════════════╪═════════════╪══════════════╪═══════════════════╪══════════╪════════╡
│ quality_roe   │ 1               │ large_cap   │ 3.21         │ t_nw              │ 0.0013   │ 0.0078 │
│ momentum_12_1 │ 1               │ large_cap   │ 2.84         │ t_nw              │ 0.0046   │ 0.0120 │
└───────────────┴─────────────────┴─────────────┴──────────────┴───────────────────┴──────────┴────────┘
```

When `bhy(..., expand_over=[k, ...])` was used, the partitioning keys
are already inside `profile.context[k]`. `compare` reads them through
the same context-key path as every other context column — there is no
sidecar `expand_over_values` field on `Survivors`, and the renderer
does no reverse lookup.

## Column policy

| Concern | Behaviour |
|---|---|
| Identity | Always flattens to `factor_id` + `forward_periods` (two columns), matching `FactorProfile.identity` and `MetricsBundle.identity`. |
| Context | Union of keys across entries, ordered by first appearance; missing keys fill with `null`. Matches `pl.concat(how="diagonal")`. |
| `sort_by` | `None` keeps input order; otherwise polars sort with `nulls_last=True`. Unknown column raises with a fuzzy suggestion. |
| Mixed-type list | `FactorProfile` and `MetricsBundle` cannot be mixed — raises with the offending indices. |
| Empty input | `[]` and empty `Survivors` raise (rather than returning a schema-undefined empty frame). |

## Errors

`compare` raises [`UserInputError`][factrix.UserInputError] for every input shape
issue (empty input, mixed types, unknown `sort_by`). Unknown
`sort_by` carries `suggestions` populated by `difflib` against the
output schema.

