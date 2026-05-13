---
title: factrix.slice_pairwise_test / factrix.slice_joint_test
---

::: factrix.slice_pairwise_test

::: factrix.slice_joint_test

Cross-slice statistical-test function pair. Both consume a metric callable
and a date-keyed DataFrame whose `label` column carries the slice
identifier; the functions partition by `label`, line up per-date metric
series across slices, and report inference on whether the slices'
means differ.

The two functions answer **different statistical questions**:

| Function | Question | Output shape |
|---|---|---|
| `slice_pairwise_test` | "Which pairs differ?" — K(K−1)/2 contrasts with family-internal multiple-testing correction | One row per pair: `(slice_a, slice_b, n_obs, stat, p_raw, p_adj)` |
| `slice_joint_test` | "Do any slices differ at all?" — single omnibus Wald χ² | One row: `(n_obs, k_slices, df, stat, p)` |

Both functions sit in the **View** class (per [#148](https://github.com/awwesomeman/factrix/issues/148)
function classification): their headline output is a comparison test
result. They do **not** participate in BHY family expansion — adjusted
p is a within-slice-family closure, not a cell-level discovery
commitment.

## Metric capability requirement

The metric callable's module must declare `per_date_series` (a
top-level capability function returning a `(date, value)` long-form
frame); IC, Fama-MacBeth, and hit_rate ship with this declaration.
A metric without it raises `TypeError` at the function call site.

See the docstring Examples blocks above for the canonical
per-sub-universe construction (`compute_ic` per sector, concatenated
with a `sector` label column).

## Date alignment is required

Both functions join all slices on `date` and run inference on the
intersected rows. Joint NW HAC over the (T, K) per-date metric panel
needs aligned rows so cross-slice covariance enters through the joint
kernel. Slices with **disjoint date supports** (e.g. regimes split by
time period) yield zero aligned rows and the functions raise `ValueError`.

For genuinely time-disjoint slices, run inference per slice and
compare summaries upstream (or wait for the future
`factor_decomposition` function, which adopts a different SE geometry).
Date-shared slices — universe, sector, market-cap tier — are the
intended use case.

## Estimator dispatch

| Estimator | Inference path | `stat` column carries |
|---|---|---|
| `WaldNWCluster` (default) | Joint NW HAC over the (T, K) per-date metric panel; per-pair Wald χ² via single-row restriction matrix on the joint variance | Wald χ² |
| `BlockBootstrap` | Joint block-bootstrap on the same panel; per-pair p from `\|mean diff\|` against the bootstrap null distribution | Signed mean diff |

`BlockBootstrap` shares one set of block indices across all pair diffs
per draw, so the bootstrap distribution preserves cross-pair
dependence — the joint structure Romano-Wolf step-down relies on.

`slice_joint_test` accepts only `WaldNWCluster`; the omnibus Wald χ² has
no canonical bootstrap analogue, so the function steers callers to
`slice_pairwise_test` if a bootstrap path is wanted.

## Multiple-testing correction (`slice_pairwise_test` only)

| Method | Default for | Notes |
|---|---|---|
| `"holm"` | `WaldNWCluster` (default) | Holm step-down — conservative under arbitrary dependence |
| `"romano_wolf"` | `BlockBootstrap` | Step-down using the joint bootstrap distribution; near-optimal for date-shared slices (universe / sector) |
| `"bonferroni"` | Manual opt-in | For literature / cross-tool reproduction |

`multiple_testing="romano_wolf"` with an analytic estimator raises
`ValueError` — RW needs a bootstrap distribution that analytic
estimators do not produce.

## Cross-axis composition

The functions accept a **single** `label` column. For cross-axis slice
analysis (regime × universe), compose a composite label upstream
with `pl.concat_str(...)`:

```python
ic_df = ic_df.with_columns(
    pl.concat_str(["regime", "universe"], separator="_").alias("regime_x_universe")
)
slice_pairwise_test(ic, ic_df, label="regime_x_universe")
```

Two-way *interaction decomposition* (main effect + interaction with
double-clustered SE) is a different statistical object and is
reserved for the future `factor_decomposition` function.

## Responsibility boundaries

| Need | Use |
|---|---|
| Descriptive per-slice metric values (no test) | [`by_slice`](by-slice.md) |
| Which slice pairs differ statistically | `slice_pairwise_test` |
| Whether any slice differs (omnibus) | `slice_joint_test` |
| FDR-adjusted survivor selection across factors | `bhy(profiles, ...)` |
| Multi-factor leaderboard rendering | `compare(...)` |

