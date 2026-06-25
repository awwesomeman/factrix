---
title: factrix.slice_pairwise_test / factrix.slice_joint_test
---

The cross-slice inference surface is **two function pairs**, split on
whether the slices share dates:

- **Cross-sectional / date-aligned** — `slice_pairwise_test` /
  `slice_joint_test` (sector, size bucket, liquidity tier).
- **Date-disjoint** — `slice_period_pairwise_test` /
  `slice_period_joint_test` (market regime, calendar period,
  in/out-of-sample). See [Date supports: aligned vs disjoint](#date-supports-aligned-vs-disjoint).

::: factrix.slice_pairwise_test

::: factrix.slice_joint_test

::: factrix.slice_period_pairwise_test

::: factrix.slice_period_joint_test

Cross-slice statistical-test function pair. Both take a date-keyed
DataFrame (data-first) and a metric callable; the `by` column carries the
slice identifier; the functions partition by `by`, line up per-date metric
series across slices, and report inference on whether the slices'
means differ.

The two functions answer **different statistical questions**:

| Function | Question | Output shape |
|---|---|---|
| `slice_pairwise_test` | "Which pairs differ?" — K(K−1)/2 contrasts with family-internal multiple-testing correction | One row per pair: `(slice_a, slice_b, n_obs, stat, p_raw, p_adj)` |
| `slice_joint_test` | "Do any slices differ at all?" — single omnibus Wald χ² | One row: `(n_obs, k_slices, df, stat, p)` |

Both functions sit in the **View** class: their headline output is a comparison test
result. They do **not** participate in Benjamini-Hochberg-Yekutieli (BHY) family expansion — adjusted
p is a within-slice-family closure, not a cell-level discovery
commitment.

## Metric capability requirement

The metric callable's module must declare `per_date_series` (a
top-level capability function returning a `(date, value)` long-form
frame); information coefficient (IC), Fama-MacBeth, and hit_rate ship with this declaration.
A metric without it raises `TypeError` at the function call site.

See the docstring Examples blocks above for the canonical
per-sub-universe construction (`compute_ic` per sector, concatenated
with a `sector` label column).

## Date supports: aligned vs disjoint

`slice_pairwise_test` / `slice_joint_test` join all slices on `date` and
run inference on the intersected rows. Joint Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) over the (T, K) per-date metric panel
needs aligned rows so cross-slice covariance enters through the joint
kernel. Slices with **disjoint date supports** (e.g. regimes split by
time period) yield zero aligned rows and these functions raise
`ValueError` (`<2 aligned dates`). Date-shared slices — universe,
sector, market-cap tier — are their intended use case.

A `<2 aligned dates` error has **two distinct causes**, and the message
distinguishes them:

- **Date-disjoint partition** — the slices share fewer than two raw dates
  by construction (the case above). The message names the date-disjoint
  partition and points at `slice_period_*`.
- **Date-aligned but metric-dropped** — the slices *do* share dates, but
  the per-slice metric dropped most of its per-date values, so the joined
  panel still collapses below two rows. The usual cause is too few assets
  per slice (e.g. `ic` drops any date below `MIN_IC_ASSETS`); a `sector`
  cut with thin cross-sections triggers it. The message reports the raw
  shared-date count and blames the thin universe — widen each slice's
  asset universe or use a coarser partition.

For genuinely time-disjoint slices, reach for
`slice_period_pairwise_test` / `slice_period_joint_test`. They build the
same per-slice per-date series but **do not** inner-join — each slice is
treated as an independent sample with block-diagonal cross-slice
covariance. A two-valued `method` flag selects the estimator:

| `method` | Per-slice SE | Pairwise `p_adj` | Best for |
|---|---|---|---|
| `"bootstrap"` (default) | Independent stationary block bootstrap (Politis-White automatic block length) | Romano-Wolf step-down | Short regimes (T ≈ 30-80); never invalid |
| `"analytic"` | Per-slice Newey-West HAC, Welch-style pairwise contrast | Holm step-down | Long spans (T ≳ 100); fast, deterministic |

Pairwise output is `(slice_a, slice_b, n_periods_a, n_periods_b,
mean_diff, stat, p_raw, p_adj)` — per-slice `n_periods_*` because
disjoint spans differ in length. The omnibus is a block-diagonal Wald χ²
returning `(k_slices, df, stat, p_value)`.

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

The functions accept a **single** `by` column. For cross-axis slice
analysis (regime × universe), compose a composite label upstream
with `pl.concat_str(...)`:

```python
ic_df = ic_df.with_columns(
    pl.concat_str(["regime", "universe"], separator="_").alias("regime_x_universe")
)
slice_pairwise_test(ic_df, ic, by="regime_x_universe")
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

