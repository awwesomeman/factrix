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
| `slice_pairwise_test` | "Which pairs differ?" — K(K−1)/2 contrasts with family-internal multiple-testing correction | One row per pair: `(slice_a, slice_b, n_obs, mean_diff, stat, p_raw, p_adj, stat_type, reference_dist, df_num, df_denom, multiplicity)` |
| `slice_joint_test` | "Do any slices differ at all?" — single omnibus Wald χ² | One row: `(n_obs, k_slices, stat, p_value, stat_type, reference_dist, df_num, df_denom, multiplicity)` |

Both functions sit in the **View** class: their headline output is a comparison test
result. They do **not** participate in Benjamini-Hochberg-Yekutieli (BHY) family expansion — adjusted
p is a within-slice-family closure, not a cell-level discovery
commitment.

## Evaluation-grid overlap (`overlap_periods`)

The joint NW HAC bandwidth is floored at the evaluation-grid overlap —
overlapping forward returns make each per-date series autocorrelated, and a
kernel that does not cover the overlap under-estimates the variance. There
is no sample floor to gate here, so the overlap drives the bandwidth alone.
Resolution follows `evaluate`'s contract, the same one the `slice_period_*`
pair applies: the stamp left by `compute_forward_return` is the truth, a
declared `overlap_periods=` disagreeing with it is rejected, and an
unstamped (self-attached `forward_return`) panel must declare it rather than
fall back to a silent default.

## Metric capability requirement

The metric callable's module must declare `per_date_series` (a
top-level capability function returning a `(date, value)` long-form
frame); information coefficient (IC), Fama-MacBeth, and positive_rate ship with this declaration.
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
  per slice (e.g. `ic` drops any date below `MIN_IC_ASSETS_HARD`); a `sector`
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

Under `method="bootstrap"` both take `n_resamples=` (default 999, floored
at 199) and `seed=` — the library-wide resampling knobs, see
[Resampling knobs](../reference/statistical-methods.md#resampling-knobs).
`"analytic"` ignores both.

Each slice's per-period series must clear the metric's own sample floor,
resolved at the panel's stamped or declared `overlap_periods` — the same
floor `by_slice` short-circuits on — otherwise the test raises `ValueError`
rather than return an uncalibrated contrast. Plan the partition against
[`sample_requirements`](inspect-data.md#resolved-floor-for-a-configured-metric)
(e.g. `positive_rate()` needs 10 periods per slice at `overlap_periods=1`,
50 at the default 5). Resolution follows `evaluate`'s contract: the stamp
left by `compute_forward_return` is the truth, a declared
`overlap_periods=` disagreeing with it is rejected, and an unstamped
(self-attached `forward_return`) panel must declare it — the same value
`by_slice(..., overlap_periods=)` takes, so the descriptive and the
inferential path never gate at different floors.

Pairwise output is `(slice_a, slice_b, n_periods_a, n_periods_b,
mean_diff, stat, p_raw, p_adj, stat_type, reference_dist, df_num,
df_denom, multiplicity, min_periods, reason)` — per-slice `n_periods_*`
because disjoint spans differ in length, `min_periods` the floor they
were gated on. In either slice test a pair whose contrast variance
collapses carries NaN in `stat` / `p_raw` / `p_adj` (no test, not a
non-rejection) and is left out of the Holm family; the period tests name
it `reason="degenerate_variance"`. The omnibus is a block-diagonal Wald
χ² returning `(k_slices, n_periods_min, stat, p_value, stat_type,
reference_dist, df_num, df_denom, multiplicity, min_periods, reason)`.

### Non-strict batch mode (`slice_period_*` only)

`strict=False` is the counterpart of `evaluate(strict=False)` for batch
regime research, where one thin regime must not abort a sweep over
factors × samples. A slice below the floor no longer raises; the affected
rows come back in the same schema with `reason="insufficient_periods"`
and NaN in `stat` / `p_*`:

- **joint** — the omnibus restriction spans every slice, so one thin
  slice makes the whole test unavailable: a single row with
  `n_periods_min` (the shortest slice) against `min_periods`.
- **pairwise** — every pair touching a thin slice is an unavailable row;
  the remaining pairs are tested and form their own multiplicity family
  (identical to running the test on the valid slices alone).

`reason` is null on a tested row, so `reason.is_null()` is the filter to
apply before folding rows from many runs into one family.

### Warning contract (`slice_period_joint_test`)

The slicing functions follow the same *marked, never dropped* contract as
`evaluate(..., expected_warnings=)`. The omnibus has one advisory —
`short_slice_joint_test`, fired when `K >= 3` and the shortest slice has
fewer than 150 periods, the regime where the joint Wald over-rejects
(8–9% at a nominal 5% for `K=5` with 50–90-period slices; see the
function's `Warns:` block for the measured grid). It is delivered on the
row, not only on stderr:

| Column | Meaning |
|---|---|
| `warning_codes` | every `WarningCode` the test raised (`list[str]`, empty when clean) |
| `unexpected_warning_codes` | the subset not declared via `expected_warnings` — the alert view |
| `short_slice_periods` | the calibration threshold (150) the code is gated on |

`k_slices` and `n_periods_min` already carry the values it was read
against, so a stacked frame from a batch of regime tests answers which
cells tripped it, whether the caller had declared it, and why. Declaring
the code — `slice_period_joint_test(..., expected_warnings=("short_slice_joint_test",))`
— keeps it in `warning_codes`, drops it from `unexpected_warning_codes`
and stops the per-call `UserWarning` echo, so a sweep over candidates and
rebalance frequencies prints nothing for an accepted limitation while the
audit record survives. Undeclared codes keep the echo. Unknown codes are
rejected. `by_slice` takes the same argument and forwards it to every
per-slice `evaluate`, applying it to its own `slice_boundary_truncation`
record as well; `slice_period_pairwise_test` and the cross-sectional slice
tests raise no advisory today and carry no such columns.

## Inference path (cross-sectional slice tests)

`slice_pairwise_test` and `slice_joint_test` run one analytic path and
take no estimator or multiple-testing argument:

| Function | Inference | Multiplicity | `stat` column carries |
|---|---|---|---|
| `slice_pairwise_test` | Joint Newey-West HAC over the (T, K) per-date metric panel, slice-clustered; per-pair Wald via a single-row restriction on the joint variance | Holm step-down on `p_raw` → `p_adj` (`multiplicity="holm"`) | Wald statistic; `mean_diff` carries the signed effect |
| `slice_joint_test` | Same joint HAC; omnibus Wald over all K−1 contrasts | — (one test) | Omnibus Wald statistic |

Holm is applied internally because the pairwise family is generated by
the function itself; the user-assembled families that `bhy_adjust`
serves are a different object. A bootstrap path exists only where slices
do not share dates: `slice_period_pairwise_test(method="bootstrap")` and
`slice_period_joint_test(method="bootstrap")` resample each period
slice independently (see [Date supports: aligned vs disjoint](#date-supports-aligned-vs-disjoint)).

## Cross-axis composition

The functions accept a **single** `by` column. For cross-axis slice
analysis (regime × universe), compose a composite label upstream
with `pl.concat_str(...)`:

```python title="Illustrative"
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
| FDR-adjusted survivor selection across factors | `bhy(results, ...)` |
| Multi-factor leaderboard rendering | `compare(...)` |

