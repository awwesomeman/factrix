---
title: factrix.preprocess
---

Helpers for shaping a raw panel before [`evaluate`](evaluate.md). The
canonical entry point, `compute_forward_return`, attaches a
`forward_return` column to a raw `(date, asset_id, price)` panel — the
output `(date, asset_id, factor, forward_return)` panel is the canonical
input to `evaluate`.

The surrounding helpers cover the rest of the documented preprocessing
pipeline and are independently usable on a canonical panel: return
cleaning (`winsorize_forward_return`, `compute_abnormal_return`), factor
normalization (`mad_winsorize`, `cross_sectional_zscore`), and
orthogonalization against base factors (`orthogonalize_factor`).

## Column adaptation

Use `adapt` when the input is already a long panel but carries vendor- or
project-specific column names such as `trade_date`, `ticker`, or `close_adj`.
It renames those columns to factrix's canonical `date`, `asset_id`, `price`,
and optional OHLCV names before `compute_forward_return`. It does not reshape
data, construct factors, or compute returns.

`adapt` preserves Polars eager/lazy inputs, converts pandas input to Polars,
and leaves unrelated columns such as factors, industries, market caps, or
regime labels unchanged. Optional `fill_forward` is a raw-OHLCV convenience:
it maps non-finite numeric values to null and forward-fills per asset before
forward returns are computed.

::: factrix.adapt.adapt

## The panel contract

Every public entry point — `evaluate`, `evaluate_horizons`, `inspect_data`,
`compute_forward_return` — puts its input through one structural gate before
anything else happens. Three things it enforces, each of which was previously
left to individual producers and therefore missed by whichever path forgot it:

- **`date` must be `Date` or `Datetime`.** Only column *names* were validated,
  so a `String` date flowed through `sort` / `shift` / `over` / `group_by` as
  text and was ordered lexicographically. With ISO-8601 strings that
  accidentally works, which is what makes it dangerous — it passes every test
  and every demo. With `MM/DD/YYYY` the panel is silently reordered and every
  forward return is paired with the wrong neighbour. Parse first, e.g.
  `pl.col("date").str.to_datetime("%m/%d/%Y")`.
- **`(date, asset_id)` must be unique.** A duplicated row makes an asset's
  "next period" that same date's twin, so `price / price - 1 = 0`: a four-row
  panel concatenated with itself came back half fabricated zeros, biasing every
  downstream mean toward zero with no error and no warning.
- **NaN and ±Inf become `null`.** Applied to *inputs*, not to computed outputs.
  This makes the whole class structurally unrepresentable downstream rather
  than patching instances of it: polars ranks NaN as larger than every real
  value, `pl.corr(method="spearman")` silently ranks it, and a `+Inf`
  denominator makes `finite / inf` evaluate to `0.0` — a *finite* fabricated
  −100% return that an output-side `is_finite()` filter cannot catch.

The first two raise [`UserInputError`](errors.md); the third is a silent,
deliberate normalization. `_finite_expr` remains in place inside the producers
as defence in depth.

## Forward return

`compute_forward_return` accepts `forward_periods` as a positive `int`.
`0`, negative values, floats, strings, and `bool` values raise
[`UserInputError`](errors.md).

**The horizon is measured on the panel's own period grid** — the distinct
sorted `date` values present in the panel, indexed 0, 1, 2, … Each asset's
forward return pairs its row at period index `i + 1` with its row at
`i + 1 + forward_periods`. This used to be a positional shift within each
asset, which equals a time horizon only on a *complete* per-asset panel: with
one asset missing 20 periods mid-sample, a "5-period" return silently spanned
25 real periods, contaminating both the return and the overlap stamped in
`_overlap_periods` that every downstream HAC inference reads. Suspensions,
halts, delist-relist and staggered entry are ordinary in regional equity data,
and sparse event panels are ragged by construction.

A ragged grid raises `ragged_period_grid`. The pairing is correct for every
asset either way; the point of the warning is that an asset with a gap has no
observation to pair at the exit period and so contributes fewer rows than a
complete one. Reindex onto a common grid if the horizons must be comparable
across names.

Rows whose computed return is not finite are dropped. If the horizon is too
long for the panel, or price data leaves no finite forward returns after
filtering, the function raises [`UserInputError`](errors.md) instead of
returning an empty panel.

The returned rows are the **forward-return sample**, not a complete price
history. Event offsets and MFE/MAE windows may need the dropped tail even when
their event row remains in the sample. Preserve the raw panel and pass it as
`evaluate(panel, price_data=raw, ...)`; see
[Full price data for event paths](evaluate.md#full-price-data-for-event-paths).

`winsorize_forward_return` clips `forward_return` by per-date quantiles.
Its bounds must satisfy `0 <= lower <= upper <= 1`; invalid ordering,
out-of-range values, non-numeric values, and `bool` bounds raise
[`UserInputError`](errors.md).

## Evaluating on a coarser grid

A factor is often observed on a finer grid than it is traded on: the price
grid supplies the return horizon, while the evaluation happens on a
caller-chosen rebalance grid. Two separate quantities then live on the panel,
and `compute_forward_return` stamps both:

| Stamp | Name on `EvaluationResult` | Meaning |
|---|---|---|
| `_forward_periods` | `forward_periods` | The **return horizon** — the `forward_periods` the return was built with, in periods of the price grid. It names the hypothesis: the `(factor, forward_periods, *params)` identity that `compare` / `bhy` use, and the axis `expand_over` may name. Never changes with the evaluation grid. |
| `_overlap_periods` | `overlap_periods` | The **overlap of adjacent observations on the evaluation grid** — what inference consumes: the HAC bandwidth and effective degrees of freedom, the non-overlapping stride, and the stride-scaled sample floors. Bookkeeping only; it does not join the identity, because the same horizon evaluated on two grids is one hypothesis estimated twice. |

On the full grid the two coincide. That is why sub-sampling a panel **by
hand after** `compute_forward_return` goes wrong: the `overlap_periods` stamp
still says "horizon", so a 60-period return evaluated every 60 periods is
treated as 60-fold overlapping and the stride-scaled floor (`50 × 60`
periods for `ic()`) rejects a healthy 24-period panel with
`insufficient_ic_periods`. The short-circuit message says so and points here.

Pass the evaluation grid as `dates=` instead. The return is still computed on
the full grid at `forward_periods`, only rows on those dates are kept, and
the overlap is derived on the **full** period index:

```
overlap_periods = 1 + max_i #{ j in dates : 0 < idx(j) − idx(i) < h }
```

The row at period `i` covers `(i + 1, i + 1 + h]`, so two kept rows overlap
exactly when they are fewer than `h` periods apart. Stride 60 at `h = 60`
gives 1; stride 20 gives 3; the full grid gives `h`.

```python
import factrix as fx
from factrix.metrics import ic, quantile_spread
from factrix.preprocess import compute_forward_return

raw = fx.datasets.make_cs_panel(n_assets=30, n_dates=1500)      # fine grid
grid = raw["date"].unique().sort()
rebalance_dates = grid.gather_every(60)                        # every 60th period

panel = compute_forward_return(raw, forward_periods=60, dates=rebalance_dates)
result = fx.evaluate(
    panel, metrics={"ic": ic(), "spread": quantile_spread()}, factor_cols=["factor"]
)["factor"]

result.forward_periods   # 60 — the hypothesis
result.overlap_periods   # 1  — derived: adjacent evaluations share no future period
result.metrics["ic"].n_obs        # 24, the sampled periods (no stride applied)
result.metrics["spread"].n_obs    # 24
```

Three details of the derivation are deliberate:

- **Every date must be on the grid; nothing is snapped.** Snapping backward
  would read a factor observed after the entry period (look-ahead); snapping
  forward would silently move the entry period. A value off the grid raises
  [`UserInputError`](errors.md). A `Date` value names a `Datetime` period
  unambiguously and a `Datetime` in another time unit is the same instant;
  any other dtype mismatch is rejected rather than parsed.
- **The maximum, not a typical count.** The evaluation grid may be spaced
  unevenly on the period grid. Under-stating the overlap by one leaves
  dependence the non-overlapping stride does not remove and the stride
  t-test over-rejects (measured 8.3% at a nominal 5% on the test grid in
  `tests/stats/test_uneven_grid_overlap_size.py`, 21.7% with no stride);
  over-stating it by one only thins the strided sample (size intact, some
  power lost). The Newey-West path is insensitive either way (5.5%).
- **Derived inside `compute_forward_return`, on the full index.** The
  `is_finite` filter removes any period on which every asset's return is
  non-finite; an index rebuilt from the returned panel's own dates would
  compress those gaps and under-count the overlap — the dangerous direction.

The unit of `forward_return` does not change with the grid: it is the return
**per period of the horizon** (the `/ forward_periods` normalisation), not
per evaluation period. Every p-value on the affected paths is
scale-invariant, and a per-evaluation-period rate is not defined on an
unevenly spaced grid.

`evaluate_horizons(..., dates=)` forwards the same grid to every horizon of
a sweep; each result carries its own derived `overlap_periods` while the
`(factor, forward_periods)` identity is unaffected.

::: factrix.preprocess.compute_forward_return

::: factrix.preprocess.winsorize_forward_return

::: factrix.preprocess.compute_abnormal_return

## Factor normalization

Both normalizers scale by the median absolute deviation (MAD). Three places
where factrix departs from the most common implementation, and why:

**Finite-sample MAD constant.** The standard scale is `1.4826 × MAD`, where
`1.4826 = 1/Φ⁻¹(0.75)` is the *asymptotic* consistency constant — this is what
R's `mad()` and `scipy.stats.median_abs_deviation` ship. factrix multiplies in
the Croux-Rousseeuw (1992) finite-sample factor `b_n` as well (tabulated for
`n ≤ 9`, `n/(n−0.8)` above), keyed on the per-date count of finite values,
because this library targets cross-sections of 5–20 names where the asymptotic
constant is materially biased. Measured on a standard-normal factor
(true σ = 1), `E[1.4826 × MAD]` is 0.82 at n=5, 0.91 at n=10 and 0.96 at n=20,
against 0.99–1.01 at every size with `b_n`. The bias matters twice over: the
output of a function called "z-score" was not unit-scale, and because the bias
is *n*-dependent, thin dates in an unbalanced panel (delistings, staggered
entry, sparse event triggers) were handed systematically larger z-scores, so
anything pooling or weighting by z over-weighted the noisiest cross-sections.
Fewer than three finite values on a date leaves no estimable robust scale at
all: the date comes back null rather than as a fabricated `0.0`.

**Centring.** The textbook z-score subtracts the mean. `cross_sectional_zscore`
defaults to `center="median"` to stay robust to the outliers the MAD scale is
chosen for — with the consequence that the output is **not** mean-zero on a
skewed factor, so `w ∝ z` carries a net long or short leg (measured: `+0.32`
net exposure on a 2-of-20 sparse trigger column). Pass `center="mean"` when the
weights must be dollar-neutral by construction; the scale stays the robust MAD
either way.

**Regime switches are announced.** When the per-date MAD collapses to zero
(more than 50% ties at the median) the scale falls back to the non-robust
per-date standard deviation. That is a data-driven change of estimator, so it
raises a `UserWarning` carrying
[`WarningCode.ZERO_MAD_STD_FALLBACK`](../reference/warning-codes.md) rather
than mixing robust and non-robust dates into one column silently. On a sparse
`{0, R}` trigger column `mad_winsorize` skips the clip entirely
(`SPARSE_WINSORIZE_SKIPPED`): that column's standard deviation is produced *by
the triggers*, so a 3-std band shrinks with the trigger rate and destroyed 58%
of a unit event's magnitude at one trigger in fifty names.

**Non-finite input.** NaN and ±Inf are blanked to **null** on output by both
functions, not clipped into the band. A non-finite tick is a data error, not an
extreme value; winsorizing it produced a plausible finite number that survived
every downstream `drop_nulls().drop_nans()` and put that asset at the top of
the date's ranking.

`cross_sectional_zscore` names its output column after its input:
`factor` → `factor_zscore`, `momentum` → `momentum_zscore`, so several factors
can be standardized in one panel without colliding.

::: factrix.preprocess.mad_winsorize

::: factrix.preprocess.cross_sectional_zscore

## Orthogonalization

`orthogonalize_factor` runs a per-date cross-sectional OLS and returns the
residual. Two guards that the textbook formulation leaves to the analyst:

**A degrees-of-freedom floor, not a solvability floor.** The regression is only
fitted on dates leaving at least `min_residual_df` residual degrees of freedom
(`n_assets − n_base − 1`, default 10 — the `N ≥ K + 10` form of the
Fama-MacBeth convention). Raw R² is mechanically ≈ `K/(N − 1)` even when the
true R² is zero, so on a six-name book regressed on four exposures the old
one-df floor reported `mean_r_squared = 0.79` (adjusted: `−0.03`) after
stripping 83% of the factor's variance — the factor reads as redundant when it
is orthogonal by construction. Dates below the floor keep their original values,
are counted in `n_dates_insufficient_df`, and raise
`insufficient_regression_df`. `mean_adj_r_squared` is reported alongside the raw
figure. Pass `min_residual_df=1` to restore the old behaviour.

**Rank deficiency is detected, not assumed away.** An intercept is always
prepended, so a full industry dummy set is exactly singular. `np.linalg.lstsq`
does not raise on that — it returns the minimum-norm solution — so the residual
is still exact (the projection onto the column space is unique) but `mean_betas`
was an arbitrary point in the solution space. Rank is read from the singular
values `lstsq` already computes, at `matrix_rank`'s tolerance so *near*-collinear
designs are caught too; deficient dates are excluded from `mean_betas` and raise
`rank_deficient_design`. Drop one dummy category as the reference level.

**Residual scale.** The residual's dispersion is `sqrt(1 − R²)` times the
input's, and R² varies by date, so the output scale varies by date. Rank-based
metrics are unaffected; any magnitude-based use is otherwise quietly on a
time-varying scale. Pass `restandardize=True` to rescale each date's residual
back to the input's per-date dispersion.

::: factrix.preprocess.orthogonalize_factor

::: factrix.preprocess.OrthogonalizeResult
