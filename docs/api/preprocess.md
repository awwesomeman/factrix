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

## Forward return

`compute_forward_return` accepts `forward_periods` as a positive `int`
row horizon. `0`, negative values, floats, strings, and `bool` values
raise [`UserInputError`](errors.md). The function shifts by row count
within each `asset_id`, computes the per-period normalized
`forward_return`, then drops rows whose computed return is not finite
(`null`, `NaN`, `+inf`, or `-inf`). If the horizon is too long for the
panel, or price data leaves no finite forward returns after filtering,
the function raises [`UserInputError`](errors.md) instead of returning
an empty panel.

`winsorize_forward_return` clips `forward_return` by per-date quantiles.
Its bounds must satisfy `0 <= lower <= upper <= 1`; invalid ordering,
out-of-range values, non-numeric values, and `bool` bounds raise
[`UserInputError`](errors.md).

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

::: factrix.preprocess.orthogonalize_factor

::: factrix.preprocess.OrthogonalizeResult
