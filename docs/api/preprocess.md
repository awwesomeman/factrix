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

::: factrix.preprocess.mad_winsorize

::: factrix.preprocess.cross_sectional_zscore

## Orthogonalization

::: factrix.preprocess.orthogonalize_factor

::: factrix.preprocess.OrthogonalizeResult
