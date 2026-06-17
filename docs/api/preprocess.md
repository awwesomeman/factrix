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

## Forward return

::: factrix.preprocess.compute_forward_return

::: factrix.preprocess.winsorize_forward_return

::: factrix.preprocess.compute_abnormal_return

## Factor normalization

::: factrix.preprocess.mad_winsorize

::: factrix.preprocess.cross_sectional_zscore

## Orthogonalization

::: factrix.preprocess.orthogonalize_factor

::: factrix.preprocess.OrthogonalizeResult
