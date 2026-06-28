---
title: factrix.by_slice
---

::: factrix.by_slice

Cross-slice research dispatcher — the partitioned counterpart of
[`evaluate`](index.md). `by_slice` partitions a **raw panel** on a column
already present in it and runs the standard `evaluate` pipeline
independently on each slice, returning the same
`dict[str, EvaluationResult]` shape as `evaluate` (keyed by slice value
rather than factor).

The axis name does not bake into the API — market, sector, regime,
market-cap tier, ADV bucket all share the same dispatcher.

## Argument contract

`by_slice(data, metric, *, by, factor_col, forward_periods=None, strict=True)`:

- `data` — a raw long-format panel, the **same input contract as
  `evaluate`** (`date, asset_id, <factor_col>, forward_return`), with the
  slicing column `by` already present.
- `metric` — a metric **instance** (`ic()`, `caar()`),
  consistent with `evaluate(metrics={...})`. The bare class (`ic`) is
  rejected.
- `by` — the partition column.
- `factor_col` — the single factor to evaluate (multi-factor batching is
  the job of `evaluate`).

The producer→consumer DAG runs **per slice**, so DAG-consumer metrics
(`ic`, `caar`, `fm_beta`, …) work with no pre-computation — exactly as
they do under `evaluate`.

## Mental model: "evaluate, partitioned"

Each slice is evaluated as an **independent dataset** — it sees only its
own rows. The consequence depends on the slicing axis:

- **Cross-sectional partition** (sector, size bucket; the value is
  constant within an asset): each slice is an independent universe with
  intact per-asset history. This is the primary intent.
- **Date-axis partition** (year, regime; the value varies within an asset
  over time): a metric whose aggregation looks across dates —
  rolling-window betas, per-asset time-series regressions, event windows
  (`ts_beta`, `mfe_mae`, `oos_decay`) — sees **truncated history** at
  slice boundaries, so its per-slice value differs from the full-sample
  value decomposed by period. Per-date metrics (`ic`, `fm_beta`,
  `quantile`, `hit_rate`) are unaffected.

`by_slice` emits a `WarningCode.SLICE_BOUNDARY_TRUNCATION` warning when a
cross-date metric is sliced on a date axis. If you want the full-sample
metric decomposed by period instead, compute it once on the whole panel
and group the per-date output yourself.

If `by` is not in `data.columns`, `by_slice` raises `ValueError` —
compose the column upstream with `data.with_columns(...)` or
`data.join(...)`.

## Cross-slice comparison table

`by_slice` returns a plain `dict[str, EvaluationResult]` (the same shape
as `evaluate`), so the standard `EvaluationResult.to_frame` stacking
idiom builds a comparison table — tag each row with its slice key:

```python
result = by_slice(panel, ic(), by="sector", factor_col="factor")

pl.concat([
    r.to_frame().with_columns(pl.lit(k).alias("slice"))
    for k, r in result.items()
]).sort("slice")
# columns: slice, factor, n_assets, metric_name, value, p_value, stat, n_obs, warning_codes
```

Each `EvaluationResult` carries per-slice `n_periods` / `n_assets`, so
sample-size differences across slices are visible directly.

!!! warning "`p_value` is **per-slice**, not cross-slice-adjusted"

    Each slice's `p_value` tests that slice alone against its own null
    (e.g. `ic` mean = 0). Filtering across K parallel slices inflates the
    family-wise error rate (FWER) — under H0, K=10 sectors yields ≈ 0.4
    expected "significant" slices by pure chance. `by_slice` is for
    **exploration**; for inference claims with FWER / false discovery rate
    (FDR) control, use
    [`slice_pairwise_test`](slice-test.md#factrix.slice_pairwise_test)
    (Holm / Romano-Wolf / Bonferroni) or
    [`slice_joint_test`](slice-test.md#factrix.slice_joint_test)
    (omnibus χ²).

## What it does **not** do

`by_slice` performs **no cross-slice statistical inference**. It returns
the per-slice results and stops. Per-slice t-stats / SE are computed on
that slice alone (different `N`, different autocorrelation structure per
slice) and are **not** directly comparable — picking the top slice by
t-stat is not a defensible selection rule. A generic cross-slice test
(Benjamini-Hochberg-Yekutieli (BHY) adjustment, Sharpe-diff Wald,
paired-difference Newey-West (NW), etc.) cannot be applied honestly across
the metric matrix — the appropriate test depends on the metric family.
For metrics that expose a `per_date_series` capability (`ic`, `fm_beta`,
`hit_rate`),
[`slice_pairwise_test`](slice-test.md#factrix.slice_pairwise_test) /
[`slice_joint_test`](slice-test.md#factrix.slice_joint_test) provide
cross-slice contrasts with joint-heteroskedasticity-and-autocorrelation-consistent
(HAC) or block-bootstrap inference.

## Universe overlap reference patterns

`by_slice` only partitions on a single column's distinct values. Any
overlapping-universe scenario — same row needs to count toward
multiple slices — is composed with three lines of polars upstream.
The shared idiom: `filter` + `with_columns(by=...)` per target
slice, then `pl.concat`.

### 1. Superset (subset and full set side-by-side)

`market` is "TWSE" / "OTC"; you want three slices: 上市, 上櫃, 全市場
(every row also belongs to 全市場).

```python
import polars as pl

mapping = {"TWSE": "上市", "OTC": "上櫃"}
expanded = pl.concat([
    panel.with_columns(pl.col("market").replace(mapping).alias("uni")),
    panel.with_columns(pl.lit("全市場").alias("uni")),
])
by_slice(expanded, ic(), by="uni", factor_col="factor")
```

### 2. Multi-membership (one stock in multiple indices)

Each index membership is a separate boolean column; the same row may
have several `True` values.

```python
expanded = pl.concat([
    panel.filter(pl.col("in_sp500")).with_columns(pl.lit("SP500").alias("uni")),
    panel.filter(pl.col("in_nasdaq100")).with_columns(pl.lit("N100").alias("uni")),
])
by_slice(expanded, ic(), by="uni", factor_col="factor")
```

### 3. Hierarchical nesting (Top-10 ⊂ Top-50 ⊂ LargeCap ⊂ All)

Nested by market-cap rank; each tier contains every smaller tier.

```python
tiers = [(10, "Top10"), (50, "Top50"), (200, "LargeCap")]
expanded = pl.concat([
    *[
        panel.filter(pl.col("market_cap_rank") <= cutoff)
             .with_columns(pl.lit(name).alias("tier"))
        for cutoff, name in tiers
    ],
    panel.with_columns(pl.lit("All").alias("tier")),
])
by_slice(expanded, ic(), by="tier", factor_col="factor")
```

### 4. Sliding window (overlapping ADV buckets)

Adjacent deciles overlap to smooth boundary noise:

```python
windows = [(0, 30), (10, 40), (20, 50), (30, 60),
           (40, 70), (50, 80), (60, 90), (70, 100)]
expanded = pl.concat([
    panel.filter((pl.col("adv_pct") >= lo) & (pl.col("adv_pct") < hi))
         .with_columns(pl.lit(f"W[{lo},{hi})").alias("adv_win"))
    for lo, hi in windows
])
by_slice(expanded, ic(), by="adv_win", factor_col="factor")
```

### 5. Cross-product / multi-axis (universe × sector)

Not an overlap problem — `by` only takes a single column. Compose a
composite column with `pl.concat_str`:

```python
panel = panel.with_columns(
    pl.concat_str(["market", "sector"], separator="-").alias("uni_sec")
)
by_slice(panel, ic(), by="uni_sec", factor_col="factor")
# keys: "TWSE-Tech", "TWSE-Finance", "OTC-Tech", ...
```

The API does not accept `by: list[str]`: single-column vs
multi-column semantics would diverge on output dict key type
(`str` vs `tuple`), breaking the `dict[str, EvaluationResult]` convention
downstream.

### General template

Cases 1–4 share one idiom — per-target-slice `filter` +
`with_columns(by=...)`, then `pl.concat`. Overlapping rows are
duplicated naturally by the concat.

```python
expanded = pl.concat([
    panel.filter(expr).with_columns(pl.lit(name).alias("group"))
    for name, expr in user_definitions.items()
])
by_slice(expanded, ic(), by="group", factor_col="factor")
```
