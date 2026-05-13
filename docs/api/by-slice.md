---
title: factrix.by_slice
---

::: factrix.by_slice

::: factrix.SliceResult

Axis-agnostic research dispatcher. Slices any metric's date-keyed
input by a column already present in the DataFrame and runs the metric
per slice. Returns a [`SliceResult`](#sliceresult) — a
`Mapping[str, MetricOutput]` with a `.to_frame()` long-form renderer.

The axis name does not bake into the API — market, sector, regime,
market-cap tier, ADV bucket all share the same dispatcher.

## Argument contract

The first argument is the **metric callable** (e.g. `ic`, `caar`,
`fama_macbeth`); the second is the metric's primary date-keyed
DataFrame **with the slicing column already present**; `label` names
that column; remaining keyword args (`forward_periods=...`, etc.)
forward unchanged on every per-slice call. See the docstring
Examples block above for the canonical call shape.

`SliceResult` is a `Mapping[str, MetricOutput]`:
`result["<slice-key>"].value` for dict-style access,
`result.to_frame()` for the long-form `pl.DataFrame`.

## Why "label is a column name"

`by_slice` does not accept a separate labels DataFrame. Users typically
already carry the partition key (market / sector / regime) on the panel
or can join it once upstream — splitting that join into the
dispatcher's signature is redundant and forces a fixed
`(date, label)` shape that does not generalise across axes (sector
labels are by-asset, not by-date).

If `label` is not in `df.columns`, `by_slice` raises `ValueError` —
compose the column upstream with `df.with_columns(...)` or
`df.join(...)`.

## What it does **not** do

`by_slice` performs **no cross-slice statistical inference**. It
returns the per-slice outputs and stops. Per-slice t-stats / SE in
each `MetricOutput` are computed on that slice alone (different `N`,
different autocorrelation structure per slice) and are **not**
directly comparable — `max(out.values(), key=lambda m: m.tstat)` is
not a defensible cross-regime selection rule. A generic cross-slice test
(BHY adjustment, Sharpe-diff Wald, paired-difference NW, etc.) cannot
be applied honestly across the metric matrix — the appropriate test
depends on the metric family. For metrics that expose a
`per_date_series` capability (`ic`, `fama_macbeth`, `hit_rate`),
[`slice_pairwise_test`](slice-test.md#factrix.slice_pairwise_test)
/ [`slice_joint_test`](slice-test.md#factrix.slice_joint_test)
provide cross-slice contrasts with joint-HAC or block-bootstrap
inference.

## Universe overlap reference patterns

`by_slice` only partitions on a single column's distinct values. Any
overlapping-universe scenario — same row needs to count toward
multiple slices — is composed with three lines of polars upstream.
The shared idiom: `filter` + `with_columns(label=...)` per target
slice, then `pl.concat`.

### 1. Superset (subset and full set side-by-side)

`market` is "TWSE" / "OTC"; you want three slices: 上市, 上櫃, 全市場
(every row also belongs to 全市場).

```python
import polars as pl

mapping = {"TWSE": "上市", "OTC": "上櫃"}
expanded = pl.concat([
    df.with_columns(pl.col("market").replace(mapping).alias("uni")),
    df.with_columns(pl.lit("全市場").alias("uni")),
])
by_slice(ic, expanded, label="uni")
```

### 2. Multi-membership (one stock in multiple indices)

Each index membership is a separate boolean column; the same row may
have several `True` values.

```python
expanded = pl.concat([
    df.filter(pl.col("in_sp500")).with_columns(pl.lit("SP500").alias("uni")),
    df.filter(pl.col("in_nasdaq100")).with_columns(pl.lit("N100").alias("uni")),
])
by_slice(ic, expanded, label="uni")
```

### 3. Hierarchical nesting (Top-10 ⊂ Top-50 ⊂ LargeCap ⊂ All)

Nested by market-cap rank; each tier contains every smaller tier.

```python
tiers = [(10, "Top10"), (50, "Top50"), (200, "LargeCap")]
expanded = pl.concat([
    *[
        df.filter(pl.col("market_cap_rank") <= cutoff)
          .with_columns(pl.lit(name).alias("tier"))
        for cutoff, name in tiers
    ],
    df.with_columns(pl.lit("All").alias("tier")),
])
by_slice(ic, expanded, label="tier")
```

### 4. Sliding window (overlapping ADV buckets)

Adjacent deciles overlap to smooth boundary noise:

```python
windows = [(0, 30), (10, 40), (20, 50), (30, 60),
           (40, 70), (50, 80), (60, 90), (70, 100)]
expanded = pl.concat([
    df.filter((pl.col("adv_pct") >= lo) & (pl.col("adv_pct") < hi))
      .with_columns(pl.lit(f"W[{lo},{hi})").alias("adv_win"))
    for lo, hi in windows
])
by_slice(ic, expanded, label="adv_win")
```

### 5. Cross-product / multi-axis (universe × sector)

Not an overlap problem — `label` only takes a single column. Compose a
composite column with `pl.concat_str`:

```python
df = df.with_columns(
    pl.concat_str(["market", "sector"], separator="-").alias("uni_sec")
)
by_slice(ic, df, label="uni_sec")
# keys: "TWSE-Tech", "TWSE-Finance", "OTC-Tech", ...
```

The API does not accept `label: list[str]`: single-column vs
multi-column semantics would diverge on output dict key type
(`str` vs `tuple`), breaking the `dict[str, MetricOutput]` convention
downstream.

### General template

Cases 1–4 share one idiom — per-target-slice `filter` +
`with_columns(label=...)`, then `pl.concat`. Overlapping rows are
duplicated naturally by the concat.

```python
expanded = pl.concat([
    df.filter(expr).with_columns(pl.lit(name).alias("group"))
    for name, expr in user_definitions.items()
])
by_slice(metric, expanded, label="group")
```

## SliceResult

`by_slice` returns a `SliceResult`, a `Mapping[str, MetricOutput]`
subclass — every `dict`-shaped consumer (`for k, v in result.items()`,
`result["bull"]`, `len(result)`) keeps working unchanged. The added
value is `.to_frame()`, which flattens per-slice `MetricOutput` rows
into a fixed-schema long-form `pl.DataFrame` for plotting,
leaderboards, and Notebook rendering.

```python
result = by_slice(ic, ic_df, label="regime")
result.to_frame().sort("slice")           # plot-ready, lexicographic order
# shape: (2, 5)
# ┌────────┬──────┬───────┬───────┬──────────┐
# │ slice  │ name │ value │ stat  │ p_value  │
# │ ---    │ ---  │ ---   │ ---   │ ---      │
# │ str    │ str  │ f64   │ f64   │ f64      │
# ╞════════╪══════╪═══════╪═══════╪══════════╡
# │ bear   │ ic   │ -0.02 │ -0.41 │ 0.683    │
# │ bull   │ ic   │ 0.07  │ 2.31  │ 0.024    │
# └────────┴──────┴───────┴───────┴──────────┘

# leaderboard: rank slices by t-stat magnitude
result.to_frame().sort(pl.col("stat").abs(), descending=True)
```

!!! warning "p_value is **per-slice**, not cross-slice-adjusted"

    Each row's `p_value` tests that slice alone against its own null
    (e.g. `ic` mean = 0). Filtering `df.filter(pl.col("p_value") < 0.05)`
    across K parallel slices inflates the family-wise error rate —
    under H0, K=10 sectors yields ≈ 0.4 expected "significant" slices
    by pure chance. The container is for **exploration**; for
    inference claims with FWER / FDR control, use
    [`slice_pairwise_test`](slice-test.md#factrix.slice_pairwise_test)
    (Holm / Romano-Wolf / Bonferroni) or
    [`slice_joint_test`](slice-test.md#factrix.slice_joint_test)
    (omnibus χ²).

### Schema

`to_frame()` always returns the same five columns in the same order:

| Column      | Source                     | `None` when                                  |
|-------------|----------------------------|----------------------------------------------|
| `slice`     | mapping key (rename via `slice_col=`) | never                              |
| `name`      | `MetricOutput.name`        | never                                        |
| `value`     | `MetricOutput.value`       | never                                        |
| `stat`      | `MetricOutput.stat`        | descriptive metric / short-circuit failure   |
| `p_value`   | `metadata["p_value"]`      | descriptive metric / short-circuit failure   |

`stat` and `p_value` semantics follow the underlying metric (`stat` may
be a *t*, *z*, *F*, or *χ²* — see `metadata["stat_type"]`; `p_value`
may be one- or two-sided per the metric's null hypothesis). When
concatenating frames from multiple metrics
(`pl.concat([by_slice(ic, ...).to_frame(), by_slice(hit_rate, ...).to_frame()])`),
the `stat` column mixes statistic types and is not directly comparable
across rows of different `name`.

Row order matches iteration order of the result, which matches the
upstream `polars.DataFrame.partition_by` order (insertion order of
distinct values, **not** lexicographic). For plotting and
leaderboards, sort explicitly downstream (`.sort("slice")` for
deterministic axis order, `.sort("value")` / `.sort("stat")` for
ranking). The example block above shows the plot-ready idiom.

### Why a fixed schema instead of `cols=...`?

Quant exploration overwhelmingly wants the same five columns — slice,
metric name, effect size, test statistic, p-value. A configurable
`cols=` parameter would have to choose between (a) a fixed lookup set
that excludes p-value (which lives in `metadata`) or (b) per-slice
metadata key discovery, whose candidate set drifts across metrics and
even across success vs. short-circuit paths within one metric. Fixed
schema avoids both failure modes; for metric-specific metadata
(`tie_ratio`, `shanken_correction`, ...), build the frame directly:

```python
pl.DataFrame(
    [{"slice": k, **m.metadata} for k, m in result.items()]
)
```

