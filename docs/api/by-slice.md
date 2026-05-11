# by_slice

Axis-agnostic research dispatcher. Slices any metric's date-keyed
input by a column already present in the DataFrame and runs the metric
per slice. Returns `dict[label_value, MetricOutput]`.

> Earlier docs called this role "Layer A" and curated wrappers
> "Layer B"; renamed in
> [#157](https://github.com/awwesomeman/factrix/issues/157). Roles
> unchanged.

`by_slice` supersedes [`by_regime`](by-regime.md) (deprecated since
v0.10.0) and generalises it to any cross-section axis — market,
sector, regime, market-cap tier, ADV bucket — without baking the axis
name into the API.

## Call shape

```python
import polars as pl
from factrix import by_slice
from factrix.metrics import compute_ic, ic

ic_df = compute_ic(panel)
ic_df = ic_df.join(regime_labels, on="date")  # adds 'regime' column

per_regime = by_slice(ic, ic_df, label="regime")
# {"bull": MetricOutput(name="ic", ...), "bear": MetricOutput(name="ic", ...)}
```

The first argument is the **metric callable** (e.g. `ic`, `caar`,
`fama_macbeth`); the second is the metric's primary date-keyed
DataFrame **with the slicing column already present**; `label` names
that column; remaining keyword args (`forward_periods=...`, etc.)
forward unchanged on every per-slice call.

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
not a defensible cross-regime selection rule. A generic second-layer test
(BHY adjustment, Sharpe-diff Wald, paired-difference NW, etc.) cannot
be applied honestly across the metric matrix — the appropriate test
depends on the metric family. For metrics that expose a
`per_date_series` capability (`ic`, `fama_macbeth`, `hit_rate`),
[`slice_pairwise_test`](slice-test.md#factrix.slicing.inference.slice_pairwise_test)
/ [`slice_joint_test`](slice-test.md#factrix.slicing.inference.slice_joint_test)
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

## API reference

::: factrix.slicing.dispatcher
    options:
      members:
        - by_slice
