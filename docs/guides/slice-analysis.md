---
title: Slice Analysis
---

!!! abstract "Answers"
    What slice analysis is, the two-role split (`by_slice` dispatcher vs `slice_pairwise_test` / `slice_joint_test` inference), and when to reach for each.
    For the API surface, see [`by_slice`](../api/by-slice.md) and [`slice_pairwise_test` / `slice_joint_test`](../api/slice-test.md).

Slice analysis asks "is this factor stable across a partition of the panel?" The partition can be a market regime (bull / bear, high-vol / low-vol), a universe (large-cap / small-cap, listed-board / OTC), a sector, an ADV bucket, or any other column you can attach to the panel. The statistical question is the same regardless of axis, so factrix exposes one axis-agnostic surface rather than one function per slice dimension. Concretely: `by_slice` is the dispatcher, `slice_pairwise_test` / `slice_joint_test` are the inference function pair.

factrix splits this work into two roles because **slicing the panel** and **testing significance across slices** are different jobs that need different APIs. The legacy regime-specific surface (`by_regime`, `regime_ic`) was removed in v0.12.0 — see the CHANGELOG migration recipe.

## The two roles

| Role | Function | What it does | What it does not do |
|---|---|---|---|
| Dispatcher | [`by_slice(data, metric, *, by, factor_col)`](../api/by-slice.md) | Partitions a raw panel on an existing column and runs `evaluate` per slice; returns `dict[str, EvaluationResult]` (same shape as `evaluate`, keyed by slice) | **No cross-slice statistical test** |
| Inference | [`slice_pairwise_test`](../api/slice-test.md) / [`slice_joint_test`](../api/slice-test.md) | Pairwise contrasts (Wald χ² + Holm / Romano-Wolf / Bonferroni) or omnibus χ² that all slice means are equal | Only accepts metrics with a `per_date_series` capability (`ic`, `fm_beta`, `hit_rate`) |

**Use the dispatcher when:** you want raw per-slice numbers, or you want to compose your own cross-slice test.

**Use the inference functions when:** you want a calibrated cross-slice statistic with multiple-testing control and your metric exposes `per_date_series`.

## Why no generic cross-slice test on `by_slice`

A single dispatcher carrying a single built-in cross-slice test would silently over-claim. The appropriate test depends on the metric family:

- **Information coefficient (IC), Fama-MacBeth λ** — mean-zero per-date series → joint Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) over the per-date K-vector panel is the natural Wald object (`slice_pairwise_test` default).
- **Sharpe** — variance-stabilised difference (Memmel 2003 / Ledoit-Wolf) is needed; not currently exposed as a slice function.
- **CAAR** — per-slice event clustering interacts with pooled-vs-split SE choice; needs a bespoke reconciliation.
- **Turnover, hit_rate, monotonicity ρ** — for `hit_rate` the `per_date_series` path applies; for the rest cross-slice differences remain descriptive.

`by_slice` therefore returns the raw per-slice `EvaluationResult`s without a cross-slice aggregate. For inferential contrasts on the supported metric families, reach for the slice-test function pair.

## Constructing slice labels

The label is just a column on `df`. The constructions below use regime labels as the running example; the same pattern works for any partition (universe id, sector code, ADV bucket index):

```python
import polars as pl

# 1. External classifier (NBER recession, factor return sign, ...)
vol_labels = pl.DataFrame({"date": dates, "regime": ["bull", "bull", "bear", ...]})

# 2. From a market-vol panel — see lookahead warning below
vix_thresh = vix["vix"].quantile(0.7)
vol_labels = vix.with_columns(
    pl.when(pl.col("vix") > vix_thresh).then(pl.lit("high_vol")).otherwise(pl.lit("low_vol")).alias("regime")
).select("date", "regime")
```

Join the labels onto the metric's per-date input upstream:

```python
ic_df = compute_ic(panel)["factor"].join(vol_labels, on="date", how="inner")
```

!!! warning "Lookahead bias when constructing labels"
    The `vix.quantile(0.7)` snippet above uses **full-sample** statistics
    to label every date — that leaks future information into every
    per-slice IC. For decision-grade analysis, derive thresholds from
    an expanding or rolling window so each date's label only depends on
    information available at that date. The full-sample form is fine
    only for descriptive ex-post analysis.

## Discovering eligible metrics

`by_slice` accepts any metric instance — it runs the full `evaluate` pipeline per slice, so the producer→consumer DAG resolves automatically. The inference functions are narrower: they require the metric module to declare a `per_date_series` capability and take the metric's per-date series (not a raw panel) as input. Enumerate the candidate set for the inference path from the [`list_metrics()`](../api/metrics/index.md#factrix.list_metrics) catalog by filtering each spec's `input_shape`:

```python
import factrix as fx

panel_metrics = [
    spec.name
    for specs in fx.list_metrics().values()
    for spec in specs
    if spec.input_shape.value == "panel"
]
```

Scalar-input utilities (`breakeven_cost`, `net_spread`) are excluded — they consume pre-aggregated scalars and have no date column to slice on.

## Worked example: IC across volatility regimes

The dispatcher and the inference pair take **different inputs**, because
they answer different questions. `by_slice` partitions the **raw panel**
and runs `evaluate` per slice (each slice an independent dataset); the
inference functions consume the **per-date metric series** to line up
aligned observations across slices for a cross-slice Wald test.

```python
import polars as pl
from factrix import by_slice, slice_pairwise_test
from factrix.metrics import compute_ic, ic

# --- Dispatcher: per-regime IC, raw panel in, dict[str, EvaluationResult] out
panel_reg = panel.join(vol_labels, on="date", how="inner")  # attach regime to panel
per_regime = by_slice(panel_reg, ic(), by="regime", factor_col="value")
for label, result in per_regime.items():
    m = result.metrics["metric"]
    print(label, m.value, m.stat)

# Cross-slice comparison table (stack EvaluationResult.to_frame, tag the slice)
pl.concat([
    r.to_frame().with_columns(pl.lit(k).alias("slice"))
    for k, r in per_regime.items()
])

# --- Inference: per-date IC series in, pairwise Wald contrasts out
ic_df = compute_ic(panel, factor_cols=["value"], return_col="forward_return")["value"]
merged = ic_df.join(vol_labels, on="date", how="inner")
pairs = slice_pairwise_test(merged, ic, by="regime")
print(pairs)  # columns: slice_a, slice_b, n_obs, stat, p_raw, p_adj
```

## Related

- [`by_slice`](../api/by-slice.md) — dispatcher surface and universe-overlap recipes.
- [`slice_pairwise_test` / `slice_joint_test`](../api/slice-test.md) — cross-slice inference function pair.
- [Benjamini-Hochberg-Yekutieli (BHY) screening](../api/bhy.md) — false discovery rate (FDR) control across factor candidates rather than slices.
