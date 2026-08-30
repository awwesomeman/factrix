---
title: Preparing data
---

The reader-flow from a raw price / signal dataset to a
`(date, asset_id, factor, forward_return)` panel that
[`evaluate`](../api/evaluate.md) consumes. For the column-level four-column
contract, see [Data schema](../api/data-schema.md); this page is the
task-oriented walk-through.

## At a glance

| Step | What you do | Function | Output added |
|---|---|---|---|
| 1 | Reshape raw inputs to long format with `price` and canonical names | manual / Polars ops, [`adapt`](../api/preprocess.md#factrix.adapt.adapt) | `(date, asset_id, price, factor)` |
| 2 | Ensure regular spacing per asset on the time axis | manual / Polars ops | spacing-regular panel |
| 3 | Attach forward return | [`compute_forward_return`](../api/preprocess.md) | adds `forward_return` |
| 4 | (Optional) normalize / residualize factor values | [`mad_winsorize`](../api/preprocess.md#factrix.preprocess.mad_winsorize), [`cross_sectional_zscore`](../api/preprocess.md#factrix.preprocess.cross_sectional_zscore), [`orthogonalize_factor`](../api/preprocess.md#factrix.preprocess.orthogonalize_factor) | processed factor column |
| 5 | (Optional) drop / impute NaN, align frequencies | manual | clean panel |

!!! tip "Screening many factors at once?"
    A panel wide enough to hold 100–1000+ candidate columns can exhaust
    RAM in a single `evaluate` call. See
    [Large-scale evaluation](large-scale-evaluation.md) for the caller-side
    batched-loop pattern that bounds peak memory to a fixed working set.

## 1. Long-format shape with `price` and the factor column

factrix expects **long-format** panel data — one row per
`(date, asset_id)` pair. Wide-format (one column per asset) is not
accepted by any entry point.

If your panel is already long but uses source-specific names, adapt the
column names first:

```python title="Illustrative"
from factrix.adapt import adapt

raw = adapt(
    vendor_df,
    date="trade_date",
    asset_id="ticker",
    price="close_adj",
)
```

`adapt` only maps names to factrix canonicals; it does not pivot wide data,
construct factor values, or attach `forward_return`.

[`compute_forward_return`](../api/preprocess.md) computes the
look-ahead return from a `price` column; the factor column is a
parallel signal you construct yourself (factor construction is outside
factrix's scope — see
[Where factrix fits § 1](../where-factrix-fits.md#1-what-factrix-is)).

The factor column name is **user-defined** — `evaluate()` accepts a
`factor_cols` list that binds one or more columns to the
canonical role at dispatch time. The examples below use
`momentum` to make this binding visible; you can equally pick
`alpha`, `value_score`, or whatever is meaningful for the strategy.

For per-asset factors (`INDIVIDUAL` scope), each `(date, asset_id)`
carries its own factor value alongside the price:

```python
import polars as pl
from datetime import date

raw = pl.DataFrame({
    "date":          [date(2024, 1, 1), date(2024, 1, 1),
                      date(2024, 1, 2), date(2024, 1, 2)],
    "asset_id":      ["AAPL", "MSFT", "AAPL", "MSFT"],
    "price":         [185.0, 372.0, 186.5, 374.5],
    "momentum": [0.42, -0.15, 0.51, -0.08],
})
```

For market-wide factors (`COMMON` scope, e.g. VIX, DXY), the factor
value is identical across `asset_id` on a given `date`. Verify with
the same check the [data schema](../api/data-schema.md) states
(swap the column name for whichever the panel carries) — aggregate to a
per-date boolean column, then reduce that `Series`, since `.all()` is a
`Series` method and a `DataFrame` has none:

```python
common_raw = pl.DataFrame({
    "date":     [date(2024, 1, 1), date(2024, 1, 1),
                 date(2024, 1, 2), date(2024, 1, 2)],
    "asset_id": ["AAPL", "MSFT", "AAPL", "MSFT"],
    "price":    [185.0, 372.0, 186.5, 374.5],
    "vix":      [13.2, 13.2, 14.1, 14.1],
})

(
    common_raw.group_by("date")
    .agg((pl.col("vix").n_unique() == 1).alias("is_common"))["is_common"]
    .all()
)  # True when the factor is market-wide
```

## 2. Regular spacing per asset is load-bearing

[`compute_forward_return`](../api/preprocess.md) sorts the input by
`(asset_id, date)` itself, so an unsorted panel is fine, and it rejects
duplicate `(date, asset_id)` rows and non-temporal `date` columns at the
boundary (see the [ingestion contract](../api/data-schema.md#ingestion-contract)).
The horizon is measured on the panel's distinct-date grid: period `i` is
paired with periods `i+1` and `i+1+h` of that grid, never by row offset.

If asset A has every period but asset B is missing two periods in the
middle, asset B contributes rows only where both `i+1` and `i+1+h` exist
for it, and `compute_forward_return` emits `ragged_period_grid` so the
per-asset horizon mismatch is visible. Verify per-asset spacing before
calling if the warning is unexpected:

```python
gaps = raw.sort(["asset_id", "date"]).with_columns(
    (pl.col("date").diff().over("asset_id")).alias("gap")
)
# Inspect gaps.group_by("asset_id").agg(pl.col("gap").n_unique())
# — single unique gap per asset is the goal.
```

If the panel is sparse by design (event series, an irregular period
grid), see step 7 on sparse signals.

## 3. Attach forward return

```python
import factrix as fx
from factrix.preprocess import compute_forward_return

# From here on the examples need a panel long and wide enough to evaluate;
# your own step-1 output takes the generator's place. Its signal column is
# named `factor`, so rename it to the `momentum` this page uses.
raw = fx.datasets.make_cs_panel(n_assets=60, n_dates=260, rng=0).rename(
    {"factor": "momentum"}
)
panel = compute_forward_return(raw, forward_periods=5)
```

The function computes a **per-period normalized** forward return:

```
forward_return[t] = (price[t + 1 + forward_periods] / price[t + 1] - 1) / forward_periods
```

Three things to know about this formula:

- **Entry at `t + 1`, not `t`** — the function assumes you trade on
  the period *after* the signal is observed, preserving a strict
  signal-then-trade causal boundary.
- **Exit at `t + 1 + forward_periods`** — the holding horizon spans
  `forward_periods` periods of the panel's distinct-date grid.
- **Divided by `forward_periods`** — returns are normalized to a per-period basis,
  so `forward_periods=5` and `forward_periods=20` are directly
  comparable. This differs from the cumulative-return convention used
  by qlib (`Ref($close, -N)/$close - 1`) and alphalens.

The horizon counts **periods on the panel's distinct-date grid** — never
row offsets within an asset, and never calendar time. Aligning the panel
onto the intended cadence is the caller's responsibility — see step 5.

The `forward_periods` you pass here must match the
`forward_periods` you later pass to `evaluate`. Bind the custom factor
column(s) via the `factor_cols` parameter:

```python
import factrix as fx
from factrix.metrics import ic

results = fx.evaluate(
    panel,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["momentum"],
    forward_periods=5,
)
```

See [Data schema](../api/data-schema.md) for details on column names.

## 4. Optional factor preprocessing

`evaluate()` does not normalize factor values implicitly. That is deliberate:
factor scale and sign are part of the research hypothesis, especially for beta,
spread, event, and concentration diagnostics. If the analysis needs
cross-sectional clipping or standardization, do it explicitly with the
preprocessing helpers and then pass the processed column through `factor_cols`.
The helpers can run before or after `compute_forward_return`; the example below
processes the signal first, then attaches forward returns.

For a single dense cross-sectional factor:

```python
import factrix as fx
from factrix.metrics import ic
from factrix.preprocess import compute_forward_return
from factrix.preprocess import cross_sectional_zscore, mad_winsorize

raw = mad_winsorize(raw, factor_col="momentum", n_mad=3.0)
raw = cross_sectional_zscore(raw, factor_col="momentum")
panel = compute_forward_return(raw, forward_periods=5)

results = fx.evaluate(
    panel,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["momentum_zscore"],
    forward_periods=5,
)
```

`mad_winsorize` clips the selected factor in place within each date.
`cross_sectional_zscore` appends `<factor_col>_zscore` — `momentum_zscore`
here; it does not overwrite the original column. For multiple candidate
factors, run the helper once per column: each output carries its source
column's name, so nothing has to be renamed between passes.

If the factor should be neutralized against known exposures, first standardize
it, then pass a `(date, asset_id, factor)` frame plus the base exposure columns
to `orthogonalize_factor`. Use the returned residual factor as the column you
evaluate.

```python
import polars as pl
from factrix.preprocess import orthogonalize_factor

# The exposures are yours; these two deterministic stand-ins keep the
# example self-contained on the synthetic panel.
raw = raw.with_columns(
    pl.col("price").log().alias("size"),
    (pl.col("price") / pl.col("price").mean().over("asset_id")).alias("value"),
)

factor_df = raw.select(
    "date",
    "asset_id",
    pl.col("momentum_zscore").alias("factor"),
)
base = raw.select("date", "asset_id", "size", "value")
ortho = orthogonalize_factor(factor_df, base, base_cols=["size", "value"])

raw = raw.join(
    ortho.data.select(
        "date",
        "asset_id",
        pl.col("factor").alias("momentum_ortho"),
    ),
    on=["date", "asset_id"],
)
```

These helpers are usually for dense cross-sectional factor research. Sparse
event flags, macro dummies, and signed event magnitudes should normally keep
their original sign / event semantics unless the research design explicitly
calls for a transformed signal.

## 5. Frequency alignment is the caller's job

factrix never reads the calendar — it counts periods on the panel's own
date grid, not calendar time (the principle is stated once in
[Period grid, not calendar](../development/architecture.md#period-grid-not-calendar)).
Three responsibilities sit upstream of `compute_forward_return`:

- **Same date axis for factor and price source.** If the factor is
  monthly and the price source is daily, downsample (or upsample) one
  side before joining. A frequency mismatch will not raise; it will
  silently mean the wrong thing.
- **Same `forward_periods` interpretation.** The horizon is a count of
  periods on the joined grid, so both sources must already agree on what
  one period is. Pick the horizon against your panel's actual cadence.
- **Slice / regime labels aligned by date.** If you attach a
  `regime_id` or `universe` column for downstream slicing, align it on
  the same date axis the panel uses; mismatched labels propagate
  silently into `by_slice` and screening calls.

### Cross-source alignment: where the boundary sits

When prices and factors come from several sources (several markets, a
vendor feed and an internal signal), the joins are the caller's, and
`compute_forward_return` sees only the result. The boundary is:

- **The horizon is counted on the union grid.** The panel's distinct-date
  grid is the union of every asset's dates, and `forward_periods` is a step
  along that grid. Two markets with different non-trading periods share one
  grid; the horizon is `h` periods of the union.
- **Prices on the common observable grid stay `null` on non-trading
  periods — never forward-fill them.** A filled entry price makes the
  position enter at the *previous* period's price, and a filled exit price
  shortens the true holding period; both fabricate a return that was not
  available. A market closed at the entry or exit period simply has no
  observation to pair, so its assets drop out of that period's
  cross-section rather than stretching the window. Expect
  `ragged_period_grid` from `compute_forward_return` on such a union grid;
  it is the documented consequence, not a defect.
- **The factor is a point-in-time as-of join.** Align each factor
  observation to the latest value known at the period it is evaluated on
  (backward as-of), with a freshness tolerance that drops values older than
  the study allows. That join is the caller's; factrix does not infer
  staleness.
- **The evaluation grid can be coarser than the price grid.** Pass the
  rebalance dates as `compute_forward_return(..., dates=)` rather than
  filtering the panel afterwards, so the horizon stays what it is and the
  overlap inference consumes is derived on the full grid
  ([Evaluating on a coarser grid](../api/preprocess.md#evaluating-on-a-coarser-grid)).
  A grid whose adjacent kept rows are not a constant number of periods
  apart raises `uneven_evaluation_grid`, which names the paths calibrated
  on a constant spacing; pass `dates=` at a constant stride if those paths
  matter, since factrix does not resample
  ([HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns)).

## 6. Missing data

| Source | factrix behaviour | Caller action |
|---|---|---|
| NaN in `factor` | Rewritten to `null` at ingestion, so it is a missing value like any other: it depresses `n_obs` and may trip sample-size guards, and never enters a rank. | Drop or impute before optional factor preprocessing or `compute_forward_return`. |
| NaN / inf in `price` | Rewritten to `null` at ingestion *before* the return is formed, so a `+inf` price is a gap rather than a fabricated `-100 %` return; rows whose `forward_return` cannot be formed are dropped. Tail rows where `i + 1 + forward_periods` runs off the asset's grid are dropped by the same rule. | If the gap is real (suspended trading, a closed market), the drop is correct: the asset has no observation to pair at that period and leaves that period's cross-section. Do not forward-fill a price to close the gap — a filled entry price enters at the previous period's price and a filled exit price shortens the true holding period (see [§5](#cross-source-alignment-where-the-boundary-sits)). Only a genuine feed error should be repaired, from the source. |
| `forward_periods <= 0`, non-`int`, or `bool` | Raises [`UserInputError`](../api/errors.md); the horizon must be a positive integer count of periods. | Pass an explicit period horizon such as `1`, `5`, or `20`. |
| Horizon too long / no finite returns after filtering | Raises [`UserInputError`](../api/errors.md) instead of returning an empty panel. | Shorten the horizon, extend the panel, or clean price values before calling. |
| Single-asset panel (`n_assets == 1`) | `DataStructure` auto-switches to `TIMESERIES`. Dense PANEL metrics (`individual_continuous` and `common_continuous`) raise [`IncompatibleAxisError`](../api/errors.md). | Use `predictive_beta` for dense predictive-regression slope inference, `directional_hit_rate` for sign prediction, or a sparse metric whose cell allows `TIMESERIES`. |
| Effective sample below a metric's own hard floor | Under `strict=True`, raises [`InsufficientSampleError`](../api/errors.md) carrying `.axis` / `.actual` / `.required`; metrics never silently produce a result on under-sampled data. The floor is **per metric and per axis** — there is no single global `T` rule — and it is checked on the *effective* sample (post-stride periods, surviving cross-section), not the raw row count. | Read `.axis` first. `periods` → extend the window. `assets` → widen the universe or lower the metric's bucket count (`monotonicity(n_groups=3)`, `k_spread(k=2)`). Or pass `strict=False` for NaN placeholders. |

## 7. Sparse and event signals

For `(INDIVIDUAL, SPARSE)` or `(COMMON, SPARSE)` factors — buy/sell
flags, FOMC dummies, event magnitudes — the `factor` column is the
`{0, R}` event vector:

- `0` on non-event rows.
- any real value on event rows (`R` is unrestricted — positive,
  negative, or any magnitude). Common forms: `{0, 1}` for a pure
  event flag and `{0, R}` for an event carrying signed or unsigned
  magnitude.
- expect >=50% zeros for automatic sparse routing.

The sparse detector is intentionally zero-value based. `null` means
"missing / unavailable factor value" and is excluded from the
`sparse_ratio` denominator; it is not treated as a non-event. If a
missing upstream value should mean "no event", fill it to `0` before
calling `inspect_data()` or `evaluate()`. If you want to run sparse
event metrics on a continuous exposure, transform the event-of-interest
upstream into an explicit event column, for example:

```python
event_panel = panel.with_columns(
    pl.when(pl.col("momentum_zscore").abs() > 2.0)
    .then(pl.col("momentum_zscore"))
    .otherwise(0.0)
    .alias("momentum_event")
)
```

If a regime label defines the event-of-interest, use the same contract on
the panel carrying that label:

```python title="Illustrative"
regime_event_panel = regime_panel.with_columns(
    pl.when(pl.col("macro_regime") == "stress")
    .then(1.0)
    .otherwise(0.0)
    .alias("regime_event")
)
```

Sort and forward-return attachment are identical to step 2-3; the
dispatch routes sparse signals to event-study procedures (`caar`,
`bmp_z`, `event_ic`, `event_hit_rate`, `profit_factor`, and related
event diagnostics). These remain available on single-asset data when
the metric's cell allows `DataStructure.TIMESERIES`; metrics that need
an asset cross-section still refuse `n_assets == 1`.

When a factor has zero-valued rows intended as non-events but fewer than
50% zeros, `inspect_data` keeps automatic discovery on the dense side
and marks sparse event metrics as degraded rather than unusable. If the
user explicitly requests a sparse metric such as `caar`, `evaluate()`
runs it with a `frequent_event_signal` warning; this is a frequent-event
design, so inspect clustering and event-window overlap before trusting
borderline p-values.

Always-in-market `{-1, +1}` signals are dense directional signals, not
sparse events: there is no non-event zero state. Route them through
`predictive_beta` for single-asset dense slope inference and
`directional_hit_rate` for sign prediction. Low-cardinality dense
signals such as `{-1, +1}` or regime scores remain dense; `inspect_data`
emits an advisory instead of rerouting them to sparse event metrics. See
[Concepts](../getting-started/concepts.md) for the axis contract.

## See also

- [Data schema](../api/data-schema.md) — column-level four-column contract and dtype rules.
- [`adapt`](../api/preprocess.md#factrix.adapt.adapt) — column-name adapter for external long-format panels.
- [`compute_forward_return`](../api/preprocess.md) — symbol reference.
- [Quickstart](../getting-started/quickstart.md) — minimal end-to-end, uses `datasets.make_cs_panel` to skip steps 1-3.
- [Concepts](../getting-started/concepts.md) — three-axis taxonomy (scope / signal / metric / mode).
- [Reading results](reading-results.md) — what `evaluate` returns once the panel is ready.
