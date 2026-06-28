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

```python
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
the one-liner from [Concepts](../getting-started/concepts.md)
(swap the column name for whichever the panel carries):

```python
raw.group_by("date").agg(pl.col("vix").n_unique() == 1).all()
```

## 2. Regular spacing per asset is load-bearing

[`compute_forward_return`](../api/preprocess.md) sorts the input by
`(asset_id, date)` itself, so an unsorted panel is fine. What it
**does not** inspect is the calendar gap between successive rows —
the function shifts by row count, not by date.

If asset A has daily rows but asset B is missing two trading days in
the middle, asset B's row-shift skips the gap silently and the forward
return on the row before the gap measures the wrong horizon. Verify
per-asset spacing before calling:

```python
gaps = raw.sort(["asset_id", "date"]).with_columns(
    (pl.col("date").diff().over("asset_id")).alias("gap")
)
# Inspect gaps.group_by("asset_id").agg(pl.col("gap").n_unique())
# — single unique gap per asset is the goal.
```

If the panel is sparse by design (event series, irregular trading
days), see step 7 on sparse signals.

## 3. Attach forward return

```python
from factrix.preprocess import compute_forward_return

panel = compute_forward_return(raw, forward_periods=5)
```

The function computes a **per-period normalized** forward return:

```
forward_return[t] = (price[t + 1 + N] / price[t + 1] - 1) / N
```

Three things to know about this formula:

- **Entry at `t + 1`, not `t`** — the function assumes you trade on
  the bar *after* the signal is observed, preserving a strict
  signal-then-trade causal boundary.
- **Exit at `t + 1 + N`** — the holding horizon spans `N` rows of the
  asset's own date series, where `N = forward_periods`.
- **Divided by `N`** — returns are normalized to a per-period basis,
  so `forward_periods=5` and `forward_periods=20` are directly
  comparable. This differs from the cumulative-return convention used
  by qlib (`Ref($close, -N)/$close - 1`) and alphalens.

The horizon counts **rows of the asset's own date series**, not
calendar days. `forward_periods=5` on a daily panel is a
five-trading-day lookahead; on a monthly panel it is five months.
Frequency is the user's responsibility — see step 5.

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
    factor_cols=["factor_zscore"],
    forward_periods=5,
)
```

`mad_winsorize` clips the selected factor in place within each date.
`cross_sectional_zscore` appends `factor_zscore`; it does not overwrite the
original column. For multiple candidate factors, run the helper per column and
rename `factor_zscore` to a factor-specific name before processing the next
one.

If the factor should be neutralized against known exposures, first standardize
it, then pass a `(date, asset_id, factor)` frame plus the base exposure columns
to `orthogonalize_factor`. Use the returned residual factor as the column you
evaluate.

```python
import polars as pl
from factrix.preprocess import orthogonalize_factor

factor_df = raw.select(
    "date",
    "asset_id",
    pl.col("factor_zscore").alias("factor"),
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

factrix is calendar-agnostic — it shifts rows, not calendar time.
Three responsibilities sit upstream of `compute_forward_return`:

- **Same date axis for factor and price source.** If the factor is
  monthly and the price source is daily, downsample (or upsample) one
  side before joining. A frequency mismatch will not raise; it will
  silently mean the wrong thing.
- **Same `forward_periods` interpretation.** Five rows on a daily panel
  is one week of trading days; five rows on a monthly panel is five
  months. Pick the horizon against your panel's actual cadence.
- **Slice / regime labels aligned by date.** If you attach a
  `regime_id` or `universe` column for downstream slicing, align it on
  the same date axis the panel uses; mismatched labels propagate
  silently into `by_slice` and screening calls.

## 6. Missing data

| Source | factrix behaviour | Caller action |
|---|---|---|
| NaN in `factor` | Not auto-imputed; flows through to the procedure, where it depresses `n_obs` and may trip sample-size guards. | Drop or impute before optional factor preprocessing or `compute_forward_return`. |
| NaN / inf in `price` | `compute_forward_return` drops rows whose computed `forward_return` is not finite (`null`, `NaN`, `+inf`, or `-inf`). Tail rows where `t + 1 + N` runs off the end of the series are dropped by the same filter. | If a daily NaN reflects a true gap (suspended trading, holiday), the drop is correct. If imputable (forward-fill from previous close), impute before calling. |
| `forward_periods <= 0`, non-`int`, or `bool` | Raises [`UserInputError`](../api/errors.md); the horizon must be a positive integer row count. | Pass an explicit row horizon such as `1`, `5`, or `20`. |
| Horizon too long / no finite returns after filtering | Raises [`UserInputError`](../api/errors.md) instead of returning an empty panel. | Shorten the horizon, extend the panel, or clean price values before calling. |
| Single-asset panel (N = 1) | `DataStructure` auto-switches to `TIMESERIES`. Dense PANEL metrics (`individual_continuous` and `common_continuous`) raise [`IncompatibleAxisError`](../api/errors.md). | Use a sparse metric whose cell allows `TIMESERIES`, or a scope-agnostic series diagnostic. |
| T < `MIN_PERIODS_HARD` (= 20) periods | Raises [`InsufficientSampleError`](../api/errors.md); procedures never silently produce a result on under-sample data. | Extend the window or accept the procedure's refusal. |

## 7. Sparse and event signals

For `(INDIVIDUAL, SPARSE)` or `(COMMON, SPARSE)` factors — buy/sell
flags, FOMC dummies, event magnitudes — the `factor` column is the
`{0, R}` event vector:

- `0` on non-event rows.
- any real value on event rows (`R` is unrestricted — positive,
  negative, or any magnitude). Common forms: `{0, 1}` for a pure
  event flag and `{0, R}` for an event carrying signed or unsigned
  magnitude.
- expect ≥ 50% zeros.

Sort and forward-return attachment are identical to step 2-3; the
dispatch routes sparse signals to event-study procedures (`caar`,
`ts_beta` on dummies). See [Concepts](../getting-started/concepts.md)
for the contract.

## See also

- [Data schema](../api/data-schema.md) — column-level four-column contract and dtype rules.
- [`adapt`](../api/preprocess.md#factrix.adapt.adapt) — column-name adapter for external long-format panels.
- [`compute_forward_return`](../api/preprocess.md) — symbol reference.
- [Quickstart](../getting-started/quickstart.md) — minimal end-to-end, uses `datasets.make_cs_panel` to skip steps 1-3.
- [Concepts](../getting-started/concepts.md) — three-axis taxonomy (scope / signal / metric / mode).
- [Reading results](reading-results.md) — what `evaluate` returns once the panel is ready.
