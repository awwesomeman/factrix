# by_regime

!!! warning "Deprecated since v0.10.0"
    `by_regime` is superseded by the axis-agnostic
    [`by_slice`](by-slice.md). Importing or calling `by_regime` emits
    a `DeprecationWarning`. Behaviour is unchanged for the deprecation
    window (at least one minor) — see [Migration](#migration) below.

Cross-cutting research dispatcher. Slices any metric's date-keyed
input by regime label and runs the metric per slice. Returns
`dict[regime_label, MetricOutput]`.

## Call shape

```python
import polars as pl
from factrix.metrics import by_regime, ic, compute_ic

ic_df = compute_ic(panel)
labels = pl.DataFrame(
    {"date": dates, "regime": ["bull", "bull", "bear", "bear"]}
)

per_regime = by_regime(ic, ic_df, regime_labels=labels)
# {"bull": MetricOutput(name="ic", ...), "bear": MetricOutput(name="ic", ...)}
```

The first argument is the **metric callable** (e.g. `ic`, `caar`,
`fama_macbeth`); the second is the metric's primary date-keyed
DataFrame; remaining keyword args (`forward_periods=...`, etc.)
forward unchanged on every per-regime call.

## Input contract

`by_regime`'s second argument is the metric's **primary date-keyed
input** — typically the output of a `compute_*` step, not the raw
panel. Each metric family has its own prep:

| Metric family | Prep step | Pass to `by_regime` |
|---|---|---|
| `ic`, `ic_ir` | `ic_df = compute_ic(panel)` | `ic_df` |
| `caar` | `caar_df = compute_caar(panel)` | `caar_df` |
| `fama_macbeth` | `beta_df = compute_fm_betas(panel)` | `beta_df` |

Common families shown — for the full eligible set, use
[`list_metrics`](list-metrics.md) with
`format="json"` and filter on `input_kind == "panel"`.

If `regime_labels` is omitted, falls back to time-bisection labelled
`first_half` / `second_half` and emits a `UserWarning` — that path is
a structural-break sanity check, not a regime test.

## What it does **not** do

`by_regime` performs **no cross-regime statistical inference**. It
returns the per-regime outputs and stops. A generic second-layer test
(BHY adjustment, min-|t|, Sharpe-diff Wald, etc.) cannot be applied
honestly across the metric matrix — the appropriate test depends on
the metric family:

| Metric family | Cross-regime test |
|---------------|-------------------|
| IC / Fama-MacBeth λ (mean-zero, t-like) | BHY across regimes + min-\|t\| |
| Sharpe-like ratios | Memmel (2003) / Ledoit-Wolf |
| CAAR | pooled vs per-regime clustering reconciliation |
| Turnover, hit_rate, monotonicity ρ | no canonical cross-regime test |

For curated families that have a defensible second-layer test,
factrix provides curated wrappers — see
[`regime_ic`](metrics/ic.md#factrix.metrics.ic.regime_ic) for the IC family. Use
`by_regime` directly when no curated wrapper exists or when you want
raw per-regime outputs to compose your own analysis.

**As of v0.9.0, `regime_ic` is the only curated wrapper.** `regime_caar`
and `regime_fama_macbeth` are tracked in
[#107 Phase 2](https://github.com/awwesomeman/factrix/issues/107).

> Earlier docs called the dispatcher "Layer A" and curated wrappers
> "Layer B"; renamed in
> [#157](https://github.com/awwesomeman/factrix/issues/157). Roles
> unchanged.

## Which metrics work

Any metric whose primary first argument is a DataFrame with a `date`
column. Discover the eligible set via
[`list_metrics`](list-metrics.md) and filter on `input_kind == "panel"`
— scalar-input utilities like `breakeven_cost` / `net_spread` consume
pre-aggregated scalars and have no date column to slice on:

```python
import factrix as fx
from factrix.metrics import by_regime, caar, fama_macbeth, compute_caar, compute_fm_betas

candidates = [
    r["name"]
    for r in fx.list_metrics(
        fx.FactorScope.INDIVIDUAL, fx.Signal.SPARSE, format="json",
    )
    if r["input_kind"] == "panel"
]
# ['caar', 'event_ic', ...]

caar_df = compute_caar(panel)              # raw panel → per-date CAAR series
beta_df = compute_fm_betas(panel)          # raw panel → per-date β series

per_regime = by_regime(caar, caar_df, regime_labels=labels)
per_regime = by_regime(fama_macbeth, beta_df, regime_labels=labels)
```

`by_regime` does not police metric-vs-input compatibility — that is
the metric's own job. Each per-regime call inherits the metric's
existing guards (sample-size short-circuits, `IncompatibleAxisError`,
warnings); you see the same signals you would see calling the metric
directly on each slice.

## Migration

Replace every `by_regime` call with `by_slice` plus an explicit join
(or pre-existing column).

```python
# Before (deprecated since v0.10.0)
from factrix.metrics import by_regime
results = by_regime(ic, ic_df, regime_labels=regime_df)

# After
from factrix.metrics import by_slice
results = by_slice(
    ic,
    ic_df.join(regime_df, on="date", how="inner"),  # match by_regime's inner-join
    label="regime",
)
```

The time-bisection fallback (`regime_labels=None` → `first_half` /
`second_half`) has **no `by_slice` equivalent**. That path is a
structural-break sanity check, not a regime test, and the deprecation
is the right moment to make that explicit. If you still want the
median-date split, compose the label yourself:

```python
import polars as pl

mid = df.get_column("date").quantile(0.5)
df = df.with_columns(
    pl.when(pl.col("date") < mid).then(pl.lit("first_half"))
      .otherwise(pl.lit("second_half"))
      .alias("regime")
)
by_slice(ic, df, label="regime")
```

This is more honest than the old fallback: the caller picks the split
point and accepts that this is structural-break diagnostics, not
regime analysis. Genuine regime work needs a domain-driven
classification (NBER recession dating, volatility-state classifier,
etc.), which neither `by_regime` nor `by_slice` ships.

Curated wrappers (`regime_ic`, `regime_caar`, ...) are unchanged —
they remain the right entry point when you need a metric-family-aware
cross-regime test.

## API reference

::: factrix.metrics.regime
    options:
      members:
        - by_regime
