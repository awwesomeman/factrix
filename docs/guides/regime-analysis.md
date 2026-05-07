# Regime Analysis

Regime analysis asks "is this factor stable across market environments?" — a different question from out-of-sample decay (overfitting) and from cross-asset robustness (specification). factrix splits regime analysis into two layers because **slicing by regime** and **testing significance across regimes** are different jobs that need different APIs.

## The two layers

| Layer | Function | What it does | What it does not do |
|---|---|---|---|
| A — slicing | [`by_regime(metric, df, *, regime_labels, **kwargs)`](../api/by-regime.md) | Slices `df` by regime label, calls `metric` per slice, returns `dict[regime, MetricOutput]` | **No cross-regime statistical test** |
| B — bespoke inference | `regime_<metric>` (e.g. [`regime_ic`](../api/metrics/ic.md#factrix.metrics.ic.regime_ic)) | Layer A + a metric-appropriate cross-regime test (BHY, min-\|t\|, etc.) | Only exists for curated metrics |

**Use Layer A when:** you want raw per-regime numbers, or you want to compose your own cross-regime test, or no Layer B wrapper exists for the metric you care about.

**Use Layer B when:** a curated wrapper exists for your metric and the bundled second-layer test matches your research question.

## Why no generic cross-regime test

A single dispatcher carrying a single second-layer test would silently over-claim. The appropriate test depends on the metric family:

- **IC, Fama-MacBeth λ** — mean-zero t-tests on per-regime samples → BHY across regimes is defensible.
- **Sharpe** — variance-stabilised difference (Memmel 2003 / Ledoit-Wolf) is needed; BHY on raw t is wrong.
- **CAAR** — per-regime clustering interacts with pooled-vs-split SE choice; needs a bespoke reconciliation.
- **Turnover, hit_rate, monotonicity ρ** — no canonical cross-regime test; differences are descriptive.

Bundling BHY into a generic dispatcher would apply it to all of these, including the cases where it is wrong. We chose explicit Layer B wrappers per metric family instead.

## Constructing regime labels

`regime_labels` is a `(date, regime)` DataFrame. Common constructions:

```python
import polars as pl

# 1. External classifier (e.g. NBER recession, vol regime, factor return sign)
vol_labels = pl.DataFrame({"date": dates, "regime": ["bull", "bull", "bear", ...]})

# 2. From a market-vol panel — see lookahead warning below
vix_thresh = vix["vix"].quantile(0.7)
vol_labels = vix.with_columns(
    pl.when(pl.col("vix") > vix_thresh).then(pl.lit("high_vol")).otherwise(pl.lit("low_vol")).alias("regime")
).select("date", "regime")
```

!!! warning "Lookahead bias when constructing labels"
    The `vix.quantile(0.7)` snippet above uses **full-sample** statistics
    to label every date — that leaks future information into every
    per-regime IC. For decision-grade analysis, derive thresholds from
    an expanding or rolling window so each date's label only depends on
    information available at that date. The full-sample form is fine
    only for descriptive ex-post analysis.

If `regime_labels=None`, factrix applies a time-bisection fallback
(`first_half` / `second_half`) and emits a `UserWarning`. This is a
structural-break check, **not** a regime test — domain-driven labels
are required for any defensible regime claim.

## Discovering eligible metrics

Layer A only accepts metrics whose primary input is a date-keyed
DataFrame. Use [`list_metrics`](../api/list-metrics.md) with
`format="json"` and filter on `input_kind == "panel"` to enumerate the
candidate set for your `(scope, signal)` cell:

```python
import factrix as fl

panel_metrics = [
    r["name"]
    for r in fl.list_metrics(
        fl.FactorScope.INDIVIDUAL, fl.Signal.CONTINUOUS, format="json",
    )
    if r["input_kind"] == "panel"
]
```

Scalar-input utilities (`breakeven_cost`, `net_spread`) are excluded —
they consume pre-aggregated scalars and have no date column to slice
on.

## Worked example: IC across volatility regimes

```python
from factrix.metrics import by_regime, compute_ic, ic, regime_ic

ic_df = compute_ic(panel, factor_col="value", return_col="forward_return")

# Layer A — raw per-regime IC summaries
per_regime = by_regime(ic, ic_df, regime_labels=vol_labels)
for label, out in per_regime.items():
    print(label, out.value, out.stat)

# Layer B — IC-specific cross-regime second layer (BHY + min-|t|)
result = regime_ic(ic_df, regime_labels=vol_labels)
result.metadata["per_regime"]              # raw + BHY-adjusted p per regime
result.metadata["p_value_bhy_adjusted"]    # worst adjusted p (aggregate)
result.metadata["direction_consistent"]    # sign agreement across regimes
```

## Related

- [`by_regime`](../api/by-regime.md) — Layer A surface and registered metrics.
- [`regime_ic`](../api/metrics/ic.md#factrix.metrics.ic.regime_ic) — Layer B wrapper for IC.
- [Batch screening with BHY](batch-screening.md) — the same BHY family-correction principle, applied to factor candidates rather than regimes.
