---
title: Concepts
---

## What factrix is for

factrix evaluates factor **signal validity**, not portfolio
performance — see
[Guides § Standalone metrics — What factrix evaluates](../guides/standalone-metrics.md#what-factrix-evaluates)
for the full boundary (what is in, what is downstream).

Once the axes below are familiar, the
[API reference landing](../api/index.md) maps a research question to the
function that answers it.

## Three orthogonal axes

`AnalysisConfig` is defined by three user-facing
axes. They are **orthogonal**: each supported combination maps to one
specific statistical test.

| Axis | Values | What it asks |
|------|--------|--------------|
| `scope` | `INDIVIDUAL` / `COMMON` | Does each asset have its own factor value, or do all assets share one? |
| `signal` | `CONTINUOUS` / `SPARSE` | Real-valued signal, or `{0, R}` event trigger (zero on non-events; `R` is any real magnitude — positive, negative, or unsigned)? |
| `metric` | `IC` / `FM` / *(N/A)* | Only for `(INDIVIDUAL, CONTINUOUS)` — refines the research question |

### scope — a factor attribute, not a data shape

factrix input is **panel data** — both a time axis (dates) and a cross-section
axis (assets). `scope` describes how the factor varies along the cross-section
axis, not the panel layout itself:

- **`INDIVIDUAL`** — each `(date, asset_id)` has its own factor value (P/E,
  momentum, quality).
- **`COMMON`** — every `asset_id` on a given date shares one factor value
  (VIX, DXY, FOMC dummy). Quick check in Polars:
  `df.group_by("date").agg(pl.col("factor").n_unique() == 1).all()`.

### signal

- **`CONTINUOUS`** — real-valued (z-score, percentile, momentum, …).
- **`SPARSE`** — `{0, R}` event trigger: zero on non-event entries,
  any real value otherwise (`R` is unrestricted — positive, negative,
  or any magnitude); expect ≥ 50% zeros. Common forms: `{0, 1}` for a
  pure event flag and `{0, R}` for an event carrying signed or
  unsigned magnitude.

### metric (only for `INDIVIDUAL × CONTINUOUS`)

- **`IC`** (default) — rank-based predictive ordering: Spearman ρ → Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t-test.
- **`FM`** — unit-of-exposure premium: Fama-MacBeth λ → NW HAC t-test.

Choose by research question, not data shape:

| | `IC` | `FM` |
|--|------|------|
| Question | Does the factor predict rank ordering of returns? | What premium does each unit of factor exposure earn? |
| Method | Rank-based, outlier-robust | Slope-based, economic interpretation |

## Axis dispatch

How the three axes above pick a factory:

```
                          your panel
                              │
            ┌─────────────────┴─────────────────┐
   factor varies per asset?          shared by all assets?
            │                                   │
       INDIVIDUAL                            COMMON
            │                                   │
    ┌───────┴───────┐                   ┌───────┴───────┐
 continuous     sparse                continuous     sparse
    │              │                       │              │
  IC / FM   individual_sparse     common_continuous   common_sparse
```

The table below pairs each factory with the procedure it runs and the
literature behind it.

## Five analysis scenarios

| Factory | Run by `evaluate()` | Procedure | Literature |
|---------|---------------------|-----------|------------|
| `individual_continuous(metric=IC)` | [`ic`][factrix.metrics.ic.ic] | per-date Spearman ρ → NW HAC t on E[information coefficient (IC)] | [Grinold 1989][grinold-1989]; [Newey-West 1987][newey-west-1987] |
| `individual_continuous(metric=FM)` | [`fm_beta`][factrix.metrics.fm_beta.fm_beta] | per-date ordinary least squares (OLS) λₜ → NW HAC t on E[λ] | [Fama-MacBeth 1973][fama-macbeth-1973] |
| `individual_sparse()` | [`caar`][factrix.metrics.caar.caar] | per-event AR → CAAR → cross-event t | [Brown-Warner 1985][brown-warner-1985] |
| `common_continuous()` | [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | per-asset TS β → cross-asset t on E[β] | [Black-Jensen-Scholes 1972][black-jensen-scholes-1972] |
| `common_sparse()` | [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | per-asset TS β on dummy → cross-asset t | TS-β + event-study hybrid |

## PanelMode: PANEL vs TIMESERIES

`PanelMode` is **not user-facing** — it is derived at evaluate-time from
`N = panel["asset_id"].n_unique()`. Both modes return a real `primary_p`;
neither is degraded.

For the field-order walk of `FactorProfile.primary_p` and its companion
fields, see [Reading results](../guides/reading-results.md).

For the full sample-guard contract (PANEL/TIMESERIES tiers, behaviour
per factory at each `n_assets` regime, special N = 1 paths) see
[Guides § Panel vs timeseries](../guides/panel-timeseries.md).
For the consolidated `MIN_*` threshold values, see
[Reference § Sample-size constants](../reference/metric-applicability.md#sample-size-constants).
