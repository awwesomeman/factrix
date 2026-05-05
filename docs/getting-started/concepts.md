# Concepts

## Three orthogonal axes

[`AnalysisConfig`][factrix.AnalysisConfig] is defined by three user-facing
axes. They are **orthogonal**: each supported combination maps to one
specific statistical test.

| Axis | Values | What it asks |
|------|--------|--------------|
| `scope` | `INDIVIDUAL` / `COMMON` | Does each asset have its own factor value, or do all assets share one? |
| `signal` | `CONTINUOUS` / `SPARSE` | Real-valued signal, or `{−1, 0, +1}` trigger? |
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
- **`SPARSE`** — `{−1, 0, +1}` trigger; expect ≥ 50% zeros.

### metric (only for `INDIVIDUAL × CONTINUOUS`)

- **`IC`** (default) — rank-based predictive ordering: Spearman ρ → NW HAC t-test.
- **`FM`** — unit-of-exposure premium: Fama-MacBeth λ → NW HAC t-test.

Choose by research question, not data shape:

| | `IC` | `FM` |
|--|------|------|
| Question | Does the factor predict rank ordering of returns? | What premium does each unit of factor exposure earn? |
| Method | Rank-based, outlier-robust | Slope-based, economic interpretation |

## Decision tree

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

## Five analysis scenarios

| Factory | Procedure | Literature |
|---------|-----------|------------|
| `individual_continuous(metric=IC)` | per-date Spearman ρ → NW HAC t on E[IC] | Grinold (1989); Newey & West (1987) |
| `individual_continuous(metric=FM)` | per-date OLS λₜ → NW HAC t on E[λ] | Fama & MacBeth (1973) |
| `individual_sparse()` | per-event AR → CAAR → cross-event t | Brown & Warner (1985) |
| `common_continuous()` | per-asset TS β → cross-asset t on E[β] | Black-Jensen-Scholes (1972) |
| `common_sparse()` | per-asset TS β on dummy → cross-asset t | TS-β + event-study hybrid |

## Mode: PANEL vs TIMESERIES

`Mode` is **not user-facing** — it is derived at evaluate-time from
`N = panel["asset_id"].n_unique()`. Both modes return a real `primary_p`;
neither is degraded.

For the full sample-guard contract (PANEL/TIMESERIES tiers, behaviour
per factory at each `n_assets` regime, special N = 1 paths) see
[Guides § PANEL vs TIMESERIES](../guides/panel-timeseries.md).
