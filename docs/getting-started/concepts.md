# Concepts

## Three orthogonal axes

`AnalysisConfig` is defined by three user-facing axes. They are **orthogonal**: each supported combination maps to one specific statistical test.

| Axis | Values | 問的是 |
|------|--------|--------|
| `scope` | `INDIVIDUAL` / `COMMON` | 因子值是每個 asset 自己一套還是所有 asset 共用同一個？ |
| `signal` | `CONTINUOUS` / `SPARSE` | 訊號是連續實數還是 `{−1, 0, +1}` 觸發？ |
| `metric` | `IC` / `FM` / *(N/A)* | 只在 `(INDIVIDUAL, CONTINUOUS)` 情境細分研究問題 |

### scope — factor attribute, not data structure

factrix input is **panel data** — both a time axis (dates) and a cross-section axis (assets). `scope` describes the factor's shape along the cross-section axis:

- **`INDIVIDUAL`**: each `(date, asset_id)` has its own factor value (P/E, momentum, quality).
- **`COMMON`**: every `asset_id` on a date shares one factor value (VIX, DXY, FOMC dummy). Polars one-liner to check: `df.group_by("date").agg(pl.col("factor").n_unique() == 1).all()`.

### signal

- **`CONTINUOUS`**: real-valued (z-score, percentile, price momentum)
- **`SPARSE`**: `{−1, 0, +1}` trigger with ≥ 50% zeros

### metric (only for `INDIVIDUAL × CONTINUOUS`)

- **`IC`** (default): rank-based predictive ordering — Spearman ρ → NW HAC t-test
- **`FM`**: unit-of-exposure premium — Fama-MacBeth λ → NW HAC t-test

Choose based on the research question, not data shape:

| | IC | FM |
|--|----|----|
| 問的是 | Factor 對未來報酬有沒有 predictive ordering？ | Factor 每多一單位 exposure 對應多少報酬溢酬？ |
| Method | Rank-based, outlier-robust | Slope-based, economic interpretation |

## Decision tree

```
                        你手上的 panel
                              │
          ┌───────────────────┴───────────────────┐
 同 date 下每 asset 因子值不同？         所有 asset 共用同一個值？
          │                                       │
      INDIVIDUAL                               COMMON
          │                                       │
  ┌───────┴───────┐                       ┌───────┴───────┐
連續實數     {-1,0,+1}               連續實數       {-1,0,+1}
  │              │                       │              │
IC 或 FM    individual_sparse      common_continuous  common_sparse
```

## Five analysis scenarios

| Factory | Procedure | Literature |
|---------|-----------|------------|
| `individual_continuous(metric=IC)` | per-date Spearman ρ → NW HAC t on E[IC] | Grinold (1989); Newey & West (1987) |
| `individual_continuous(metric=FM)` | per-date OLS λ_t → NW HAC t on E[λ] | Fama & MacBeth (1973) |
| `individual_sparse()` | per-event AR → CAAR → cross-event t | Brown & Warner (1985) |
| `common_continuous()` | per-asset TS β → cross-asset t on E[β] | Black-Jensen-Scholes (1972) |
| `common_sparse()` | per-asset TS β on dummy → cross-asset t | TS-β + event-study hybrid |

## Mode: PANEL vs TIMESERIES

`Mode` is **not user-facing** — it is derived at evaluate-time from `N = panel["asset_id"].n_unique()`:

| Mode | Condition | Statistics |
|------|-----------|------------|
| `PANEL` | N ≥ 2 | cross-sectional / cross-asset aggregation |
| `TIMESERIES` | N = 1 | time-series aggregation (NW HAC) |

Both modes produce real `primary_p` — neither is degraded.

Special N=1 paths:

- `(INDIVIDUAL, CONTINUOUS, *) × N=1`: raises `ModeAxisError` with `suggested_fix=common_continuous(...)` — mathematically undefined (no cross-sectional dispersion).
- `(*, SPARSE, *) × N=1`: `individual_sparse()` and `common_sparse()` both route to the same timeseries dummy procedure. Profile carries `InfoCode.SCOPE_AXIS_COLLAPSED` for auditability.
