# factorlib

Modular factor evaluation toolkit — independent, composable factor analysis modules covering cross-sectional, event signal, macro panel, and macro common factor types.

**定位：Factor Signal Analyzer**，不是回測引擎（Zipline / Backtrader）。Turnover / Breakeven / Net Spread 是理想化的 proxy（等權、無滑價），不代表真實可交易收益。

## Quick Start

```python
import factorlib as fl

# 一行完成：preprocess + evaluate
result = fl.quick_check(df, "Mom_20D")
print(result)
```

```
Factor: Mom_20D | Status: PASS

Gates:
  significance: PASS (via IC, Q1-Q5_spread)
  oos_persistence: PASS

┌─────────────────┬──────────┬──────────┬───────┐
│ metric          │    value │     stat │   sig │
├─────────────────┼──────────┼──────────┼───────┤
│ ic              │   0.0512 │     3.42 │   *** │
│ ic_ir           │   0.2850 │          │       │
│ q1_q5_spread    │   0.0034 │     2.91 │    ** │
│ turnover        │   0.4200 │          │       │
│ ...             │          │          │       │
└─────────────────┴──────────┴──────────┴───────┘
```

## Usage Scenarios

### Level 0 — Quick Screening

```python
import factorlib as fl

result = fl.quick_check(factor_df, "Mom_20D")

# 切換因子類型
result = fl.quick_check(macro_df, "CPI_spread", factor_type="macro_panel")
result = fl.quick_check(event_df, "EarningsSurprise", factor_type="event_signal")
```

### Level 1 — Full Pipeline

```python
config = fl.CrossSectionalConfig(forward_periods=5, n_groups=10)
prepared = fl.preprocess(factor_df, config=config)
result = fl.evaluate(prepared, "Mom_20D", config=config)

# Event signal
config = fl.EventConfig(forward_periods=5, event_window_post=20)
prepared = fl.preprocess(event_df, config=config)
result = fl.evaluate(prepared, "GoldenCross", config=config)
```

### Level 2 — Individual Metrics

```python
from factorlib.metrics import compute_ic, ic, ic_ir, quantile_spread

ic_series = compute_ic(prepared)
print(ic(ic_series, forward_periods=5))
print(quantile_spread(prepared, forward_periods=5, n_groups=10))

# Event signal metrics
from factorlib.metrics import compute_caar, caar, bmp_test, corrado_rank_test

caar_series = compute_caar(event_prepared)
print(caar(caar_series, forward_periods=5))
print(bmp_test(event_prepared, forward_periods=5))
print(corrado_rank_test(event_prepared))  # standalone non-parametric test
```

### Level 3 — Batch Evaluate + Compare

```python
results = fl.batch_evaluate(
    {"Mom_20D": df1, "Value": df2, "Size": df3},
    factor_type="cross_sectional",
)
table = fl.compare(results, sort_by="ic")
```

### Level 4 — Charts

```python
from factorlib.charts import report_charts

figs = report_charts(result)  # requires: pip install factorlib[charts]
```

### Level 5 — MLflow Tracking

```python
from factorlib.integrations.mlflow import FactorTracker  # requires: pip install factorlib[mlflow]

tracker = FactorTracker("Factor_Zoo")
results = fl.batch_evaluate(factors, on_result=tracker.log_evaluation)
```

## Factor Types

```python
fl.describe_factor_types()
```

| Type | Example | Signal | Core Question |
|------|---------|--------|---------------|
| `cross_sectional` | momentum, value, size | 連續值，每期每資產 | 排序能預測截面報酬差異嗎？ |
| `event_signal` | 營收發佈、黃金交叉 | 離散觸發 {-1,0,+1} | 事件後報酬有異常嗎？ |
| `macro_panel` | 各國 CPI、利差 | 連續值，小截面 N<30 | 宏觀指標能預測跨國配置嗎？ |
| `macro_common` | VIX、黃金、USD index | 單一時序，全資產共用 | 資產對共同因子的 exposure 穩定嗎？ |

### Per-Type Default Profile

```python
fl.describe_profile("cross_sectional")
fl.describe_profile("event_signal")
fl.describe_profile("macro_panel")
fl.describe_profile("macro_common")
```

---

### Metric Reference

#### Type-Specific Metrics

**cross_sectional (ic.py, quantile.py, ...)**

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| `ic` | Per-date Spearman rank corr(factor, return), non-overlapping t-test | Mean IC > 0 = factor ranks predict return ranks. `***` = significant |
| `ic_ir` | mean(IC) / std(IC) | IC stability. > 0.3 is strong |
| `monotonicity` | Spearman rho across quantile group mean returns | 1.0 = perfect Q1>Q2>...>Q5 ordering. Non-monotonic = unstable |
| `q1_q5_spread` | mean(Q1 return - Q5 return), t-test | Long-short spread in per-period units |
| `q1_concentration` | Effective N / Total N in top quantile (HHI-based) | < 0.5 = alpha driven by few stocks |

**event_signal (caar.py, event_quality.py, event_horizon.py, mfe_mae.py, ...)**

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| `caar` | mean(signed_car), non-overlapping t-test. signed_car = return x sign(factor) | Mean abnormal return per event. `***` = events have real directional effect |
| `bmp_sar` | Standardize each event's AR by pre-event vol (σ_i), then z-test on SARs | Same hypothesis as CAAR, but robust to event-induced variance inflation |
| `event_hit_rate` | fraction(signed_car > 0), binomial z-test H₀: p=0.5 | Direction correct rate. > 55% with significance = useful signal |
| `event_ic` | Spearman corr(\|factor\|, signed_car) among event rows only | Signal strength → return. > 0 = stronger signal works better. Auto-skips for discrete {±1} signals |
| `profit_factor` | sum(positive signed_car) / sum(negative signed_car) | Gross P&L ratio. > 1 = gains exceed losses across events |
| `event_skewness` | Fisher skewness of signed_car, D'Agostino test | Positive = occasional big wins, frequent small losses (desirable) |
| `mfe_mae` | MFE p50 / \|MAE p75\| from per-event price path | Path quality. > 2 = favorable excursion dominates adverse. Requires `price` column |
| `event_around_return` | Per-offset return profile at T-6..T+24 relative to event | value = mean \|pre-event return\|. High = potential information leakage. Requires `price` column |
| `multi_horizon_hit_rate` | Win rate at horizons [1, 6, 12, 24] bars | Shows optimal holding period. Increasing = slow alpha (needs patience) |
| `signal_density` | mean(total_bars / n_events) per asset | Signal frequency. High = selective signal (sparse). Low = fires often |
| `clustering_hhi` | HHI = sum(event_share_per_date²) | Event date concentration. High = independence assumption violated. N>1 only |

**macro_panel (fama_macbeth.py)**

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| `fm_beta` | Per-date cross-sectional OLS: return ~ factor → β series, Newey-West t-test | Factor premium across countries/assets. `***` = significant premium |
| `pooled_beta` | Single pooled OLS on all observations, clustered SE | Robustness check. FM and Pooled should agree on sign |
| `beta_sign_consistency` | fraction(per-date β > 0) or (< 0) | > 0.6 = factor direction is stable across dates |

**macro_common (ts_beta.py)**

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| `ts_beta` | Per-asset time-series OLS: return ~ factor → β per asset, cross-asset t-test | Mean exposure to common factor. `***` = assets respond consistently |
| `mean_r_squared` | mean(per-asset R²) | < 0.01 = common factor explains very little variation |
| `ts_beta_sign_consistency` | fraction(per-asset β > 0) or (< 0) | > 0.6 = assets agree on exposure direction |

#### Shared Metrics (cross-type)

These metrics are reused across multiple factor types. Each type feeds a
different intermediate series, but the statistical computation is identical.

**`oos_decay`** (oos.py) — used by: CS, ES, MP, MC

Multi-split IS/OOS persistence test. Splits the value series chronologically
at 60/40, 70/30, 80/20, computes `decay = |mean_OOS| / |mean_IS|`.

| Factor type | Input series | What it measures |
|-------------|-------------|------------------|
| cross_sectional | IC series (date, ic → value) | IC persistence out of sample |
| event_signal | CAAR series (date, caar → value) | Event effect persistence |
| macro_panel | FM β series (date, beta → value) | Factor premium persistence |
| macro_common | Rolling mean β (date, value) | Exposure stability over time |

Interpretation: decay ≥ 0.5 = acceptable (McLean & Pontiff 2016 average ~0.68).
Sign flip in any split → VETOED (signal reversed direction OOS).

**`trend`** (trend.py, shown as `ic_trend` / `caar_trend` / `beta_trend`) — used by: CS, ES, MP, MC

Theil-Sen robust slope on the value series. Detects systematic decay or
improvement over time.

| Factor type | Profile name | Input series |
|-------------|-------------|--------------|
| cross_sectional | `ic_trend` | IC series |
| event_signal | `caar_trend` | CAAR series |
| macro_panel | `beta_trend` | FM β series |
| macro_common | `beta_trend` | Rolling mean β |

Interpretation: slope < 0 with CI excluding zero → significant decay → CAUTION.

**`hit_rate`** (hit_rate.py) — used by: CS, ES

| Factor type | Input | What "hit" means |
|-------------|-------|-------------------|
| cross_sectional | IC series | IC > 0 on that date (factor ranked correctly) |
| event_signal | signed_car per event | Event return in predicted direction |

Note: CS `hit_rate` uses non-overlapping sampling; ES `event_hit_rate`
does not (events are already sparse).

**`quantile_spread`** (quantile.py) — used by: CS, MP

| Factor type | n_groups | Interpretation |
|-------------|----------|----------------|
| cross_sectional | 10 (default) | Fine-grained sort, large N supports it |
| macro_panel | 3 (default) | Coarse sort, small N can't support 10 groups |

**`turnover` / `breakeven_cost` / `net_spread`** (tradability.py) — used by: CS, MP

Same calculation: fraction of portfolio that changes per rebalance.
Not applicable to event_signal (no portfolio) or macro_common (no rebalance).

---

## Architecture

```
factorlib/
├── __init__.py              # Top-level exports
├── _api.py                  # quick_check, compare, batch_evaluate
├── _types.py                # MetricOutput, FactorType(StrEnum), constants
├── _stats.py                # t-stat, p-value, Newey-West SE, BHY threshold
├── _ols.py                  # Shared OLS helpers
├── config.py                # BaseConfig → CrossSectionalConfig / EventConfig / ...
├── adapt.py                 # Column name mapping
├── validation.py            # Per-type pandera schema
│
├── preprocess/              # Raw data → evaluation-ready format
│   ├── pipeline.py          # preprocess() dispatcher + per-type orchestrators
│   ├── returns.py           # Forward return (t+1 entry), winsorize, abnormal return
│   ├── normalize.py         # MAD winsorize, z-score
│   └── orthogonalize.py     # Factor orthogonalization
│
├── evaluation/              # Gate pipeline → profile → caution
│   ├── pipeline.py          # evaluate(), build_artifacts()
│   ├── profile.py           # compute_profile() per-type dispatch
│   ├── presets.py           # Default gate lists per type
│   ├── _protocol.py         # Artifacts, FactorProfile, EvaluationResult
│   ├── _caution.py          # CAUTION condition checks
│   └── gates/               # Gate functions
│       ├── significance.py      # CS: IC or spread t-stat
│       ├── event_significance.py # event_signal: CAAR/BMP/hit_rate
│       ├── fm_significance.py   # macro_panel: FM β or Pooled β
│       ├── ts_significance.py   # macro_common: mean TS β
│       └── oos_persistence.py   # Shared: OOS decay + sign flip
│
├── metrics/                 # Independent, composable metric tools
│   ├── ic.py                # IC, IC_IR, regime_ic, multi_horizon_ic
│   ├── caar.py              # CAAR significance tests, BMP standardized AR
│   ├── event_quality.py     # Per-event descriptive: hit rate, IC, profit factor, skewness, density
│   ├── event_horizon.py     # Multi-horizon: event-around return profile, horizon hit rate
│   ├── mfe_mae.py           # MFE/MAE path excursion
│   ├── clustering.py        # Event clustering HHI diagnostic
│   ├── corrado.py           # Corrado (1989) nonparametric rank test
│   ├── quantile.py          # Quantile spread, group returns
│   ├── monotonicity.py      # Spearman monotonicity
│   ├── concentration.py     # Q1 HHI concentration
│   ├── hit_rate.py          # IC direction hit rate
│   ├── trend.py             # Theil-Sen IC/β trend
│   ├── oos.py               # Multi-split OOS decay
│   ├── spanning.py          # Spanning alpha, forward selection
│   ├── tradability.py       # Turnover, breakeven cost, net spread
│   ├── fama_macbeth.py      # FM β, Pooled OLS, sign consistency
│   └── ts_beta.py           # Per-asset TS β, R², rolling β
│
├── factors/                 # Factor generators
├── charts/                  # Plotly charts (optional: factorlib[charts])
└── integrations/            # MLflow, Streamlit (optional deps)
```

### Dependency Flow (one-way, no cycles)

```
_types, _stats, _ols        ← zero dependencies
       ↑
config                       ← depends on _types
       ↑
metrics/                     ← depends on _types, _stats
preprocess/                  ← depends on _types, _ols
       ↑
evaluation/                  ← depends on _types, _stats, metrics, config
       ↑
_api                         ← depends on evaluation, metrics, preprocess, config
       ↑
__init__                     ← re-exports

integrations/, charts/       ← depends on evaluation (optional, not depended on)
```

## Configuration

```python
# Cross-sectional (default)
fl.CrossSectionalConfig(
    forward_periods=5,     # Forward return horizon
    n_groups=10,           # Quantile groups
    q_top=0.2,             # Q1 fraction for concentration
    orthogonalize=False,   # Factor orthogonalization applied?
    mad_n=3.0,             # MAD winsorization
    estimated_cost_bps=30, # Trading cost estimate
)

# Event signal
fl.EventConfig(
    forward_periods=5,       # Return horizon for CAAR
    event_window_post=20,    # MFE/MAE bar window after event
    cluster_window=3,        # Clustering detection window
    adjust_clustering="none", # "none" | "kolari_pynnonen"
)

# Macro panel
fl.MacroPanelConfig(
    forward_periods=5,
    n_groups=3,              # Fewer groups for small N
    demean_cross_section=False,
    min_cross_section=10,
)

# Macro common
fl.MacroCommonConfig(
    forward_periods=5,
    ts_window=60,    # Rolling window for TS beta
    tradable=False,  # Is the common factor tradable?
)
```

## Gate Pipeline

Each factor type has default gates that run sequentially. First failure short-circuits.

| Type | Gate 1 | Gate 2 |
|------|--------|--------|
| cross_sectional | IC or spread t-stat >= 2.0 | OOS decay >= 0.5, no sign flip |
| event_signal | CAAR or BMP or hit_rate stat >= 2.0 | OOS decay (on CAAR series) |
| macro_panel | FM beta or Pooled beta t-stat >= 2.0 | OOS decay (on beta series) |
| macro_common | Mean TS beta t-stat >= 2.0 | OOS decay (on rolling beta) |

Status: `PASS` → `CAUTION` (passed with warnings) → `FAILED` → `VETOED`

Custom gates:

```python
from functools import partial
from factorlib.evaluation.gates.significance import significance_gate

strict = partial(significance_gate, threshold=3.0)
result = fl.evaluate(df, "X", config=config, gates=[strict])
```

## Statistical Safeguards

- **t+1 entry:** Forward return uses `price[t+1+N] / price[t+1]` — entry at next bar after signal, enforcing causal boundary
- **Non-overlapping sampling:** IC, CAAR, and spread t-tests use every N-th date to avoid autocorrelation from overlapping forward returns
- **BMP standardized AR test:** Normalizes event abnormal returns by pre-event volatility, robust to event-induced variance inflation
- **Newey-West SE:** FM beta inference uses HAC standard errors (Bartlett kernel) for time-series autocorrelation
- **BHY correction:** `bhy_threshold()` for multiple testing across factor zoo
- **Clustering HHI:** Detects event concentration on few dates (violates CAAR independence assumption)
- **N-awareness:** Profile and caution logic adapt to single-asset (N=1) vs multi-asset panels
- **Small N warning:** UserWarning when median cross-section < 30 (suggest MacroPanelConfig)
- **Per-group warning:** UserWarning when quantile groups have < 5 assets
- **IS/OOS split:** Time-based only, no random split
- **MAD z-score:** Robust to outliers (median + MAD x 1.4826)
- **EPSILON guard:** 1e-9 division protection across all metrics
- **ddof=1:** Sample std consistency (Polars default)

## Installation

```bash
pip install factorlib              # Core (polars only)
pip install factorlib[charts]      # + plotly
pip install factorlib[mlflow]      # + mlflow tracking
pip install factorlib[all]         # Everything
```

## Metrics Organization

### File = One Statistical Question

Each `metrics/` file answers one question. Don't group by factor type.

```
ic.py           — "rank correlation between signal and return?"
caar.py         — "event abnormal return significantly different from zero?"
event_quality.py — "per-event quality: hit rate, strength, P&L shape?"
mfe_mae.py      — "what does the price path look like after events?"
fama_macbeth.py — "cross-sectional regression coefficient significant?"
ts_beta.py      — "time-series exposure stable across assets?"
quantile.py     — "top/bottom groups have different returns?"
tradability.py  — "trading cost eats the alpha?"
```

### Cross-Type Sharing

Some metrics are shared across multiple factor types:

| Metric | CS | ES | MP | MC |
|--------|:--:|:--:|:--:|:--:|
| `oos_decay` (oos.py) | x | x | x | x |
| `trend` (trend.py) | x | x | x | x |
| `hit_rate` (hit_rate.py) | x | x | | |
| `quantile_spread` (quantile.py) | x | | x | |
| `turnover` (tradability.py) | x | | x | |

Shared metrics operate on a generic `(date, value)` series — the profile
renames type-specific intermediates (ic_values, caar_values, beta_values)
to this schema before passing to shared functions.

### Adding a New Metric

```
1. Create metrics/<question>.py with function(s) returning MetricOutput
2. metrics/__init__.py: add re-export
3. _api.py: add to _PROFILE_METRICS or _STANDALONE_METRICS
4. evaluation/profile.py: call from the relevant _xxx_profile() function
5. tests/: add test file
```

Principles:
- One file per statistical question (not per factor type)
- If a metric can serve multiple factor types, keep it in its own file
  and call it from each profile function (like oos.py, trend.py)
- Return `MetricOutput` — never raw floats
- Use `MIN_EVENTS` / `MIN_IC_PERIODS` guards for small samples
- Profile auto-skips metrics that are mathematically undefined for the
  current data (e.g., `event_ic` when signal has no magnitude variance,
  `clustering_hhi` when N=1)

## Adding a New Factor Type

```
1.  _types.py: add FactorType StrEnum value
2.  config.py: add XxxConfig(BaseConfig) with factor_type ClassVar
3.  __init__.py: export new Config
4.  _api.py: update FACTOR_TYPES, _DESCRIPTIONS, _PROFILE_METRICS, _STANDALONE_METRICS
5.  validation.py: add schema to _SCHEMAS
6.  preprocess/pipeline.py: add case branch in preprocess()
7.  evaluation/pipeline.py: add case branch in build_artifacts()
8.  evaluation/profile.py: add case branch in compute_profile()
9.  evaluation/_caution.py: add case branch in check_caution()
10. evaluation/presets.py: add gate list + update _DEFAULT_GATES
11. metrics/: add metric module(s) — one per statistical question
12. metrics/__init__.py: add re-exports
13. tests/: add tests
```
