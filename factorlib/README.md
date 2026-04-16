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

**cross_sectional:**
ic, ic_ir, hit_rate, ic_trend, monotonicity, oos_decay,
q1_q5_spread, turnover, breakeven_cost, net_spread, q1_concentration

**event_signal:**
caar, bmp_sar, event_hit_rate, oos_decay, caar_trend,
profit_factor, event_skewness, mfe_mae, clustering_hhi

**macro_panel:**
fm_beta, pooled_beta, beta_sign_consistency, oos_decay, beta_trend,
q1_q5_spread, turnover, breakeven_cost, net_spread

**macro_common:**
ts_beta, mean_r_squared, ts_beta_sign_consistency, oos_decay, beta_trend

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
│   ├── event_quality.py     # Per-event descriptive: hit rate, IC, profit factor, skewness
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
11. metrics/: add metric module
12. metrics/__init__.py: add re-exports
13. tests/: add tests
```
