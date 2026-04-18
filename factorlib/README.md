# factorlib

Modular factor evaluation toolkit — independent, composable factor analysis modules covering cross-sectional, event signal, macro panel, and macro common factor types.

**定位：Factor Signal Analyzer**，不是回測引擎（Zipline / Backtrader）。Turnover / Breakeven / Net Spread 是理想化的 proxy（等權、無滑價），不代表真實可交易收益。

## Quick Start

```python
import factorlib as fl

profile = fl.evaluate(df, "Mom_20D", factor_type="cross_sectional")

print(profile.verdict())        # 'PASS' or 'FAILED' (canonical p vs threshold)
print(profile.canonical_p)      # the one p-value verdict is based on
print(profile.ic_mean, profile.ic_tstat, profile.ic_ir, profile.q1_q5_spread)

for d in profile.diagnose():
    print(d.severity, d.code, d.message)
```

`fl.evaluate` returns a typed, frozen dataclass (``CrossSectionalProfile`` /
``EventProfile`` / ``MacroPanelProfile`` / ``MacroCommonProfile``) — not a
dict. Fields are IDE-discoverable and feed polars expressions directly.

## Usage Scenarios

### Level 0 — Single-factor verdict

```python
import factorlib as fl

profile = fl.evaluate(factor_df, "Mom_20D", factor_type="cross_sectional")
profile = fl.evaluate(macro_df, "CPI_spread", factor_type="macro_panel")
profile = fl.evaluate(event_df, "EarningsSurprise", factor_type="event_signal")
```

### Level 1 — Full control

```python
config = fl.CrossSectionalConfig(forward_periods=5, n_groups=10)
profile = fl.evaluate(factor_df, "Mom_20D", config=config)

# Event signal
config = fl.EventConfig(forward_periods=5, event_window_post=20)
profile = fl.evaluate(event_df, "GoldenCross", config=config)

# Reuse existing preprocessed data
prepared = fl.preprocess(factor_df, config=config)
profile = fl.evaluate(prepared, "Mom_20D", config=config, preprocess=False)
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

### Level 3 — Batch + multiple-testing (BHY) + ranking

```python
import polars as pl
import factorlib as fl

profiles = fl.evaluate_batch(
    {"Mom_20D": df1, "Value": df2, "Size": df3},
    factor_type="cross_sectional",
)

top = (
    profiles
    .multiple_testing_correct(p_source="canonical_p", fdr=0.05)
    .filter(pl.col("bhy_significant"))
    .rank_by("ic_ir")
    .top(10)
)
print(top.to_polars())
```

``ProfileSet`` is polars-native: ``filter`` accepts ``pl.Expr`` or a
``Callable[[Profile], bool]``, ``rank_by`` sorts by any dataclass field,
``to_polars()`` hands back a DataFrame for joins / export. See
`tests/test_profile_set.py` for the full API surface.

### Level 4 — Redundancy matrix

```python
# Both methods require per-factor Artifacts — evaluate_batch does NOT retain
# them, so build them yourself in a loop. See evaluate_batch docstring note.
from factorlib.evaluation.pipeline import build_artifacts

arts = {}
for name, fdf in factors.items():
    prep = fl.preprocess(fdf, config=fl.CrossSectionalConfig())
    a = build_artifacts(prep, fl.CrossSectionalConfig())
    a.factor_name = name
    arts[name] = a

redund = fl.redundancy_matrix(profiles, method="value_series", artifacts=arts)
```

### Level 5 — Charts

```python
from factorlib.evaluation.pipeline import build_artifacts
from factorlib.charts import report_charts

artifacts = build_artifacts(prepared, config)       # keeps prepared + intermediates
figs = report_charts(artifacts)                      # requires: pip install factorlib[charts]
```

### Level 6 — MLflow Tracking

```python
from factorlib.integrations.mlflow import FactorTracker

tracker = FactorTracker("Factor_Zoo")
fl.evaluate_batch(
    factors,
    factor_type="cross_sectional",
    on_result=lambda name, p: tracker.log_profile(p, factor_type="cross_sectional"),
)
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

**`oos_survival_ratio`** (oos.py) — used by: CS, ES, MP, MC (field name `oos_survival_ratio` on every profile)

Multi-split IS/OOS persistence test. Splits the value series chronologically
at 60/40, 70/30, 80/20 and computes `survival = |mean_OOS| / |mean_IS|`.

| Factor type | Input series | What it measures |
|-------------|-------------|------------------|
| cross_sectional | IC series (date, ic → value) | IC persistence out of sample |
| event_signal | CAAR series (date, caar → value) | Event effect persistence |
| macro_panel | FM β series (date, beta → value) | Factor premium persistence |
| macro_common | Rolling mean β (date, value) | Exposure stability over time |

Interpretation: survival ≥ 0.5 = acceptable (McLean & Pontiff 2016 average
~0.68); values below fire a `*.oos_survival_low` diagnose rule.
`oos_sign_flipped` (bool) fires a separate `veto`-severity diagnose.

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
├── _api.py                  # evaluate, evaluate_batch, redundancy_matrix, describe_profile
├── _types.py                # PValue, Verdict, Diagnostic, MetricOutput, FactorType(StrEnum), constants
├── _stats.py                # t-stat, p-value, Newey-West SE helpers
├── stats/
│   └── multiple_testing.py  # bhy_adjust, bhy_adjusted_p (p-value-based)
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
├── evaluation/              # Profile-era pipeline
│   ├── pipeline.py          # build_artifacts() per-type dispatch
│   ├── _protocol.py         # Artifacts (+ _CompactedPrepared sentinel)
│   ├── profile_set.py       # ProfileSet[P] polars-native collection + BHY
│   ├── profiles/            # Typed per-type Profile dataclasses
│   │   ├── _base.py         # FactorProfile Protocol + @register_profile
│   │   ├── cross_sectional.py
│   │   ├── event.py
│   │   ├── macro_panel.py
│   │   └── macro_common.py
│   └── diagnostics/
│       └── _rules.py        # Per-type Rule list feeding profile.diagnose()
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
│   ├── ts_beta.py           # Per-asset TS β, R², rolling β
│   └── redundancy.py        # Pairwise |ρ| redundancy matrix (used by fl.redundancy_matrix)
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
    forward_periods=5,                # Forward return horizon
    n_groups=10,                      # Quantile groups
    q_top=0.2,                        # Q1 fraction for concentration
    orthogonalize=False,              # Factor orthogonalization applied?
    mad_n=3.0,                        # MAD winsorization
    return_clip_pct=(0.01, 0.99),     # Forward-return percentile winsorize
    estimated_cost_bps=30,            # Trading cost estimate
    multi_horizon_periods=[1, 5, 10, 20],  # Horizons for multi_horizon_ic
)

# Event signal
fl.EventConfig(
    forward_periods=5,                # Return horizon for CAAR
    event_window_pre=5,               # Pre-event bar window (MFE/MAE, leakage)
    event_window_post=20,             # Post-event bar window (MFE/MAE)
    cluster_window=3,                 # Clustering detection window
    adjust_clustering="none",         # "none" | "calendar_block_bootstrap" | "kolari_pynnonen"
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

## Verdict + Diagnose (binary + contextual)

Each profile exposes a single canonical p-value (``CANONICAL_P_FIELD``) that
drives ``verdict()``. All "significant-but-with-caveats" nuance is
reported as structured diagnostics instead of re-entering the verdict.

| Type | Canonical p | Statistical origin |
|------|-------------|--------------------|
| cross_sectional | `ic_p` | IC non-overlapping t-test |
| event_signal | `caar_p` | CAAR non-overlapping t-test (MacKinlay 1997) |
| macro_panel | `fm_beta_p` | Fama-MacBeth λ Newey-West t-test |
| macro_common | `ts_beta_p` | Cross-sectional t-test on per-asset TS β |

`verdict(threshold=2.0)` → `'PASS' | 'FAILED'`; the threshold is in
t-stat units and translated through the same t-distribution that
generated the canonical p (df = n_periods − 1), so the verdict is
conservative at small n.

Batch-wide family-wise control uses BHY (Benjamini-Yekutieli step-up):

```python
profiles.multiple_testing_correct(p_source="canonical_p", fdr=0.05)
```

``p_source`` is whitelisted against ``Profile.P_VALUE_FIELDS`` — composed
p-values (e.g. `min(ic_p, spread_p)`) are rejected because feeding BHY a
mix of hypotheses violates same-test-family semantics.

Customizing verdict threshold:

```python
profile.verdict(threshold=3.0)  # Harvey-Liu-Zhu (2016) strict bar
```

## Statistical Safeguards

- **t+1 entry:** Forward return uses `price[t+1+N] / price[t+1]` — entry at next bar after signal, enforcing causal boundary
- **Non-overlapping sampling:** IC, CAAR, and spread t-tests use every N-th date to avoid autocorrelation from overlapping forward returns
- **BMP standardized AR test:** Normalizes event abnormal returns by pre-event volatility, robust to event-induced variance inflation
- **Newey-West SE:** FM beta inference uses HAC standard errors (Bartlett kernel) for time-series autocorrelation
- **BHY correction:** `ProfileSet.multiple_testing_correct(...)` applies the Benjamini-Yekutieli step-up on canonical p-values across the batch
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
| `oos_survival_ratio` (oos.py) | x | x | x | x |
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
3. Pick the Profile dataclass(es) that should expose the value and add
   typed fields. Populate them in from_artifacts() by calling the new metric.
4. If the metric produces a p-value that should be BHY-eligible, add the
   field name to P_VALUE_FIELDS on the profile (same-test-family only).
5. If the metric warrants a warning, add a Rule to diagnostics/_rules.py.
6. tests/: add test file
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
4.  _api.py: update FACTOR_TYPES and _DESCRIPTIONS
5.  validation.py: add schema to _SCHEMAS
6.  preprocess/pipeline.py: add case branch in preprocess()
7.  evaluation/pipeline.py: add case branch in build_artifacts()
8.  evaluation/profiles/<new_type>.py: add a @register_profile(FactorType.X)
    @dataclass(frozen=True, slots=True) with typed fields, CANONICAL_P_FIELD,
    P_VALUE_FIELDS, and from_artifacts classmethod
9.  evaluation/profiles/__init__.py: import to trigger registration
10. evaluation/diagnostics/_rules.py: add per-type Rule list + isinstance branch
11. metrics/: add metric module(s) — one per statistical question
12. metrics/__init__.py: add re-exports
13. tests/profiles/: add per-type profile test file
14. tests/: add integration tests
```
