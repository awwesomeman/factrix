# Quickstart

> **`forward_periods` is rows, not calendar time.** factrix is frequency-agnostic — it only shifts row indices. `forward_periods=5` on a daily panel = 5 trading days; on a weekly panel = 5 weeks. The caller is responsible for ensuring the panel is sorted per asset and has regular time spacing.

## 30-second smoke test

```python
import factrix as fl
from factrix.preprocess.returns import compute_forward_return

raw   = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

# Path A — let factrix infer the scenario from data shape
result  = fl.suggest_config(panel)
profile = fl.evaluate(panel, result.suggested)

# Path B — supply the config directly (type-safe, IDE validates illegal combos)
cfg     = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)
profile = fl.evaluate(panel, cfg)

print(profile.verdict(), '| primary_p =', round(profile.primary_p, 4))
# → pass | primary_p = 0.0

print(profile.diagnose())
# {'mode': 'panel', 'n_obs': 494, 'n_assets': 100,
#  'primary_p': 2.13e-40, 'warnings': [], 'info_notes': [],
#  'stats': {'ic_mean': 0.0722, 'ic_t_nw': 14.60, ...}}
```

Path A suits first-time exploration — `result.reasoning` explains why each axis was inferred, `result.warnings` lists potential risks.

## Research question → factory lookup

| 你想問的問題 | Factory |
|---|---|
| 我的 per-asset 因子（P/E、momentum）能否預測 cross-section 排序？ | `individual_continuous(metric=fl.Metric.IC)` |
| 我的 per-asset 因子每多一單位 exposure 對應多少報酬溢酬？ | `individual_continuous(metric=fl.Metric.FM)` |
| 個股事件（earnings / rating / 併購公告）有沒有 abnormal return？ | `individual_sparse()` |
| Macro 因子（VIX / DXY / 利率）對 cross-section 有沒有 systematic exposure？ | `common_continuous()` |
| Macro 事件（FOMC / index rebalance / 政策公布）有沒有市場效應？ | `common_sparse()` |

> **N=1（single asset / series）**: mode auto-switches to TIMESERIES. Macro and sparse rows work as-is; `individual_continuous` at N=1 raises `ModeAxisError` with a `suggested_fix` pointing to `common_continuous()`.

## `profile.diagnose()` and WarningCode

`diagnose()` returns a flat dict for human reading or AI agent triage:

```python
{
    "mode": "panel",
    "n_obs": 500,
    "primary_p": 0.0001,
    "warnings": ["unreliable_se_short_periods"],
    "info_notes": [],
    "stats": {"ic_mean": 0.082, "ic_t_nw": 4.21, "nw_lags_used": 5},
}
```

| WarningCode | Trigger |
|---|---|
| `UNRELIABLE_SE_SHORT_PERIODS` | `n_periods < 30` — NW HAC SE unstable |
| `PERSISTENT_REGRESSOR` | factor ADF p > 0.10 |
| `EVENT_WINDOW_OVERLAP` | event windows overlap (CAAR / sparse) |
| `SERIAL_CORRELATION_DETECTED` | Ljung-Box p < 0.05 on residuals |

`warnings` does **not** affect `verdict()`. It is a risk flag — user decides whether to filter before BHY. `verdict()` reads only `primary_p < threshold`.
