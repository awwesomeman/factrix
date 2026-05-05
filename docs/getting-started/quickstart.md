# Quickstart

> **`forward_periods` counts rows, not calendar time.** factrix is
> frequency-agnostic — it only shifts row indices. `forward_periods=5` on a
> daily panel means 5 trading days; on a weekly panel, 5 weeks. The caller is
> responsible for ensuring the panel is sorted per asset and has regular time
> spacing.

## 30-second smoke test

```python
import factrix as fl
from factrix.preprocess.returns import compute_forward_return

raw   = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

# Path B — supply the config directly (type-safe; the IDE rejects illegal combos)
cfg     = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)
profile = fl.evaluate(panel, cfg)

print(profile.verdict(), '| primary_p =', round(profile.primary_p, 4))
# → pass | primary_p = 0.0

print(profile.diagnose())
# {'mode': 'panel', 'n_obs': 494, 'n_assets': 100,
#  'primary_p': 2.13e-40, 'warnings': [], 'info_notes': [],
#  'stats': {'ic_mean': 0.0722, 'ic_t_nw': 14.60, ...}}
```

If you are not sure which factory to use, let factrix infer it from the
panel shape:

```python
# Path A — inferred config + reasoning trace
result  = fl.suggest_config(panel)
profile = fl.evaluate(panel, result.suggested)
# result.reasoning explains how each axis was inferred
# result.warnings flags potential risks (small N, persistence, …)
```

See [Concepts](concepts.md) for what each axis means.

## Research question → factory

The five supported research questions and their factory calls live in
[Concepts § Five analysis scenarios](concepts.md#five-analysis-scenarios)
— that page is also the SSOT for the procedure and literature behind each
factory. For task-oriented help on **picking** the right factory (IC vs FM,
when to add standalone metrics), see [Choosing a metric](../guides/choosing-metric.md).

> **N = 1 (single asset / series):** `Mode` auto-switches to `TIMESERIES`. The
> macro and sparse factories work as-is. `individual_continuous` at N=1
> raises `ModeAxisError` with `suggested_fix=common_continuous(...)`.

## `profile.diagnose()` and warnings

`diagnose()` returns a flat dict for human inspection or AI agent triage:

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

The most common warnings:

| `WarningCode` | Trigger |
|---------------|---------|
| `UNRELIABLE_SE_SHORT_PERIODS` | `n_periods < 30` — NW HAC SE unstable |
| `PERSISTENT_REGRESSOR` | factor ADF p > 0.10 |
| `EVENT_WINDOW_OVERLAP` | event windows overlap (CAAR / sparse) |
| `SERIAL_CORRELATION_DETECTED` | Ljung-Box p < 0.05 on residuals |

`warnings` does **not** affect `verdict()` — it is a risk flag. The user
decides whether to filter on warnings before BHY. `verdict()` reads only
`primary_p < threshold`.
