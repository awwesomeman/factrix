---
title: Quickstart
---

!!! warning "`forward_periods` counts rows, not calendar time"
    factrix is frequency-agnostic — it only shifts row indices.
    `forward_periods=5` on a daily panel means 5 trading days; on a
    weekly panel, 5 weeks. The caller is responsible for ensuring the
    panel is sorted per asset and has regular time spacing.

## 30-second smoke test

```python
import factrix as fx
from factrix.preprocess import compute_forward_return

raw   = fx.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

# example pending v0.14.0 docs rewrite
cfg     = ...  # config construction pending v0.14.0 docs rewrite
profile = fx.evaluate(panel, cfg)

print('primary_p =', round(profile.primary_p, 4))
# → pass | primary_p = 0.0

print(profile.diagnose())
# {'identity': {'factor_id': 'factor', 'forward_periods': 5},
#  'context': {},
#  'cell': {'scope': 'individual', 'signal': 'continuous',
#           'metric': 'ic', 'mode': 'panel'},
#  'n_obs': 494, 'n_pairs': 49400, 'n_periods': 494, 'n_assets': 100,
#  'primary_p': 2.13e-40, 'primary_stat': 14.60, 'primary_stat_name': 't_nw',
#  'warnings': [], 'info_notes': [],
#  'stats': {'mean': 0.0722, 't_nw': 14.60, 'p_nw': 2.13e-40},
#  'metadata': {'t_nw': {'nw_lags': 5}, 'p_nw': {'nw_lags': 5}}}
```

See [Concepts](concepts.md) for what each axis means.

---

## Research question → factory

The five supported research questions and their factory calls — the
`AnalysisConfig.*()` classmethod constructors such as
`individual_continuous()` used above — live in
[Concepts § Five analysis scenarios](concepts.md#five-analysis-scenarios).
That page is also the SSOT for the procedure and literature behind each
factory. For task-oriented help on **picking** the right factory (information coefficient (IC) vs FM,
when to add standalone metrics), see [Choosing a metric](../guides/choosing-metric.md).

!!! note "N = 1 (single asset / series)"
    `DataStructure` auto-switches to `TIMESERIES`. The `common_continuous` and
    `*_sparse` metric families work as-is. `individual_continuous` at N=1 raises
    `IncompatibleAxisError` — use a `common_*` or `*_sparse` metric instead.

---

## `profile.diagnose()` and warnings

`diagnose()` returns a flat dict for human inspection or AI agent triage:

```python
{
    "identity": {"factor_id": "momentum", "forward_periods": 5},
    "context": {},
    "cell": {"scope": "individual", "signal": "continuous",
             "metric": "ic", "mode": "panel"},
    "n_obs": 500, "n_pairs": 50000, "n_periods": 500, "n_assets": 100,
    "primary_p": 0.0001,
    "primary_stat": 4.21,
    "primary_stat_name": "t_nw",
    "warnings": ["unreliable_se_short_periods"],
    "info_notes": [],
    "stats": {"mean": 0.082, "t_nw": 4.21, "p_nw": 0.0001},
    "metadata": {"t_nw": {"nw_lags": 5}, "p_nw": {"nw_lags": 5}},
}
```

The most common warnings:

- `UNRELIABLE_SE_SHORT_PERIODS` — `20 ≤ T < 30`; Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) SE unstable.
  `T < 20` raises `InsufficientSampleError`.
- `PERSISTENT_REGRESSOR` — factor augmented Dickey-Fuller (ADF) *p* > 0.10.
- `EVENT_WINDOW_OVERLAP` — event windows overlap on the same asset.
- `SERIAL_CORRELATION_DETECTED` — Ljung-Box *p* < 0.05 on residuals.

For the full enum and the trigger conditions for each `WarningCode`,
`InfoCode`, and `StatCode`, see
[Reference § Warning / info / stat codes](../reference/warning-codes.md).
For exception classes (`InsufficientSampleError`, `IncompatibleAxisError`,
`UserInputError`, ...) and their catch patterns, see
[Errors](../api/errors.md).

`warnings` does **not** affect `primary_p` —
it is a risk flag. The user decides whether to filter on warnings before
Benjamini-Hochberg-Yekutieli (BHY). For single-factor pre-registered analysis compare `primary_p` against your nominal threshold directly.

For the full field-order walk of `FactorProfile` and `Survivors`
(after `bhy`), see [Reading results](../guides/reading-results.md).

---

## Next steps

You have one `FactorProfile` for one factor. The common follow-ups:

| You want to… | Reach for | Guide |
|---|---|---|
| Screen N candidate factors with false discovery rate (FDR) control | [`multi_factor.bhy(profiles)`](../api/multi-factor.md) — or `partial_conjunction` / `bhy_hierarchical` for nested structure | [Batch screening with BHY](../guides/batch-screening.md) |
| Rank factors after screening | [`compare(survivors)`](../api/compare.md) — leaderboard with `adj_q` | — |
| Explore one metric across slices (sector / regime / universe / ADV bucket) | [`by_slice`](../api/by-slice.md) → `SliceResult.to_frame()` | [Slice analysis](../guides/slice-analysis.md) |
| Test whether slices differ statistically | [`slice_pairwise_test`](../api/slice-test.md) / [`slice_joint_test`](../api/slice-test.md) | [Slice analysis](../guides/slice-analysis.md) |

For function semantics, the input contract, and cross-function topics
(`expand_over` semantics, regime-analysis dispatch), see the
[API reference landing](../api/index.md) and
[Cross-function reference](../api/decision-tree.md). For the field-order
walk of `FactorProfile` / `Survivors`, see
[Reading results](../guides/reading-results.md).
