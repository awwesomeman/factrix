---
title: factrix.evaluate
---

::: factrix.evaluate
    options:
      show_root_heading: true
      show_root_full_path: true
      show_root_toc_entry: true
      heading_level: 1
      separate_signature: true
      show_signature_annotations: true

## Use cases

- **Single-factor significance.** One panel + one [`AnalysisConfig`][factrix.AnalysisConfig]
  → one [`FactorProfile`][factrix.FactorProfile] carrying `primary_p` and the
  cell-specific statistics.
- **Batch screening with FDR.** Loop `evaluate` over candidate signal columns
  and feed the resulting list of profiles to
  [`bhy`][factrix.multi_factor.bhy] for false-discovery-rate control —
  see [Batch screening](../guides/batch-screening.md).
- **Cross-cell apples-to-apples.** Swap the `AnalysisConfig` factory to
  compare e.g. IC rank-ordering against Fama-MacBeth λ on the same panel,
  or individual-asset factors against broadcast macro factors. Return
  shape is identical across cells.
- **TIMESERIES auto-routing.** `Common × Continuous` with `N == 1` falls
  back to single-series OLS with Newey-West HAC SE, so single-asset macro
  factors flow through the same entry point without a parallel code path.

## Worked example — single-factor smoke test

Full runnable example complementing the doctest snippets in **Examples**
above with realistic console output and a `diagnose()` dump.

```python
import factrix as fx
from factrix.preprocess import compute_forward_return

raw   = fx.datasets.make_cs_panel(
    n_assets=100, n_dates=500, ic_target=0.08, seed=2024,
)
panel = compute_forward_return(raw, forward_periods=5)

cfg     = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.IC, forward_periods=5,
)
profile = fx.evaluate(panel, cfg)

print("primary_p =", round(profile.primary_p, 4))
# → primary_p = 0.0

print(profile.diagnose())
# {'identity': {'factor_id': 'factor', 'forward_periods': 5},
#  'context': {},
#  'cell':     {'scope': 'individual', 'signal': 'continuous',
#               'metric': 'ic', 'mode': 'panel'},
#  'n_obs':    494, 'n_pairs': 49400, 'n_periods': 494, 'n_assets': 100,
#  'primary_p':     2.13e-40,
#  'primary_stat':  14.60,
#  'primary_stat_name': 't_nw',
#  'warnings': [], 'info_notes': [],
#  'stats':    {'mean': 0.0722, 't_nw': 14.60, 'p_nw': 2.13e-40},
#  'metadata': {'t_nw': {'nw_lags': 5}, 'p_nw': {'nw_lags': 5}}}
```

## Config recipes — one per dispatch cell

Minimum-viable `AnalysisConfig` for each of the four cells. The
`evaluate(panel, cfg)` call site is identical; only `cfg` changes.

**Individual × Continuous — rank predictive ordering (IC)**

```python
cfg = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.IC, forward_periods=5,
)
```

**Individual × Continuous — unit-of-exposure premium (Fama-MacBeth λ)**

```python
cfg = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.FM, forward_periods=5,
)
```

**Individual × Sparse — event study (`factor ∈ {-1, 0, +1}` triggers)**

```python
cfg = fx.AnalysisConfig.individual_sparse(forward_periods=5)
# Attach a `price` column on the panel to also get
# event_around_return / mfe_mae_summary in the profile.
```

**Common × Continuous — broadcast macro factor (e.g. VIX)**

```python
cfg = fx.AnalysisConfig.common_continuous(forward_periods=5)
# With N == 1 on the panel, evaluate auto-routes to single-series
# OLS with NW HAC SE (`profile.mode == "TIMESERIES"`).
```

**Common × Sparse — broadcast event dummy (FOMC, index rebalance)**

```python
cfg = fx.AnalysisConfig.common_sparse(forward_periods=5)
```

Per-cell required / optional columns and the PANEL ↔ TIMESERIES Mode
derivation are documented in the **Dispatch lore** admonition above.

## Next steps

[Batch screening](../guides/batch-screening.md) wires `evaluate` into a
multi-factor FDR pipeline and covers:

- looping `evaluate` over candidate signal columns while preserving the
  `identity` / `context` tuple downstream aggregators need;
- choosing between [`bhy`][factrix.multi_factor.bhy] /
  [`partial_conjunction`][factrix.multi_factor.partial_conjunction] /
  [`bhy_hierarchical`][factrix.multi_factor.bhy_hierarchical];
- handling mixed-cell batches (per-asset factors plus macro factors in
  one run);
- the role of `primary_p` vs the individual entries in `stats` at the
  FDR stage.

New to the input contract? Start with [Panel schema](panel-schema.md).

## See also

- [Panel schema](panel-schema.md) — input contract (`date, asset_id, factor, forward_return`) and dtype semantics.
- [TIMESERIES-mode conventions](../reference/ts-mode-conventions.md) — the `N == 1` auto-routing rules and SE conventions.
- [PANEL vs TIMESERIES sample guard](../guides/panel-timeseries.md) — sample-size floors and recovery paths.
- [`run_metrics`](run-metrics.md) — descriptive twin of `evaluate`; computes the same statistics but makes no FDR claim. Use when you want the numbers without the inference framing.
