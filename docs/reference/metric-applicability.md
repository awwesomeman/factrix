# Metric applicability

A single matrix that maps each public metric to the analysis cell
where it is in scope, the canonical inferential role it plays in
that cell, and the sample-size threshold that gates it. Per-metric
formulae, parameters, and Notes / References live in the
[Metrics API pages](../api/metrics/index.md); this page is the
cross-metric overview.

## Sample dimensions

factrix expresses sample size on three axes (see
[Concepts](../getting-started/concepts.md) for cell taxonomy):

- `N` — assets in the panel (`asset_id` unique count).
- `T` — date count (`date` unique count).
- `K` — non-zero event count for `Sparse` factors
  (`filter(factor != 0).height`).
- `T/h` — non-overlapping date count given
  `forward_periods = h`.

`Mode` is derived from `N` at evaluate-time: `PANEL` for `N ≥ 2`,
`TIMESERIES` for `N == 1`. The dispatch registry routes to the cell's
procedure in either Mode; the metric's applicability does not change
across Modes, only the sample axis that constrains it.

## Cross-metric matrix

| Metric | Cell | Canonical role | Sample axis | Min sample |
|---|---|---|---|---|
| [`ic`][factrix.metrics.ic.ic] | Individual × Continuous | Cell-canonical (IC) | `T/h` | `T/h ≥ MIN_NON_OVERLAP` |
| [`ic_newey_west`][factrix.metrics.ic.ic_newey_west] | Individual × Continuous | HAC variant of `ic` | `T` | `T ≥ MIN_IC_PERIODS` |
| [`ic_ir`][factrix.metrics.ic.ic_ir] | Individual × Continuous | IR / ICIR | `T` | `T ≥ MIN_IC_PERIODS` |
| [`regime_ic`][factrix.metrics.ic.regime_ic] | Individual × Continuous | Regime-stratified IC | `T` per regime | per-regime `T/h ≥ MIN_NON_OVERLAP` |
| [`multi_horizon_ic`][factrix.metrics.ic.multi_horizon_ic] | Individual × Continuous | Per-horizon IC + BHY | `T` per horizon | per-horizon `T/h ≥ MIN_NON_OVERLAP` |
| [`fama_macbeth`][factrix.metrics.fama_macbeth.fama_macbeth] | Individual × Continuous | Cell-canonical (FM) | `T` (λ series) | `MIN_FM_PERIODS = 20` |
| [`pooled_ols`][factrix.metrics.fama_macbeth.pooled_ols] | Individual × Continuous | Pooled OLS sibling of FM | `N × T` | `N ≥ 10`, `G ≥ 3` clusters |
| [`beta_sign_consistency`][factrix.metrics.fama_macbeth.beta_sign_consistency] | Individual × Continuous | Per-period β-sign hit rate | `T` (β series) | `T ≥ MIN_FM_PERIODS` |
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | Individual × Continuous | Top-minus-bottom spread t | `T/h` | `T/h ≥ MIN_NON_OVERLAP`; per-date `N ≥ n_groups` |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | Individual × Continuous | Value-weighted spread | `T/h` | as `quantile_spread` |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | Individual × Continuous | Group-rank monotonicity | `T/h` | per-date `N ≥ n_groups`; `T/h ≥ MIN_NON_OVERLAP` |
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | Individual × Continuous | Top-bucket HHI ratio | `T/h` | `T/h ≥ MIN_NON_OVERLAP` |
| [`turnover`][factrix.metrics.tradability.turnover] | Individual × Continuous | Rank-stability (`1 − ρ`) | `T` | `T ≥ 2 × forward_periods` |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | Individual × Continuous | Q1/Qn replacement fraction | `T` | as `turnover` |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | Individual × Continuous | bps cost where α → 0 | scalar | `notional_turnover > 0` |
| [`net_spread`][factrix.metrics.tradability.net_spread] | Individual × Continuous | Spread − cost × τ | scalar | spread + cost provided |
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | Spread-series consumer | Single-factor α post base | `T` | aligned spread series |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | Spread-series consumer | Diagnostic — not for inference | `T` | as `spanning_alpha` |
| [`caar`][factrix.metrics.caar.caar] | Individual × Sparse | Cell-canonical | `K/h` | `K/h ≥ MIN_NON_OVERLAP_EVENTS` |
| [`bmp_test`][factrix.metrics.caar.bmp_test] | Individual × Sparse | Variance-robust sibling | `K` | `K ≥ MIN_EVENTS`; `estimation_window` periods per asset |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate] | Individual × Sparse | Per-event sign hit rate | `K` | `K ≥ MIN_EVENTS` |
| [`event_ic`][factrix.metrics.event_quality.event_ic] | Individual × Sparse | Strength → return Spearman | `K` | `K ≥ MIN_EVENTS` |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | Individual × Sparse | `Σ gains / Σ losses` | `K` | `K ≥ MIN_EVENTS` |
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | Individual × Sparse | Per-event return skewness | `K` | `K ≥ MIN_EVENTS` |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | Individual × Sparse | Mean inter-event gap | `K` | `K ≥ 2` |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | Individual × Sparse | Per-offset return profile | per-offset `K` | `K ≥ MIN_EVENTS` |
| [`multi_horizon_hit_rate`][factrix.metrics.event_horizon.multi_horizon_hit_rate] | Individual × Sparse | Per-horizon binomial hit | per-horizon `K` | `K ≥ MIN_EVENTS` |
| [`mfe_mae_summary`][factrix.metrics.mfe_mae.mfe_mae_summary] | Individual × Sparse | Path-excursion summary | `K` | `K ≥ MIN_EVENTS`; `price` column required |
| [`clustering_diagnostic`][factrix.metrics.clustering.clustering_diagnostic] | Individual × Sparse (`N ≥ 2`) | Event-date HHI | `K`, `N` | `N ≥ 2`; `K ≥ MIN_EVENTS` |
| [`corrado_rank_test`][factrix.metrics.corrado.corrado_rank_test] | Individual × Sparse | Nonparametric rank test | `K` × estimation window | `K ≥ MIN_EVENTS`; per-asset `T ≥ 30` |
| [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | Common × Continuous | Cell-canonical | `N` (β distribution) | `N ≥ 2`; per-asset `T ≥ MIN_TS_OBS` |
| [`mean_r_squared`][factrix.metrics.ts_beta.mean_r_squared] | Common × Continuous | Avg explanatory R² | `N` | as `ts_beta` |
| [`compute_rolling_mean_beta`][factrix.metrics.ts_beta.compute_rolling_mean_beta] | Common × Continuous | β stability over time | `T` per window | window ≥ `MIN_TS_OBS` |
| [`ts_beta_sign_consistency`][factrix.metrics.ts_beta.ts_beta_sign_consistency] | Common × Continuous | Cross-asset β-sign rate | `N` | `N ≥ 2` |
| [`ts_quantile_spread`][factrix.metrics.ts_quantile.ts_quantile_spread] | Common × Continuous | Quantile-bucketed Wald | `T` | `T ≥ MIN_PORTFOLIO_PERIODS`; factor `n_unique ≥ n_groups × 2` |
| [`ts_asymmetry`][factrix.metrics.ts_asymmetry.ts_asymmetry] | Common × Continuous | Long/short slope asymmetry | `T` | factor has both signs (Gate B); each side `n_unique ≥ 2` for method B (Gate C) |
| [`hit_rate`][factrix.metrics.hit_rate.hit_rate] | Series-tools | Binomial hit rate | series length | `T/h ≥ MIN_NON_OVERLAP` |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | Series-tools | Theil-Sen slope + ADF flag | `T` | `T ≥ MIN_TREND_PERIODS` |
| [`multi_split_oos_decay`][factrix.metrics.oos.multi_split_oos_decay] | Series-tools | Median IS/OOS survival | `T` | `T ≥ 2 × MIN_OOS_PERIODS` |

Constant names refer to the `MIN_*` symbols in `factrix._types`;
inspect `factrix._types.MIN_FM_PERIODS` (and siblings) for current
values.

## Below-threshold behaviour

When the input fails a sample threshold, factrix never silently
returns a meaningful-looking result. Three deterministic outcomes:

- **Short-circuit** — the metric returns
  `MetricOutput(value=NaN, metadata={"reason": "..."})` and
  `FactorProfile.primary_p` is conservatively pinned to `1.0` so
  `verdict()` reports `FAILED`.
- **Fallback** — the dispatch registry routes to a degraded but
  semantically distinct procedure (e.g. `Common × Continuous` at
  `N == 1` falls back to single-asset OLS without HAC). `diagnose()`
  fires an `info`-severity rule recording the fallback.
- **Degraded** — the metric runs but a Profile field is set to
  `None` (e.g. `clustering_hhi=None` for single-asset event signals).
  `diagnose()` records the degradation.

Structural errors (wrong cell, missing column, `N == 1` on a cell that
requires `PANEL` Mode) raise `ValueError` / `ConfigError` rather than
falling back.
