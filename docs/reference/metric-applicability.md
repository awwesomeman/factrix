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
| [`ic`][factrix.metrics.ic.ic] | Individual × Continuous | Cell-canonical (IC) | `T/h` (non-overlap) | `T/h ≥ MIN_ASSETS_PER_DATE_IC` (= 10) via `_scaled_min_periods(MIN_ASSETS_PER_DATE_IC, h)` |
| [`ic_newey_west`][factrix.metrics.ic.ic_newey_west] | Individual × Continuous | HAC variant of `ic` | `T` (full series) | `T ≥ MIN_ASSETS_PER_DATE_IC` |
| [`ic_ir`][factrix.metrics.ic.ic_ir] | Individual × Continuous | IR / ICIR | `T` | `T ≥ MIN_ASSETS_PER_DATE_IC` |
| [`regime_ic`][factrix.metrics.ic.regime_ic] | Individual × Continuous | Regime-stratified IC | `T` per regime | per-regime `T/h ≥ MIN_ASSETS_PER_DATE_IC` |
| [`multi_horizon_ic`][factrix.metrics.ic.multi_horizon_ic] | Individual × Continuous | Per-horizon IC + BHY | `T` per horizon | per-horizon `T/h ≥ MIN_ASSETS_PER_DATE_IC` |
| [`fama_macbeth`][factrix.metrics.fama_macbeth.fama_macbeth] | Individual × Continuous | Cell-canonical (FM) | `T` (λ series) | `T ≥ MIN_FM_PERIODS_HARD` (= 4); warn if `T < MIN_FM_PERIODS_WARN` (= 30) |
| [`pooled_ols`][factrix.metrics.fama_macbeth.pooled_ols] | Individual × Continuous | Pooled OLS sibling of FM | `N × T` | `N ≥ 10`, effective clusters `G ≥ 3` |
| [`beta_sign_consistency`][factrix.metrics.fama_macbeth.beta_sign_consistency] | Individual × Continuous | Per-period β-sign hit rate | `T` (β series) | `T ≥ MIN_FM_PERIODS_HARD` |
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | Individual × Continuous | Top-minus-bottom spread t | `T/h` | `T/h ≥ MIN_PORTFOLIO_PERIODS_HARD` (= 3); warn if `T/h < MIN_PORTFOLIO_PERIODS_WARN` (= 20); per-date `N ≥ n_groups` |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | Individual × Continuous | Value-weighted spread | `T/h` | as `quantile_spread` |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | Individual × Continuous | Group-rank monotonicity | `T/h` | per-date `N ≥ n_groups`; series `≥ MIN_MONOTONICITY_PERIODS` (= 5) |
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | Individual × Continuous | Top-bucket HHI ratio | `T/h` | `T/h ≥ MIN_PORTFOLIO_PERIODS_HARD`; warn if `T/h < MIN_PORTFOLIO_PERIODS_WARN` |
| [`turnover`][factrix.metrics.tradability.turnover] | Individual × Continuous | Rank-stability (`1 − ρ`) | `T` | `T ≥ 2·forward_periods + 1` |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | Individual × Continuous | Q1/Qn replacement fraction | `T` | as `turnover` |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | Individual × Continuous | bps cost where α → 0 | scalar | `notional_turnover > 0` |
| [`net_spread`][factrix.metrics.tradability.net_spread] | Individual × Continuous | $\text{spread} - \text{cost} \cdot \tau$ | scalar | spread + cost provided |
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | Spread-series consumer | Single-factor α post base | `T` | aligned spread series |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | Spread-series consumer | Diagnostic — not for inference | `T` | as `spanning_alpha` |
| [`caar`][factrix.metrics.caar.caar] | Individual × Sparse | Cell-canonical | `K/h` (non-overlap) | `K/h ≥ MIN_EVENTS_HARD` (= 4); warn if `K/h < MIN_EVENTS_WARN` (= 30) via `_scaled_min_periods(MIN_EVENTS_HARD, h)` |
| [`bmp_test`][factrix.metrics.caar.bmp_test] | Individual × Sparse | Variance-robust sibling | `K` | `K ≥ MIN_EVENTS_HARD`; `estimation_window` periods per asset |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate] | Individual × Sparse | Per-event sign hit rate | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`event_ic`][factrix.metrics.event_quality.event_ic] | Individual × Sparse | Strength → return Spearman | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | Individual × Sparse | `Σ gains / Σ losses` | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | Individual × Sparse | Per-event return skewness | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | Individual × Sparse | Mean inter-event gap | `K` | `K ≥ 2` |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | Individual × Sparse | Per-offset return profile | per-offset `K` | `K ≥ MIN_EVENTS_HARD` |
| [`multi_horizon_hit_rate`][factrix.metrics.event_horizon.multi_horizon_hit_rate] | Individual × Sparse | Per-horizon binomial hit | per-horizon `K` | `K ≥ MIN_EVENTS_HARD` |
| [`mfe_mae_summary`][factrix.metrics.mfe_mae.mfe_mae_summary] | Individual × Sparse | Path-excursion summary | `K` | `K ≥ MIN_EVENTS_HARD`; `price` column required |
| [`clustering_diagnostic`][factrix.metrics.clustering.clustering_diagnostic] | Individual × Sparse (`N ≥ 2`) | Event-date HHI | `K`, `N` | `N ≥ 2`; `K ≥ MIN_EVENTS_HARD` |
| [`corrado_rank_test`][factrix.metrics.corrado.corrado_rank_test] | Individual × Sparse | Nonparametric rank test | `K` × estimation window | `K ≥ MIN_EVENTS_HARD`; per-asset `T ≥ 30` |
| [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | Common × Continuous | Cell-canonical | `N` (β distribution) | `N ≥ 2`; per-asset `T ≥ MIN_TS_OBS` (= 20) |
| [`mean_r_squared`][factrix.metrics.ts_beta.mean_r_squared] | Common × Continuous | Avg explanatory R² | `N` | as `ts_beta` |
| [`compute_rolling_mean_beta`][factrix.metrics.ts_beta.compute_rolling_mean_beta] | Common × Continuous | β stability over time | `T` per window | window ≥ `MIN_TS_OBS` |
| [`ts_beta_sign_consistency`][factrix.metrics.ts_beta.ts_beta_sign_consistency] | Common × Continuous | Cross-asset β-sign rate | `N` | `N ≥ 2` |
| [`ts_quantile_spread`][factrix.metrics.ts_quantile.ts_quantile_spread] | Common × Continuous | Quantile-bucketed Wald | `T` | `T ≥ MIN_PORTFOLIO_PERIODS_HARD`; factor `n_unique ≥ n_groups × 2` |
| [`ts_asymmetry`][factrix.metrics.ts_asymmetry.ts_asymmetry] | Common × Continuous | Long/short slope asymmetry | `T` | factor has both signs (Gate B); each side `n_unique ≥ 2` for method B (Gate C) |
| [`hit_rate`][factrix.metrics.hit_rate.hit_rate] | Series-tools | Binomial hit rate | series length | `T ≥ MIN_ASSETS_PER_DATE_IC` |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | Series-tools | Theil-Sen slope + ADF flag | `T` | `T ≥ 10` (literal floor) |
| [`multi_split_oos_decay`][factrix.metrics.oos.multi_split_oos_decay] | Series-tools | Median IS/OOS survival | `T` | `T ≥ 2 × MIN_OOS_PERIODS` (= 10) |

Constants in the `Min sample` column come from three locations and follow
a two-tier `_HARD` / `_WARN` model (see "Sample-size sensitivity" below):

- `factrix._types` — cross-metric defaults (`MIN_ASSETS_PER_DATE_IC`,
  `MIN_EVENTS_HARD = 4`, `MIN_EVENTS_WARN = 30`, `MIN_OOS_PERIODS = 5`,
  `MIN_PORTFOLIO_PERIODS_HARD = 3`, `MIN_PORTFOLIO_PERIODS_WARN = 20`,
  `MIN_MONOTONICITY_PERIODS`).
- `factrix._stats.constants` — procedure-level guards
  (`MIN_PERIODS_HARD = 20`, `MIN_PERIODS_WARN = 30`,
  `MIN_BROADCAST_EVENTS_HARD = 5`, `MIN_BROADCAST_EVENTS_WARN = 20`,
  `MIN_ASSETS = 10`, `MIN_ASSETS_WARN = 30`).
- The metric module itself for cell-specific thresholds:
  `MIN_FM_PERIODS_HARD = 4` / `MIN_FM_PERIODS_WARN = 30` in
  `factrix.metrics.fama_macbeth`, `MIN_TS_OBS = 20` in
  `factrix.metrics.ts_beta`.

For non-overlapping metrics (`ic`, `caar`, …) the effective floor is
`_scaled_min_periods(base, forward_periods)` (in
`factrix.metrics._helpers`), which scales the base constant by the
forward-return horizon `h`.

## Sample-size sensitivity — the two-tier `_HARD` / `_WARN` model

Inferential metrics enforce two separate floors:

- **`_HARD`** — the **mathematical floor**. Below it, the statistic is
  not defined (e.g. `t = (μ − 0) / (s / √n)` requires `n ≥ 2`; the FM
  small-sample HAC needs at least a few lagged covariances). `n < HARD`
  short-circuits to `MetricOutput(value=NaN, metadata={"reason": ...})`.
- **`_WARN`** — the **literature / power floor**. The statistic *is*
  computable, but the SE is biased small or power is poor; the metric
  returns the stat, emits a `UserWarning`, and adds a `WarningCode` to
  `MetricOutput.metadata["warning_codes"]` so `FactorProfile.warnings`
  can propagate it. `n ≥ WARN` is silent.

**Descriptive metrics** (`clustering_diagnostic`, `corrado_rank_test`,
`event_around_return`, `multi_horizon_hit_rate`, `event_hit_rate`,
`event_ic`, `profit_factor`, `event_skewness`, `mfe_mae_summary`,
`quantile_spread`, `ts_quantile_spread`, `ts_asymmetry`, `bmp_test`)
enforce **`_HARD` only** — they have no formal H₀ under which power
can be characterised, so the literature `_WARN` tier is undefined
for them. They accept smaller-`n` inputs than the inferential
canonicals.

A few specific caveats worth flagging:

- **`MIN_FM_PERIODS_HARD = 4` / `MIN_FM_PERIODS_WARN = 30`** for
  `fama_macbeth`. `T = 4` is the math floor at which the NW HAC `t`
  is computable; the small-sample HAC is known to **over-reject**, so
  in `T ∈ [4, 30)` the metric emits
  `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` and treats *p*-values as
  anti-conservative. Forty periods is the practical floor in the
  panel-econometrics literature.
- **`MIN_EVENTS_HARD = 4` / `MIN_EVENTS_WARN = 30`** for the
  event-study CAAR `t`. Brown & Warner (1985) tabulate well-behaved
  power at `K ≥ 50` and use `K ≥ 30` as the conventional minimum; in
  `K ∈ [4, 30)` the parametric `caar` is under-powered and
  `WarningCode.FEW_EVENTS_BROWN_WARNER` fires. The `bmp_test` /
  `corrado_rank_test` siblings only partly mitigate.
- **`MIN_PORTFOLIO_PERIODS_HARD = 3` / `MIN_PORTFOLIO_PERIODS_WARN = 20`**
  in `top_concentration` and `ts_quantile_spread`. Below 3 there is
  no spread / concentration t to compute; in `[3, 20)` the metric
  returns the stat with `WarningCode.BORDERLINE_PORTFOLIO_PERIODS`.
  **`MIN_OOS_PERIODS = 5`** in `multi_split_oos_decay` remains
  single-tier — the metric is now descriptive-only (no `p_value` in
  metadata), so a literature power floor is moot. Treat its output as
  descriptive; the formal `verdict()` reading should rely on the
  cell-canonical metric until the underlying series is materially
  longer.

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
