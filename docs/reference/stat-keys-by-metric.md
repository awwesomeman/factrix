---
title: Stat keys by metric
---

!!! abstract "Answers"
    `MetricResult` schema — which `metadata` key is the primary inference target, which are auxiliary, what the headline `stat` carries.
    For applicability gates, see [Metric applicability](metric-applicability.md).
    For computation pipeline, see [Metric pipelines](metric-pipelines.md).

Per-metric schema of the [`MetricResult`](../api/evaluation-results.md#factrix.MetricResult)
returned by every public callable in `factrix.metrics`.

For the SE / test machinery itself see
[Statistical methods](statistical-methods.md). For the
`MetricResult.name` → docs-page reverse index see
[`MetricResult`](../api/evaluation-results.md#factrix.MetricResult). The
`evaluate()`-side equivalent is `EvaluationResult.metrics`.

`metadata` keys are tagged by role in the per-metric subsections
below:

- **primary** — carries `p_value` / the inference target.
- **secondary-test** — a complementary p-value / statistic from a
  different test on the same data (e.g. `long_p_value` / `short_p_value`
  legs of `quantile_spread`).
- **descriptive** — sample-size diagnostics, method labels,
  parameter echoes; not a test result.
- **conditional** — emitted only on certain branches; the trigger
  is named in parentheses.

Hypothesis-test metrics share a common envelope (`p_value`,
`stat_type`, `h0`, `method`) — listed once here, not repeated per
metric below. Cross-slice inference functions
([`slice_pairwise_test`][factrix.slice_pairwise_test] /
[`slice_joint_test`][factrix.slice_joint_test]) are
not listed in the table: their headline output is a DataFrame of
contrasts, not a sidecar to a primary value.

## Cross-metric summary

| Metric | Primary stat (`MetricResult.stat`) | Primary `metadata` key | `value` |
|---|---|---|---|
| [`ic`][factrix.metrics.ic.ic] | `t` on per-date information coefficient (IC) series (non-overlapping default, Newey-West HAC if configured) | `p_value` | mean(IC) |
| [`ic_ir`][factrix.metrics.ic.ic_ir] | none — descriptive | — | mean(IC) / std(IC) |
| [`fm_beta`][factrix.metrics.fm_beta.fm_beta] | NW HAC `t` on per-date λ | `p_value` | mean(β) |
| [`pooled_beta`][factrix.metrics.fm_beta.pooled_beta] | clustered ordinary least squares (OLS) `t` (or `None` if G < 3) | `p_value` | pooled β |
| [`beta_sign_consistency`][factrix.metrics.fm_beta.beta_sign_consistency] | none — descriptive | — | fraction with expected sign |
| [`caar`][factrix.metrics.caar.caar] | non-overlapping `t` on event-date CAAR | `p_value` | mean(CAAR) |
| [`bmp_test`][factrix.metrics.caar.bmp_test] | BMP cross-sectional `z` on SAR | `p_value` | mean(SAR) |
| [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank] | nonparametric rank `z` | `p_value` | mean(U × sign(factor)) |
| [`hit_rate`][factrix.metrics.hit_rate.hit_rate] | binomial test (or normal `z`) | `p_value` | hit rate ∈ [0, 1] |
| [`directional_hit_rate`][factrix.metrics.directional_hit_rate.directional_hit_rate] | Pesaran-Timmermann `z` (one-sided) | `p_value` | directional hit rate ∈ [0, 1] |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate] | binomial test (or normal `z`) | `p_value` | hit rate ∈ [0, 1] |
| [`event_ic`][factrix.metrics.event_quality.event_ic] | Fisher-transformed Spearman `z` | `p_value` | Spearman ρ |
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | D'Agostino skew `z` (N ≥ 20) | `p_value` (conditional) | Fisher skewness |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | none — descriptive | — | gains / \|losses\| |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | none — descriptive | — | mean bars per event |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | none — descriptive | — | mean leakage score |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | cross-asset `t` on signed Spearman | `p_value` | mean \|Spearman\| |
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | NW HAC `t` on top-bottom spread (block-bootstrap CI when small cross-section) | `p_value` | mean(spread) |
| [`k_spread`][factrix.metrics.k_spread.k_spread] | non-overlapping `t` on top-K−bottom-K spread (block-bootstrap CI when small cross-section) | `p_value` | mean(spread) |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | NW HAC `t` on vw spread | `p_value` | mean(vw spread) |
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | one-sided `t` on diversity ratio | `p_value` | mean(eff_n / n_top) |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | none — descriptive | — | event-date Herfindahl-Hirschman index (HHI) |
| [`mfe_mae_summary`][factrix.metrics.mfe_mae.mfe_mae_summary] | none — descriptive | — | MFE_p50 / \|MAE_p75\| |
| [`oos_decay`][factrix.metrics.oos_decay.oos_decay] | none — descriptive | — | median(survival) |
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | OLS `t` on α | `p_value` | spanning α |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | none — selection meta | — | (NaN; results in metadata) |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | Theil-Sen slope `t` (CI-based) | `p_value` | Theil-Sen slope |
| [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | cross-asset `t` on per-asset β | `p_value` | mean(β) |
| [`ts_beta_sign_consistency`][factrix.metrics.ts_beta.ts_beta_sign_consistency] | none — descriptive | — | max(p, 1-p) on sign fraction |
| [`mean_r_squared`][factrix.metrics.ts_beta.mean_r_squared] | none — descriptive | — | mean(R²) |
| [`ts_asymmetry`][factrix.metrics.ts_asymmetry.ts_asymmetry] | Wald χ² (NW HAC) on slope sum / equality | `p_value` | β_long + β_short |
| [`ts_quantile_spread`][factrix.metrics.ts_quantile.ts_quantile_spread] | Wald (NW HAC) on bucket β contrast | `p_value` | top − bottom bucket β |
| [`turnover`][factrix.metrics.tradability.turnover] | none — descriptive | — | 1 − mean(rank-AC) |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | none — descriptive | — | replaced fraction |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | none — descriptive | — | breakeven spread (bps) |
| [`net_spread`][factrix.metrics.tradability.net_spread] | none — descriptive | — | net spread (bps) |

## Per-metric schemas

### `ic` family (`factrix.metrics.ic`)

#### `ic`

- *primary*: `p_value` — `t`-test on the per-date IC series (non-overlapping stride with stride `forward_periods` by default, or Newey-West HAC if configured).
- *descriptive*: `n_periods`, `forward_periods`, `tie_ratio` (median across dates), `stat_type` (`"t"`), `h0` (`"mu=0"`), `method`.

#### `ic_ir`

Descriptive metric — `MetricResult.stat` is `None` and no `p_value`
is emitted.

- *descriptive*: `mean_ic`, `std_ic`, `n_periods`, `tie_ratio`.

### `fm_beta` family (`factrix.metrics.fm_beta`)

#### `fm_beta` (emits `MetricResult.name = "fm_beta"`)

- *primary*: `p_value` — NW HAC `t` on per-date λ. With
  `is_estimated_factor=True` the Shanken EIV correction is applied
  post-hoc and the corrected `p_value` replaces the raw value.
- *secondary-test* (conditional, Shanken applied):
  `p_value_uncorrected`, `stat_uncorrected`.
- *descriptive*: `n_periods`, `newey_west_lags`, `forward_periods`,
  `is_estimated_factor`, `warning_codes` (conditional).
- *descriptive* (conditional, Shanken applied): `shanken_c`,
  `shanken_factor_return_var`, `shanken_factor_return_var_source`.
- *descriptive* (conditional, σ²_f ≈ 0): `shanken_correction` =
  `"skipped_zero_factor_variance"` — the correction is undefined
  when the factor-return variance collapses; the uncorrected NW
  result is reported.

#### `pooled_beta` (emits `MetricResult.name = "pooled_beta"`)

- *primary*: `p_value` — single- or two-way clustered OLS `t`. When
  the cluster count G < 3 the test is short-circuited with `stat =
  None` and `p_value = 1.0`.
- Sample size: `MetricResult.n_obs` (row count entering the test).
- *descriptive*: `n_clusters` (one-way) or `n_clusters_a`,
  `n_clusters_b`, `n_clusters_intersection` (two-way).
- *descriptive* (conditional, short-circuit): `reason =
  "insufficient_clusters"`, `n_clusters` (smallest G — first-class
  `n_obs` carries the row count), `min_required` (always 3).
- *descriptive* (conditional): `variance_non_psd_fallback` — names
  the fallback path when the meat matrix is non-PSD.
- *descriptive* (Driscoll-Kraay path, `driscoll_kraay=True`):
  `se_method` (`"driscoll_kraay"`), `n_periods` (length of the
  cross-sectional score-sum series), and `driscoll_kraay_lags` (the
  Bartlett bandwidth used). The DK path uses `df = n_periods − 1`,
  emits `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` below 30 periods, and
  short-circuits with `reason = "insufficient_periods"` below 3.

#### `beta_sign_consistency`

Descriptive; no test.

- *descriptive*: `expected_sign`, `n_periods`.

### `caar` family (`factrix.metrics.caar`)

#### `caar`

- *primary*: `p_value` — non-overlapping `t` on per-event-date CAAR.
- *descriptive*: `n_event_dates`, `n_sampled`,
  `warning_codes` (conditional, e.g. `FEW_EVENTS`).

#### `bmp_test`

Boehmer-Musumeci-Poulsen standardised-abnormal-return cross-sectional
`z` test, with optional Kolari-Pynnönen clustering adjustment.

- *primary*: `p_value`.
- *descriptive*: `n_events`, `n_dropped`, `std_sar`,
  `estimation_window`, `include_prediction_error_variance`.
- *descriptive* (conditional, KP applied): `kolari_pynnonen_r`,
  `kolari_pynnonen_n_eff`, `kolari_pynnonen_r_source`,
  `kolari_pynnonen_applied`, `kolari_pynnonen_scaling`,
  `stat_uncorrected`.

### `corrado` (`factrix.metrics.corrado_rank`)

#### `corrado_rank` (emits `MetricResult.name = "corrado_rank"`)

- *primary*: `p_value` — Corrado nonparametric rank `z`.
- *descriptive*: `n_events`, `n_total_obs`.

### `hit_rate` (`factrix.metrics.hit_rate`)

#### `hit_rate`

`MetricResult.stat` is the binomial hit count when the exact branch
runs, the normal `z` when the approximation branch runs;
`stat_type` discriminates (`"binomial_hits"` vs `"z"`).

- *primary*: `p_value` — binomial / normal-approximation test on
  non-overlapping wins (stride `forward_periods`).
- *descriptive*: `n_hits`, `n_total`.

### `directional_hit_rate` (`factrix.metrics.directional_hit_rate`)

#### `directional_hit_rate`

Small-N robust sibling of `hit_rate`. `MetricResult.value` is the
directional hit rate (sign-agreement fraction); `stat` is the
Pesaran-Timmermann `z` statistic (`stat_type="z"`), tested one-sided.

- *primary*: `p_value` — one-sided Pesaran-Timmermann test conditioning
  on the marginal up/down frequencies of prediction and realisation.
- *descriptive*: `p_correct` (realised hit rate), `p_expected`
  (hit rate under directional independence), `p_up_pred` (fraction of
  positive predictions), `p_up_real` (fraction of positive realisations).

### `event_quality` (`factrix.metrics.event_quality`)

#### `event_hit_rate`

Same shape as `hit_rate` (binomial / normal-approx branches).

- *primary*: `p_value`.
- *descriptive*: `n_events`, `n_hits`.

#### `event_ic`

- *primary*: `p_value` — Fisher-transformed Spearman ρ between
  `|factor|` and `signed_car`.
- *descriptive*: `n_events`.

`MetricResult.stat = None` and the short-circuit `reason` is set to
`"not_applicable_discrete_signal"` when the signal lacks magnitude
variance (e.g. binary {-1, +1}).

#### `event_skewness`

- *primary* (conditional, N ≥ 20): `p_value` — D'Agostino skew `z`.
- *descriptive*: `n_events`.

When `N < 20`, `MetricResult.stat = None` and `p_value` / `stat_type`
/ `h0` / `method` are omitted — the metric reports the Fisher
skewness in `value` only.

#### `profit_factor`

Descriptive; no test.

- *descriptive*: `total_gains`, `total_losses`, `n_events`, `n_wins`,
  `n_losses`.

#### `signal_density`

Per-asset event frequency; descriptive (the period-axis analogue
is `clustering_hhi`).

- *descriptive*: `n_events_total`, `n_assets_with_events`,
  `mean_events_per_asset`, `mean_bars_between_events`.

### `event_horizon` (`factrix.metrics.event_horizon`)

#### `event_around_return`

Pre/post-event return profile; descriptive.

- *descriptive*: `per_offset` (dict `offset → {mean, median, p25, p75,
  hit_rate, n}`), `interpretation`.
- *descriptive*: `p_value` (sentinel; not a test result — kept for
  uniform `MetricResult` shape).

### `monotonicity` (`factrix.metrics.monotonicity`)

#### `monotonicity`

`MetricResult.value` carries the *magnitude* (mean `|Spearman|`);
`MetricResult.stat` carries the cross-asset `t` on the *signed*
Spearman series. The split is intentional — magnitude and direction
consistency are read separately.

- *primary*: `p_value` — cross-asset `t` (`H₀: μ = 0`).
- *descriptive*: `mean_signed`, `n_valid_periods`, `n_groups`,
  `tie_ratio`, `tie_policy`.

### `quantile` (`factrix.metrics.quantile`)

#### `quantile_spread`

- *primary*: `p_value` — non-overlapping `t`-test on the
  (top − bottom) spread series. Small cross-sections
  (`n_assets < MIN_ASSETS_WARN`) switch to a block-bootstrap CI; see
  the shared small-N keys below.
- *secondary-test*: `long_alpha`, `long_stat`, `long_p_value` —
  long-leg attribution (mean excess and `t` / p-value).
- *secondary-test*: `short_alpha`, `short_stat`, `short_p_value`,
  `short_significance` — short-leg attribution.
- *descriptive*: `n_periods`, `tie_ratio`, `tie_policy`, `method`.

#### `quantile_spread_vw`

Value-weighted variant. Same metadata shape as `quantile_spread`
plus a `weights_lagged` flag indicating whether the weighting input
was lagged before the join (descriptive).

### `k_spread` (`factrix.metrics.k_spread`)

#### `k_spread`

Fixed-K (top-K − bottom-K) long-short spread; the small-N sibling of
`quantile_spread`.

- *primary*: `p_value` — non-overlapping `t`-test on the spread
  series, or a block-bootstrap CI in the small-cross-section regime
  (`method` records which).
- *descriptive*: `k` (names per leg), `cross_sectional_dispersion`
  (mean per-date cross-sectional return std), `top_return`,
  `bottom_return`, `n_periods`, `method`. The `k`-too-large
  short-circuit reports `max_assets_per_date`.

#### Shared small-N significance keys

Both `quantile_spread` and `k_spread` switch the headline test to a
block-bootstrap CI when `n_assets < MIN_ASSETS_WARN`. In that branch
they additionally emit `p_value_t` (the parametric `t` p-value kept
for reference), `bootstrap_block_length`, `bootstrap_n_resamples`,
and `bootstrap_seed`. The switch is **not** silent: the cross-section
tier code (`small_cross_section_n` / `borderline_cross_section_n`) is
attached to `warning_codes`, so the method change surfaces as a
`Warning` on the result.

### `concentration` (`factrix.metrics.concentration`)

#### `top_concentration`

`H₀: ratio ≥ 0.5` (one-sided). Tests whether the top-bucket
diversity ratio (effective-n / n_top, derived from HHI) falls
*below* the 0.5 threshold — i.e. concentration risk.

- *primary*: `p_value` — one-sided `t`.
- *descriptive*: `mean_n_top`, `ratio_eff_to_total`, `tie_ratio`,
  `weight_by`, `warning_codes` (conditional).

### `clustering` (`factrix.metrics.clustering_hhi`)

#### `clustering_hhi` (emits `MetricResult.name = "clustering_hhi"`)

Descriptive; period-axis concentration of event dates.

- *descriptive*: `n_events`, `n_event_dates`, `effective_n_dates`,
  `hhi_normalized`, `cluster_window`.

### `mfe_mae` (`factrix.metrics.mfe_mae`)

#### `mfe_mae_summary`

Descriptive; no test.

- *descriptive*: `mfe_p50`, `mae_p75`, `mae_p95`, `mfe_mae_ratio`,
  `bars_to_mfe_mean`, `bars_to_mae_mean`, `n_events`.
- *descriptive* (conditional, when σ-normalised inputs available):
  `mfe_z_p50`, `mae_z_p75`, `mfe_mae_ratio_z`, `n_events_z`.
- *descriptive*: `p_value` (sentinel).

### `oos` (`factrix.metrics.oos_decay`)

#### `oos_decay` (emits `MetricResult.name = "oos_decay"`)

`MetricResult.stat = None`; rank-based PASS/VETO gate, no formal
hypothesis test.

- *descriptive*: `status` (`"PASS"` / `"VETOED"`), `sign_flipped`,
  `per_split` (list of `{is_ratio, mean_is, mean_oos,
  survival_ratio, sign_flipped}`), `survival_threshold`, `n_splits`,
  `method`.

### `spanning` (`factrix.metrics.spanning`)

#### `spanning_alpha`

- *primary*: `p_value` — OLS `t` on α from the multivariate spanning
  regression. Plain (non-HAC) SE — assumes the input spread series
  are non-overlapping.
- Sample size: `MetricResult.n_obs` (length of the aligned
  candidate-series).
- *descriptive*: `n_base_factors`, `base_factors` (list of base-factor
  names), `betas` (per-base OLS slope dict), `r_squared`.
- *descriptive* (conditional, short-circuit): `reason`.

#### `greedy_forward_selection`

Stepwise selection meta-metric; `MetricResult.value` is `NaN` and
`MetricResult.stat = None`. Per-candidate `t`-stats are *not* valid
for inference (selection bias).

- *descriptive*: `selected_factors` (list of `SpanningResult`),
  `eliminated_factors`, `all_candidates`,
  `t_stats_inference_invalid` (always `True`).

### `trend` (`factrix.metrics.trend`)

#### `ic_trend`

Theil-Sen median slope on the IC series. The reported `MetricResult.stat`
is the slope-`t` derived from the rank-based confidence interval.

- *primary*: `p_value` — slope significance from the Theil-Sen CI.
- *descriptive*: `n_periods`, `ci_low`, `ci_high`,
  `ci_excludes_zero`, `intercept`.
- *descriptive* (conditional, augmented Dickey-Fuller (ADF) run): `adf_stat`, `adf_p`,
  `unit_root_suspected`.

### `ts_beta` (`factrix.metrics.ts_beta`)

#### `ts_beta`

- *primary*: `p_value` — cross-asset `t` on the per-asset OLS β
  distribution.
- *descriptive*: `n_assets`, `beta_std`, `median_beta`.

#### `mean_r_squared`

Descriptive; no test.

- *descriptive*: `n_assets`, `median_r_squared`, `min_r_squared`,
  `max_r_squared`.

#### `ts_beta_sign_consistency`

Descriptive symmetric consistency — `value ∈ [0.5, 1.0]`.

- *descriptive*: `n_assets`, `fraction_positive`.

### `ts_asymmetry` (`factrix.metrics.ts_asymmetry`)

#### `ts_asymmetry`

Two complementary methods:

- **Method A** (always): Wald χ² on `H₀: β_long + β_short = 0` with
  NW HAC SE.
- **Method B** (conditional, ≥ 2 distinct values per side):
  Wald χ² on `H₀: β_pos = β_neg`.

- *primary*: `p_value` — Method A.
- *secondary-test* (conditional, Method B ran):
  `beta_pos`, `beta_neg`, `p_wald_slopes`.
- *descriptive*: `beta_long`, `beta_short`, `abs_short_over_long`,
  `n_pos`, `n_neg`, `n_zero`, `n_periods`, `nw_lags_used`,
  `method_b_skipped` (conditional), `intercept` (conditional),
  `beta_zero` (conditional).

### `ts_quantile` (`factrix.metrics.ts_quantile`)

#### `ts_quantile_spread`

- *primary*: `p_value` — Wald `χ²` (NW HAC) on `H₀: β_top = β_bottom`
  from an OLS fit on bucket dummies.
- *secondary-test*: `spearman_rho`, `spearman_p` — small-sample
  Spearman of (bucket-idx, mean-return) for monotonicity diagnostic.
- *descriptive*: `n_groups`, `n_periods`, `n_distinct_factor`,
  `nw_lags_used`, `buckets` (list of `{idx, mean_return, n}`).

### `tradability` (`factrix.metrics.tradability`)

All four are descriptive — `MetricResult.stat = None` and no
`p_value` is emitted. They feed cost/benefit arithmetic, not
inference.

#### `turnover`

- *descriptive*: `mean_rank_autocorrelation`,
  `std_rank_autocorrelation`, `n_pairs`, `forward_periods`,
  `quantile`, `n_cross_section_mean`.

#### `notional_turnover`

- *descriptive*: `n_rebalances`, `n_groups`, `forward_periods`,
  `mean_tail_size`.

#### `breakeven_cost`

Scalar-input metric (consumes pre-aggregated scalars rather than a
date-keyed DataFrame).

- *descriptive*: `gross_spread`, `turnover`, `forward_periods`.

#### `net_spread`

Scalar-input metric.

- *descriptive*: `gross_spread`, `cost_drag`, `estimated_cost_bps`,
  `turnover`, `forward_periods`.

## Short-circuit envelope

Every metric falls back to a uniform short-circuit `MetricResult`
when input data fails the metric's preconditions (insufficient
sample, no events, degenerate signal, …). The fallback shape is:

- `value = float("nan")`, `stat = None`, `significance = ""`.
- `MetricResult.n_obs: int | None` — first-class sample size the
  estimator saw before bailing (e.g. how many periods / events were
  actually available). Populated when the short-circuit knows the
  number; `None` otherwise.
- `metadata["reason"]: str` names the short-circuit branch (e.g.
  `"insufficient_periods"`, `"no_events"`,
  `"not_applicable_discrete_signal"`, `"insufficient_clusters"`).
- `metadata["p_value"] = 1.0` — conservative default so Benjamini-Hochberg-Yekutieli (BHY) treats
  short-circuited metrics as rejected rather than crashing.
- Optional diagnostic keys naming what was missing or under-spec:
  `min_required`, `min_required_per_asset`, `min_required_per_regime`,
  `missing_column`, `std_u`, `hint`, `n_distinct`. Each is
  descriptive — emitted only on the short-circuit branch that
  needed it; consumers should branch on `reason` before reading.

The auxiliary `metadata` keys listed in the per-metric subsections
above are *not* present on the short-circuit path.
