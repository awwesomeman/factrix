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
| [`directional_pair_accuracy`][factrix.metrics.directional_pair_accuracy.directional_pair_accuracy] | none; descriptive | n/a | pooled pairwise ordering accuracy |
| [`common_beta_profile`][factrix.metrics.common_beta.common_beta_profile] | none; descriptive | n/a | positive-minus-negative beta mean spread |
| [`ic`][factrix.metrics.ic.ic] | `t` on per-period information coefficient (IC) series (non-overlapping default, Newey-West HAC if configured) | `p_value` | mean(IC) |
| [`ic_ir`][factrix.metrics.ic.ic_ir] | none — descriptive | — | mean(IC) / std(IC) |
| [`fm_beta`][factrix.metrics.fm_beta.fm_beta] | NW HAC `t` on per-period λ | `p_value` | mean(β) |
| [`pooled_beta`][factrix.metrics.fm_beta.pooled_beta] | clustered ordinary least squares (OLS) `t` (or `None` if G < 3) | `p_value` | pooled β |
| [`fm_beta_sign_consistency`][factrix.metrics.fm_beta.fm_beta_sign_consistency] | none — descriptive | — | fraction with expected sign |
| [`caar`][factrix.metrics.caar.caar] | non-overlapping `t` on event-period CAAR | `p_value` | mean(CAAR) **per period** — `CAR / h`, not cumulative |
| [`bmp_z`][factrix.metrics.caar.bmp_z] | BMP cross-sectional `z` on SAR | `p_value` | mean(SAR) |
| [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank] | nonparametric rank `z` on the event-period series (cluster-robust) | `p_value` | mean(per-period mean U × sign(factor)) |
| [`positive_rate`][factrix.metrics.positive_rate.positive_rate] | binomial test (or normal `z`) | `p_value` | hit rate ∈ [0, 1] |
| [`directional_hit_rate`][factrix.metrics.directional_hit_rate.directional_hit_rate] | Pesaran-Timmermann `z` (one-sided) | `p_value` | directional hit rate ∈ [0, 1] |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate] | binomial test (or normal `z`) | `p_value` | hit rate ∈ [0, 1] |
| [`event_ic`][factrix.metrics.event_quality.event_ic] | Fisher-transformed Spearman `z` | `p_value` | Spearman ρ |
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | D'Agostino skew `z` (N ≥ 20) | `p_value` (conditional) | Fisher skewness |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | none — descriptive | — | gains / \|losses\| |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | none — descriptive | — | mean bars per event |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | none — descriptive | — | mean leakage score |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | Patton-Timmermann (2010) MR (stationary-bootstrap p) | `p_value` | `mr_min_diff` (min adjacent bucket-return difference) |
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | non-overlapping `t` on top-bottom spread (NW HAC under `NEWEY_WEST`) | `p_value` | mean(spread) |
| [`k_spread`][factrix.metrics.k_spread.k_spread] | non-overlapping `t` on top-K−bottom-K spread (NW HAC under `NEWEY_WEST`) | `p_value` | mean(spread) |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | non-overlap `t` (default) or NW HAC `t` on vw spread | `p_value` | mean(vw spread) |
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | one-sided `t` on diversity ratio | `p_value` | mean(eff_n) = mean(1/HHI) |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | none — descriptive | — | event-period Herfindahl-Hirschman index (HHI) |
| [`mfe_mae`][factrix.metrics.mfe_mae.mfe_mae] | none — descriptive | — | MFE_p50 / \|MAE_p75\| |
| [`oos_decay`][factrix.metrics.oos_decay.oos_decay] | none — descriptive | — | survival = \|mean_oos\| / \|mean_is\| |
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | OLS `t` on α | `p_value` | spanning α |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | none — selection meta | — | (NaN; results in metadata) |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | Mann-Kendall `tau` on the index | `p_value` | Theil-Sen slope |
| [`predictive_beta`][factrix.metrics.predictive_beta.predictive_beta] | Newey-West HAC `t` on single-asset predictive slope | `p_value` | predictive beta |
| [`common_beta`][factrix.metrics.common_beta.common_beta] | cross-asset `t` on per-asset β | `p_value` | mean(β) |
| [`common_beta_sign_consistency`][factrix.metrics.common_beta.common_beta_sign_consistency] | none — descriptive | — | max(p, 1-p) on sign fraction |
| [`common_beta_r_squared`][factrix.metrics.common_beta.common_beta_r_squared] | none — descriptive | — | mean(R²) |
| [`common_asymmetry`][factrix.metrics.common_asymmetry.common_asymmetry] | Wald F (NW HAC, finite-sample) on slope sum / equality | `p_value` | β_long + β_short |
| [`common_quantile_spread`][factrix.metrics.common_quantile.common_quantile_spread] | Wald F (NW HAC, finite-sample) on bucket β contrast | `p_value` | top − bottom bucket β |
| [`rank_turnover`][factrix.metrics.tradability.rank_turnover] | none — descriptive | — | 1 − mean(rank-AC) |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | none — descriptive | — | replaced fraction |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | none — descriptive | — | breakeven one-way cost (bps) |
| [`net_spread`][factrix.metrics.tradability.net_spread] | none — descriptive | — | net spread (per-period return) |

## Per-metric schemas

### `ic` family (`factrix.metrics.ic`)

#### `ic`

- *primary*: `p_value` — `t`-test on the per-period IC series (non-overlapping stride with stride `forward_periods` by default, or Newey-West HAC if configured).
- *descriptive*: `n_periods` (the sample the `value` / `stat` / `p_value` describe — the strided subsample under `NonOverlapping`, the full series under `NeweyWest` / `StationaryBootstrap`; equals `n_obs`), `n_periods_full` and `mean_ic_full` (the full per-period series, for reference), `forward_periods`, `tie_ratio` (median across periods), `min_assets_per_period` / `warn_assets_per_period` when the upstream IC series carries per-period asset counts, `stat_type` (`"t"`), `h0` (`"mu=0"`), `method`.
- *warning*: `WarningCode.FEW_ASSETS` when retained per-period IC cross-sections are below `MIN_IC_ASSETS_WARN`.
- *short-circuit*: `reason` `insufficient_ic_periods` (too few periods) carries `min_required`; `insufficient_ic_assets` (every cross-section below `MIN_IC_ASSETS_HARD`, so no per-period IC survived — common on one-valid-pair panels) carries `min_assets_required`.

#### `ic_ir`

Descriptive metric — `MetricResult.stat` is `None` and no `p_value`
is emitted.

- *descriptive*: `mean_ic`, `std_ic`, `n_periods`, `tie_ratio`, `min_assets_per_period` / `warn_assets_per_period` when the upstream IC series carries per-period asset counts.
- *warning*: `WarningCode.FEW_ASSETS` when retained per-period IC cross-sections are below `MIN_IC_ASSETS_WARN`.

### `fm_beta` family (`factrix.metrics.fm_beta`)

#### `fm_beta` (emits `MetricResult.name = "fm_beta"`)

- *primary*: `p_value` — NW HAC `t` on per-period λ. With
  `is_estimated_factor=True` the Shanken EIV correction is applied
  post-hoc and the corrected `p_value` replaces the raw value.
- *secondary-test* (conditional, Shanken applied):
  `p_value_uncorrected`, `stat_uncorrected`.
- *descriptive*: `n_periods`, `newey_west_lags`, `forward_periods`,
  `is_estimated_factor`, `warning_codes` (conditional),
  `min_assets_per_period` / `warn_assets_per_period` when the upstream
  FM beta series carries per-period asset counts.
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

#### `fm_beta_sign_consistency`

Descriptive; no test.

- *descriptive*: `expected_sign`, `n_periods`,
  `min_assets_per_period` / `warn_assets_per_period` when the upstream
  FM beta series carries per-period asset counts.

### `caar` family (`factrix.metrics.caar`)

#### `caar`

- *primary*: `p_value` — non-overlapping `t` on per-event-period CAAR.
  `value`, `stat`, `p_value` and `n_obs` all describe the event-spaced
  subsample (`n_obs_axis = "events"`, the shared event-battery token).
- **`value` is per period.** It is `CAR / h`, inherited from
  `compute_forward_return`'s `/ forward_periods` normalisation, not the
  cumulative abnormal return the name suggests. Multiply by
  `forward_periods` for the cumulative quantity.
- *descriptive*: `n_event_periods` (number of periods with an event),
  `total_events` (underlying events behind the portfolio),
  `n_event_periods_sampled`, `mean_caar_full` / `n_event_periods_full`
  (the full per-period series, for reference),
  `n_event_periods_dropped_non_finite` (null / NaN `caar` periods dropped
  before spacing), `n_events_dropped_non_finite` (events with a non-finite
  return or factor dropped by `compute_caar`),
  `n_events_overlapping` / `n_events_sampled` (removed by, and surviving,
  the non-overlap spacing pass; a non-zero removal fires
  `EVENT_WINDOW_OVERLAP`), `n_events_dropped_no_estimation_window`,
  `abnormal_return_model` and `estimation_window_event_share` (carried up
  from `compute_caar`; the share of each tested event's estimation window
  that lies inside other events' forward windows — above
  `ESTIMATION_WINDOW_EVENT_SHARE_WARN` the mean-adjusted model's abnormal
  returns are negatively correlated and the test is conservative, which
  `ESTIMATION_WINDOW_CONTAMINATED` records; measured 4.7 / 2.3 / 5.0% size at
  h = 1 / 5 / 21 on 20 assets, 0.0% on one asset at h = 21),
  `warning_codes` (conditional, e.g. `FEW_EVENTS`).

#### `bmp_z`

Boehmer-Musumeci-Poulsen standardised-abnormal-return cross-sectional
`z` test, with optional Kolari-Pynnönen clustering adjustment.

- *primary*: `p_value`.
- *descriptive*: `n_events`, `n_event_periods` (distinct event periods —
  the effective sample once events cluster; `FEW_EVENTS` fires on it when
  below `MIN_EVENTS_WARN` and events share periods, and on the raw event
  count when it is below `MIN_EVENTS_WARN × forward_periods`),
  `n_events_overlapping` / `n_events_sampled` (removed by, and surviving,
  the per-asset non-overlap spacing pass; a non-zero removal fires
  `EVENT_WINDOW_OVERLAP`), `abnormal_return_model` / `estimation_window` /
  `estimation_window_source` / `estimation_window_lag` (the abnormal-return
  model behind the SAR numerator: `mean_adjusted` on the one-bar returns of
  `price` over the same window as the vol, lag `0`, or on lagged
  `forward_return` rows without `price`; `market_adjusted_supplied` when
  the panel carries `abnormal_return`), `estimation_window_event_share`
  (share of the tested events' estimation windows inside other events'
  forward windows; above `ESTIMATION_WINDOW_EVENT_SHARE_WARN` the
  mean-adjusted `z` is conservative and `ESTIMATION_WINDOW_CONTAMINATED`
  fires — measured 4.3 / 3.7 / 0.3% size at h = 1 / 5 / 21 on 20 assets,
  nominal 5%; the model is also not robust to heavy skew at h > 1, 18% at
  h = 5 on a lognormal null, where `corrado_rank` holds 4.3%), `n_dropped`
  (= `n_dropped_no_vol` + `n_dropped_non_finite_return`), `std_sar`,
  `estimation_window`, `include_prediction_error_variance`,
  `vol_source` (`"price"` or `"forward_return"`), `vol_estimation_lag`
  (rows the fallback std is lagged so its window ends before the event;
  `0` on the price path).
- *descriptive* (conditional, KP applied): `kolari_pynnonen_r` (one-way
  ANOVA ICC(1)), `kolari_pynnonen_n_eff`, `kolari_pynnonen_r_source`,
  `kolari_pynnonen_applied`, `kolari_pynnonen_scaling` (the design-effect
  deflator `1/sqrt(1+(n_eff-1)r)`), `stat_uncorrected`.

### `corrado` (`factrix.metrics.corrado_rank`)

#### `corrado_rank` (emits `MetricResult.name = "corrado_rank"`)

- *primary*: `p_value` — Corrado nonparametric rank `z`.
- *sample*: `n_event_periods` — distinct event periods surviving the
  per-asset non-overlap spacing pass, and the sample behind
  `stat` / `p_value` / `n_obs` (axis `events`). Same-period events are
  averaged into one observation before the test, so the event period is the
  unit of inference and within-period clustering lands in the denominator.
- *descriptive*: `n_events` (raw event rows — `n_event_periods` times the mean
  events per period, **not** the test's sample size), `events_per_period_mean`,
  `events_per_period_max` (clustering profile; read them together with
  `clustering_hhi`), `n_total_obs` (finite return cells behind the ranks),
  `n_events_dropped_non_finite`, `n_events_dropped_no_estimation_window`
  (events whose asset had too little history for the abnormal-return
  estimate), `abnormal_return_model` (`"mean_adjusted"` or
  `"market_adjusted_supplied"`), `estimation_window`, `estimation_window_lag`,
  `estimation_window_source`, `estimation_window_event_share` (see `bmp_z`;
  `ESTIMATION_WINDOW_CONTAMINATED` above the advisory share),
  `forward_periods` (the spacing stride),
  `n_events_overlapping` / `n_events_sampled` (removed by, and surviving,
  the spacing pass; a non-zero removal fires `EVENT_WINDOW_OVERLAP`).

### `positive_rate` (`factrix.metrics.positive_rate`)

#### `positive_rate`

`MetricResult.stat` is the binomial hit count
(`stat_type="binomial_hits"`); the `p_value` is the exact two-sided
binomial test at every `n`.

- *primary*: `p_value` — exact binomial test on non-overlapping wins
  (stride `forward_periods`).
- *descriptive*: `n_hits`. The trial count is the period-axis drop-stat
  `n_periods_out` (the surviving non-overlapping observations).

### `directional_hit_rate` (`factrix.metrics.directional_hit_rate`)

#### `directional_hit_rate`

Small-N robust sibling of `positive_rate`. `MetricResult.value` is the
directional hit rate (sign-agreement fraction); `stat` is the
Pesaran-Timmermann `z` statistic (`stat_type="z"`), tested one-sided.

- *primary*: `p_value` — one-sided Pesaran-Timmermann test conditioning
  on the marginal up/down frequencies of prediction and realisation.
- *descriptive*: `p_correct` (realised hit rate), `p_expected`
  (hit rate under directional independence), `p_up_pred` (fraction of
  positive predictions), `p_up_real` (fraction of positive realisations),
  `kolari_pynnonen_r` (within-period ICC of the sign-hit indicator, `None`
  on a single-asset series), `kolari_pynnonen_n_eff` (mean assets-per-period),
  `kolari_pynnonen_applied` (whether the Kolari-Pynnönen deflation fired).
- *descriptive* (conditional, adjustment applied): `stat_uncorrected`
  (the raw `S_n` before the cross-sectional-correlation deflation).

### `directional_pair_accuracy` (`factrix.metrics.directional_pair_accuracy`)

#### `directional_pair_accuracy`

Descriptive small-N ordering diagnostic. `MetricResult.value` is pooled
comparable-pair accuracy. `p_value` and `stat` are `None` because same-period
asset pairs are not treated as independent Bernoulli trials.

- *descriptive*: `method`, `n_pairs`, `n_raw_pairs`, `n_periods`,
  `n_correct_pairs`, `n_incorrect_pairs`, `factor_tie_pairs`,
  `return_tie_pairs`, `both_tie_pairs`, `dropped_pairs`,
  `dropped_rows_null`, `pooled_accuracy`, `mean_per_date_accuracy`,
  `mean_pairs_per_period`, `min_pairs_per_period`, `max_pairs_per_period`,
  `tie_epsilon`.
- *warning*: `WarningCode.FEW_ORDERING_PAIRS` when comparable pairs sit below
  `MIN_PAIR_ACCURACY_PAIRS_WARN` but clear the hard floor.
- *short-circuit*: `reason` `insufficient_ordering_pairs` carries
  `min_required` on the pairs axis; `no_factor_column` and `no_return_column`
  name missing inputs.

### `event_quality` (`factrix.metrics.event_quality`)

#### `event_hit_rate`

Same shape as `positive_rate` (exact binomial, `stat` = hit count).

- *primary*: `p_value` — a generalised sign test (Cowan 1992): the null is
  `sign_base_rate`, the frequency with which a *signed* hit happens on the
  non-event rows, not 0.5. `sign_base_rate_up` is the unsigned share of
  positive abnormal returns; a long event hits with that probability under
  the null and a short event with its complement, so `sign_base_rate` is the
  mixture weighted by the share of tested events on each side. Exact binomial
  when events do not share periods, clustered normal on the hit indicator when
  they do (`stat_type` switches from `binomial_hits` to `z`, `method` names
  which ran, and `EVENT_CLUSTERING_ADJUSTED` is the record of the switch).
  `h0` carries the null actually tested.
- *descriptive*: `sign_base_rate`, `sign_base_rate_up`,
  `sign_base_rate_source` (`non_event_rows`, or `assumed_symmetric` when
  there are too few non-event rows to estimate it), `n_base_rate_rows`.
- *descriptive*: `kolari_pynnonen_r` / `kolari_pynnonen_n_eff` /
  `kolari_pynnonen_r_source` / `kolari_pynnonen_applied` /
  `kolari_pynnonen_scaling` / `stat_uncorrected` (the within-period clustering
  estimate and the deflator),
  `n_events` (events surviving the non-overlap spacing pass —
  the binomial `n`), `n_hits`, `n_events_dropped_non_finite`
  (events with a non-finite return / factor, excluded from `n`),
  `n_events_dropped_no_estimation_window` (events whose asset had too little
  history for the abnormal-return estimate), `abnormal_return_model` /
  `estimation_window` / `estimation_window_source` / `estimation_window_lag`
  / `estimation_window_event_share` (above
  `ESTIMATION_WINDOW_EVENT_SHARE_WARN` the mean-adjusted null is
  conservative and `ESTIMATION_WINDOW_CONTAMINATED` fires: 6.0 / 1.7 / 1.7%
  size at h = 1 / 5 / 21 on 20 assets, nominal 5%),
  `n_events_overlapping` / `n_events_sampled` (removed by, and surviving,
  the spacing pass; a non-zero removal fires `EVENT_WINDOW_OVERLAP`).

#### `event_ic`

- *primary*: `p_value` — from the Fisher `z` of the Spearman ρ between
  `|factor|` and the signed abnormal return, using the
  Fieller-Hartley-Pearson Spearman SE `1.06/sqrt(n-3)` (not the Pearson
  `1/sqrt(n-3)`) and deflated for same-period clustering of the per-event rank
  score. `stat` and `p_value` therefore come from one approximation rather than
  two.
- *descriptive*: `n_events` (post-spacing), `n_events_dropped_non_finite`,
  `n_events_dropped_no_estimation_window`, `abnormal_return_model` /
  `estimation_window` / `estimation_window_lag`, `sign_base_rate_up` /
  `sign_base_rate_source` / `n_base_rate_rows` (carried by the shared event
  filter; only `event_hit_rate` tests against them),
  `n_events_overlapping` / `n_events_sampled`, `kolari_pynnonen_r` /
  `kolari_pynnonen_n_eff` / `kolari_pynnonen_r_source` /
  `kolari_pynnonen_applied` (plus `kolari_pynnonen_scaling` /
  `stat_uncorrected` when the deflator applied).

`MetricResult.stat = None` and the short-circuit `reason` is set to
`"not_applicable_discrete_signal"` when the signal lacks magnitude
variance (e.g. binary {-1, +1}).

#### `event_skewness`

- *primary* (conditional, N ≥ 20): `p_value` — D'Agostino skew `z`.
- *descriptive*: `n_events` (post-spacing), `n_events_dropped_non_finite`,
  `n_events_dropped_no_estimation_window`, `abnormal_return_model` /
  `estimation_window` / `estimation_window_lag`, `sign_base_rate_up` /
  `sign_base_rate_source` / `n_base_rate_rows`,
  `n_events_overlapping` / `n_events_sampled`.

When `n_events < 20`, `MetricResult.stat = None` and `p_value` / `stat_type`
/ `h0` / `method` are omitted — the metric reports the Fisher
skewness in `value` only.

#### `profit_factor`

Descriptive; no test.

- *descriptive*: `total_gains`, `total_losses`, `n_events`, `n_wins`,
  `n_losses`, `no_gains`, `no_losses`, `profit_factor_status`,
  `n_events_dropped_non_finite`, `n_events_dropped_no_estimation_window`,
  `abnormal_return_model` / `estimation_window` / `estimation_window_lag`,
  `sign_base_rate_up` / `sign_base_rate_source` / `n_base_rate_rows`.
  `profit_factor_status` is `"finite"` for ordinary gain/loss samples,
  `"unbounded_no_losses"` when positive gains have no offsetting losses
  (`value = inf`), and `"undefined_no_gains_or_losses"` when both gross gains
  and gross losses are zero (`value = NaN`).

#### `signal_density`

Per-asset event frequency; descriptive (the period-axis analogue
is `clustering_hhi`).

- *descriptive*: `n_events_total`, `n_assets_with_events`,
  `mean_events_per_asset`, `mean_bars_between_events`.

### `event_horizon` (`factrix.metrics.event_horizon`)

#### `event_around_return`

Pre/post-event return profile; descriptive.

- *descriptive*: `n_events` (distinct `(date, asset)` events behind the
  curve — also `n_obs`, axis `events`; one event contributes one row per
  offset, so this is not the row count), `per_offset` (dict
  `offset → {mean, se, t, median, p25, p75, hit_rate, n}`, all measured as
  **excess over** `baseline_bar_return`), `baseline_bar_return` (the panel's
  unconditional mean single-bar return, subtracted so a trending asset does
  not read as leaky), `leakage_null_scale` (`≈ 0.8 × mean se` — what the
  headline is worth under *no* leakage, since `E|x̄| > 0` always and shrinks
  as events accumulate), `interpretation`. `reason` is set to
  `no_pre_event_offset_with_enough_events` and `value` is `NaN` when no
  negative offset cleared the 5-event floor — `0.0` there was the *best*
  possible score for a quantity never computed.
- `p_value` is `None` — no hypothesis test runs, and no offset carries a
  `p`: the headline `value` is the pre-event leakage score and per-horizon
  `hit_rate` is a raw fraction of positive signed returns. Offsets overlap
  each other within an event and, unlike the event significance tests, this
  metric applies no non-overlap sampling across events, so the curve is a
  shape to read rather than a test to interpret.

### `monotonicity` (`factrix.metrics.monotonicity`)

#### `monotonicity`

`MetricResult.value` and `MetricResult.stat` both carry the
Patton-Timmermann (2010) MR statistic `mr_min_diff` — `J = min_i mean_t Δ_{i,t}`,
the smallest average adjacent bucket-return difference, in return units.
`p_value` is its stationary-bootstrap empirical p under
`H₀: min_i E[Δ_i] ≤ 0` ("the relation is *not* monotone in the declared
direction"), one-sided (`alternative="greater"`).

The headline used to be `mean |Spearman|` with a cross-asset `t` on the
signed series. That statistic has a large null floor that moves with
`n_groups` — measured 0.67 / 0.42 / 0.27 at `n_groups` = 3 / 5 / 10 on
panels where H₀ holds by construction, because `E|ρ| > 0` by Jensen — so a
reader took the noise floor for MR evidence. The MR test's own rejection
frequency on the same panels is 5.0% / 5.0% / 4.0% at a nominal 5%.

- *primary*: `p_value` — bootstrap p for the MR test.
- *MR detail*: `mr_min_diff`, `mr_adjacent_diffs` (every `Δ̄_i`, so the
  binding step is visible), `mr_direction`, `n_bootstrap`,
  `bootstrap_seed` (resolved and reported when not supplied, so an
  unseeded run is still reproducible after the fact).
- *descriptive Spearman shape*: `mean_abs_spearman` (magnitude, ≥ 0),
  `mean_signed` (direction consistency), `signed_spearman_t`,
  `signed_spearman_p_value`. A high magnitude with a near-zero signed mean
  still says the factor sorts returns but flips sign across dates.
- *descriptive*: `n_valid_periods`, `n_groups`, `tie_ratio`, `tie_policy`.

### `quantile` (`factrix.metrics.quantile`)

#### `quantile_spread`

- *primary*: `p_value` — non-overlapping `t`-test on the
  (top − bottom) spread series (Newey-West HAC on the full series under
  `inference=NEWEY_WEST`). Small cross-sections
  (`median_cross_section < MIN_ASSETS_WARN`) attach `few_assets` and
  change nothing else; see the shared small-N note below.
- *secondary-test*: `long_alpha`, `long_stat`, `long_p_value` —
  long-leg attribution (mean excess and `t` / p-value), on
  `n_periods_long_leg` observations.
- *secondary-test*: `short_alpha`, `short_stat`, `short_p_value`,
  `short_significance` — short-leg attribution, on
  `n_periods_short_leg` observations.
- *descriptive*: `n_periods` (**the sample the headline test ran on**,
  equal to `n_obs`: the strided series under `NON_OVERLAPPING`, the full
  overlapping series under `NEWEY_WEST`), `n_periods_strided` (the
  non-overlap count, always present), `median_cross_section` (median
  per-period count of finite factor values — what the small-N switch
  reads), `tie_ratio`, `tie_policy`, `method`.
- *descriptive*: `n_periods_in`, `n_periods_out`, `n_dropped`,
  `drop_rate`, `drop_reason` — the null/NaN-drop bookkeeping on the
  **strided** spread series (`n_periods_in` also appears on the
  no-surviving-periods short-circuit).
- *descriptive* (conditional, no-signal): `signal_status`
  (`"no_signal_zero_variance_factor"`) when the factor has observations
  but no cross-sectional variation. This is a valid `p_value = 1.0`
  result, not a short-circuit `reason`.

#### `quantile_spread_vw`

Value-weighted variant. Same metadata shape as `quantile_spread` —
including `median_cross_section`, `n_periods_strided` and whatever the
selected inference member contributes — plus a `weights_lagged` flag
indicating whether the weighting input was lagged before the join
(descriptive). It takes the same `inference=` knob off the same
allowlist as `quantile_spread`, so the equal-weighted / value-weighted
pair is tested the same way on the same date set; under `NEWEY_WEST` the
VW leg gives up the first date, whose lagged weight does not exist.
It also carries the same thin-cross-section diagnostics — `few_assets`
on the median per-period finite-factor count and `thin_quantile_groups`
off the shared bucket threshold. Both were previously absent: the metric
whose purpose is a capacity / robustness cross-check reported clean on a
panel whose legs held a single name each. This includes the conditional
no-signal `signal_status` (`"no_signal_zero_variance_factor"`, a valid
`p_value = 1.0` result) when the factor has no cross-sectional variation.

### `k_spread` (`factrix.metrics.k_spread`)

#### `k_spread`

Fixed-K (top-K − bottom-K) long-short spread; the small-N sibling of
`quantile_spread`.

- *primary*: `p_value` — non-overlapping `t`-test on the spread
  series (`method` records the inference member that ran).
- *descriptive*: `k` (names per leg), `tie_ratio` / `tie_policy` (the
  same tie diagnostics `quantile_spread` and `monotonicity` report —
  the leg ranking used to be hard-coded `"ordinal"` with no tie ratio at
  all, so a discrete signal's legs were filled by row order among tied
  names and the arbitrary split was reported as a spread),
  `cross_sectional_dispersion`
  (mean per-period cross-sectional return std), `top_return`,
  `bottom_return`, `n_periods` (**the sample the headline test ran on**,
  equal to `n_obs`: strided under `NON_OVERLAPPING`, full overlapping
  under `NEWEY_WEST`), `n_periods_strided`, `median_cross_section`
  (median per-period count of usable names — what the small-N switch
  reads), `method`. The `k`-too-large short-circuit reports
  `max_assets_per_date`; the no-surviving-periods short-circuit reports
  `n_periods_in`.
- *descriptive* (conditional, no-signal): `signal_status`
  (`"no_signal_zero_variance_factor"`) when the factor has observations
  but no cross-sectional variation. This is a valid `p_value = 1.0`
  result, not a short-circuit `reason`.

#### Shared small-N note

Both `quantile_spread` and `k_spread` attach `few_assets` when the
**median per-period** cross-section (`median_cross_section`) is below
`MIN_ASSETS_WARN` — how many names back a bucket mean is a per-period
quantity, so a rotating universe with many lifetime `asset_id`s but few
names quoted at a time still counts as thin. The warning is advisory; the
headline test does not change. An earlier automatic switch to a
block-bootstrap CI in this regime was removed after measurement: the
bootstrap p rejected 8–20% at a nominal 5% against the `t`'s 7–9%, and
its heavy-tail rationale had the size direction backwards (the `t` is
size-robust to heavy tails; the small-`n` bootstrap is not).

### `concentration` (`factrix.metrics.concentration`)

#### `top_concentration`

`H₀: ratio ≥ 0.5` (one-sided). Tests whether the top-bucket
diversity ratio (effective-n / n_top, derived from HHI) falls
*below* the 0.5 threshold — i.e. concentration risk.

- *primary*: `p_value` — one-sided `t`.
- *descriptive*: `mean_n_top`, `ratio_eff_to_total`, `tie_ratio`,
  `weight_by`, `q_top` (requested top fraction; per period the bucket is
  the `max(1, floor(n · q_top))` highest finite factor values),
  `n_top_members_selected` / `n_top_members_dropped` ((date, asset)
  pairs the cutoff selected, and how many of those were excluded from
  both the HHI and `n_top` for a non-finite weight),
  `warning_codes` (conditional).

### `clustering` (`factrix.metrics.clustering_hhi`)

#### `clustering_hhi` (emits `MetricResult.name = "clustering_hhi"`)

Descriptive; three concentration measures on different axes. The HHI itself
is invariant to how many assets fire per date and is bounded below by `1/D`,
so it cannot answer "do my events cluster?" on its own — read it with the two
companions.

- *descriptive*: `n_events`, `n_event_periods`, `effective_n_periods`
  (`1/HHI`), `hhi_normalized` (`(HHI - 1/D)/(1 - 1/D)`; `0` when every event
  date carries the same count, **including** under perfect cross-sectional
  clustering), `events_per_period_mean` (Kish effective cluster size — the
  cross-sectional axis, and the same `n_eff` the Kolari-Pynnönen deflator
  consumes), `max_events_per_period`, `share_events_in_bursts` (share of
  events whose same-asset predecessor sits within `cluster_window` periods on
  the full panel calendar — the temporal axis), `cluster_window`.

### `mfe_mae` (`factrix.metrics.mfe_mae`)

#### `mfe_mae` (emits `MetricResult.name = "mfe_mae"`)

Descriptive; no test.

- *descriptive*: `mfe_p50`, `mae_p25` (MAE is a signed non-positive
  excursion, so the *worst* quartile is the 25th percentile),
  `mfe_mae_ratio` (= `mfe_p50 / |mae_p25|`), `mfe_mae_ratio_status`
  (`finite`; `unbounded_no_adverse_excursion` with `value = inf` when the
  events never traded against entry — the best outcome, which must not share
  a score with the worst; `undefined_no_excursion` with `value = NaN` when
  neither excursion exists), `n_events`.
- *descriptive* (conditional, when σ-normalised inputs available):
  `mfe_z_p50`, `mae_z_p25`, `mfe_mae_ratio_z`, `mfe_mae_ratio_z_status`,
  `n_events_z`.
- `p_value` is `None` — descriptive metric, no hypothesis test.
### `oos` (`factrix.metrics.oos_decay`)

#### `oos_decay` (emits `MetricResult.name = "oos_decay"`)

`MetricResult.stat = None`; rank-based PASS/VETO gate, no formal
hypothesis test.

- *descriptive*: `status` (`"PASS"` / `"VETOED"`), `sign_flipped`,
  `is_ratio`, `mean_is`, `mean_oos`, `survival_threshold`.

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

Stepwise selection meta-metric; descriptive `MetricResult` with
`value` = count of surviving (selected) factors, `p_value = None`, and
`stat = None`. Per-candidate `t`-stats are *not* valid for inference
(selection bias).

- *descriptive*: `selected_factors` (list of `SpanningResult`),
  `eliminated_factors`, `all_candidates`,
  `t_stats_inference_invalid` (always `True`).

### `trend` (`factrix.metrics.trend`)

#### `ic_trend`

Theil-Sen median slope on the IC series for magnitude; significance from
the Mann-Kendall test on the same ranks. `MetricResult.stat` is Kendall's
`tau` between the sequence index and the series. A constant
series has no rank ordering to test: `stat` / `p_value` are `None` with
`degenerate_variance`, `value` (the zero slope) is kept.

Two departures from textbook Mann-Kendall, both because its null variance
assumes serially independent observations and the default input (a
per-period IC over h-period forward returns) is MA(h−1):

- the series is sub-sampled to non-overlapping observations at stride
  `forward_periods` before anything is computed, so `value` is a slope
  *per sampled step* and `n_obs` is the survivor count;
- `p_value` comes from the Hamed-Rao (1998) variance-inflated normal
  statistic on the survivors, not from `scipy.stats.kendalltau`'s iid p.

Measured null rejection at nominal 5% on an overlapping series: raw
Mann-Kendall 0.37–0.68, strided + Hamed-Rao 0.02–0.05. Residual
persistence that striding cannot remove (an AR factor at `h = 1`) still
leaves the test oversized and carries `serial_correlation_detected`.

- *primary*: `p_value` — Hamed-Rao corrected Mann-Kendall trend
  significance.
- *descriptive*: `n_periods` (survivors after striding), `n_periods_raw`,
  `stride`, `variance_inflation` (Hamed-Rao `n/n*`), `residual_autocorr`,
  `ci_low`, `ci_high`, `ci_excludes_zero`, `intercept`.
- *descriptive* (conditional, augmented Dickey-Fuller (ADF) run): `adf_stat`, `adf_p`,
  `unit_root_suspected` — a suspected unit root now also raises
  `persistent_regressor`.

### `predictive_beta` (`factrix.metrics.predictive_beta`)

#### `predictive_beta`

Single-asset dense predictive regression. `MetricResult.value` is the
slope in `forward_return ~ factor`; `MetricResult.stat` is the
Newey-West HAC `t` statistic for `H0: beta = 0`.

- *primary*: `p_value` — two-sided HAC slope test.
- *descriptive*: `n_periods`, `n_periods_effective`
  (`n_periods // forward_periods` — the non-overlapping observations the
  short-sample gate reads), `residual_lag1_autocorr`, `newey_west_lags`,
  `forward_periods`, `alpha`, `r_squared`, `factor_std`, `adf_stat`, `adf_p`,
  `adf_threshold`, `unit_root_suspected`.
- *warning*: `WarningCode.PERSISTENT_REGRESSOR` when the ADF p-value exceeds
  `adf_threshold`; the HAC slope is still returned, but the predictive
  regression may carry persistent-regressor bias. The flag is a unit-root
  verdict on the regressor, **not** a size guarantee: a long sample gives ADF
  the power to reject the unit root and silences the flag while the test stays
  oversized (measured 14% at a nominal 5% for `T = 2500`, `h = 21` under a
  Stambaugh design where the flag fired on 0% of draws).
- *warning*: `WarningCode.SERIAL_CORRELATION_DETECTED` when the regression
  residuals' lag-1 autocorrelation exceeds `PERSISTENT_SERIES_AUTOCORR`.
- *warning*: `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` when
  `n_periods_effective` is below `MIN_PERIODS_WARN`.
- *short-circuit*: `reason` `insufficient_predictive_periods`,
  `degenerate_factor_variance`, `no_factor_column`, or
  `no_return_column`.

### `common_beta` (`factrix.metrics.common_beta`)

#### `common_beta`

- *primary*: `p_value` — `t` on the cross-asset mean of the per-asset OLS
  β, with the calendar-time SE: `SE² = V_EW + τ̂²/N`, where `V_EW` is the
  Newey-West(`h−1`) variance of the equal-weight portfolio's slope on the
  factor (the mean beta is that slope on a rectangular panel) and `τ̂²` is
  the cross-sectional beta variance in excess of the per-asset estimation
  noise. The textbook iid `std(β)/√N` understates the SE without bound when
  assets share a residual component (44.8% size at N=8, ρ=0.5); the
  Kolari-Pynnönen factor it briefly used instead had zero power once the
  true betas were dispersed.
- *descriptive*: `n_assets`, `beta_std`, `median_beta`,
  `calendar_time_se_applied`, and — when it applied — `ew_portfolio_beta`
  (the equal-weight portfolio slope, equal to `value` on a rectangular
  panel), `ew_portfolio_beta_se`, `ew_portfolio_periods` (dates behind it),
  `beta_dispersion_excess` (`τ̂²`), `dof` (Welch-Satterthwaite df across the
  two variance components) and `stat_uncorrected` (the iid `t`).
  `calendar_time_se_source` records `unavailable_hand_built_frame` (no
  panel behind the frame) or `too_few_shared_periods`; the iid `t` is then
  the reported statistic.
- *warning*: `WarningCode.FEW_ASSETS` below `MIN_ASSETS_WARN`.

#### `common_beta_profile`

Descriptive; no test.

- *descriptive*: `n_assets`, `n_positive_beta`, `n_negative_beta`,
  `n_neutral_beta`, `positive_beta_mean`, `negative_beta_mean`,
  `abs_beta_mean`, `beta_std`, `positive_minus_negative_beta_spread`,
  `neutral_epsilon`, `method`.
- *descriptive* (conditional, one-sided profile): `spread_status` =
  `"requires_positive_and_negative_betas"` when there is no positive/negative
  split to summarize.

#### `common_beta_r_squared`

Descriptive; no test.

- *descriptive*: `n_assets`, `median_r_squared`, `min_r_squared`,
  `max_r_squared`.

#### `common_beta_sign_consistency`

Descriptive symmetric consistency — `value ∈ [0.5, 1.0]`.

- *descriptive*: `n_assets`, `fraction_positive`.

### `common_asymmetry` (`factrix.metrics.common_asymmetry`)

#### `common_asymmetry`

Two complementary methods:

- **Method A** (always): Wald F (finite-sample `F_{r, T−k}`) on
  `H₀: β_long + β_short = 0` with NW HAC SE.
- **Method B** (conditional, ≥ 2 distinct values per side):
  Wald F (finite-sample `F_{r, T−k}`) on `H₀: β_pos = β_neg`.

- *primary*: `p_value` — Method A. `stat` / `p_value` are `None` with
  `degenerate_variance` when the contrast's HAC variance collapses (e.g.
  a constant return); `value` is kept.
- *secondary-test* (conditional, Method B ran):
  `method_b`, `stat_type_method_b`, `beta_pos`, `beta_neg`,
  `p_wald_slopes`.
- *descriptive*: `beta_long`, `beta_short`, `abs_short_over_long`,
  `n_pos`, `n_neg`, `n_zero`, `n_periods`, `nw_lags_used`,
  `method_b_skipped` (conditional), `intercept` (conditional),
  `beta_zero` (conditional).

### `common_quantile` (`factrix.metrics.common_quantile`)

#### `common_quantile_spread`

- *primary*: `p_value` — Wald F (NW HAC, finite-sample `F_{r, T−k}`) on
  `H₀: β_top = β_bottom` from an OLS fit on bucket dummies. `stat` /
  `p_value` are `None` with `degenerate_variance` when the contrast's HAC
  variance collapses (e.g. identical bucket means every period); `value`
  is kept.
- *secondary-test*: `spearman_rho`, `spearman_p` — small-sample
  Spearman of (bucket-idx, mean-return) for monotonicity diagnostic.
- *descriptive*: `n_groups`, `n_periods`, `n_distinct_factor`,
  `nw_lags_used`, `buckets` (list of `{idx, mean_return, n}`).

### `tradability` (`factrix.metrics.tradability`)

All four are descriptive — `MetricResult.stat = None` and no
`p_value` is emitted. They feed cost/benefit arithmetic, not
inference.

#### `rank_turnover`

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
- `MetricResult.p_value = 1.0` — conservative scalar default for callers
  reading the field directly (descriptive short-circuits use `None`).
  `multi_factor.bhy` drops `insufficient_*` placeholders from the test
  family rather than carrying them as rejected.
- Optional diagnostic keys naming what was missing or under-spec:
  `min_required`, `min_required_per_asset`, `min_required_per_regime`,
  `missing_column`, `std_u`, `hint`, `n_distinct`. Each is
  descriptive — emitted only on the short-circuit branch that
  needed it; consumers should branch on `reason` before reading.

The auxiliary `metadata` keys listed in the per-metric subsections
above are *not* present on the short-circuit path.
