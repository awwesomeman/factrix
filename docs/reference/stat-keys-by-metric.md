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
metric below.

A `no_*_column` short-circuit is a statement about the **data**, never about
the call: it fires only for a column parameter left at its documented default
(`factor`, `forward_return`, `price`, `market_cap`) that the frame does not
carry. A column the caller *named* — any `*_col` / `*_cols` argument set away
from its default — is validated at the metric gate instead, and a name the
frame lacks raises `UserInputError` with the frame's columns as did-you-mean
candidates rather than returning a NaN envelope. Every `reason` is a fixed
token, so consumers can branch on it.

Cross-slice inference functions
([`slice_pairwise_test`][factrix.slice_pairwise_test] /
[`slice_joint_test`][factrix.slice_joint_test]) are
not listed in the table: their headline output is a DataFrame of
contrasts, not a sidecar to a primary value.

## Cross-metric summary

| Metric | Primary stat (`MetricResult.stat`) | Primary `metadata` key | `value` |
|---|---|---|---|
| [`directional_pair_accuracy`][factrix.metrics.directional_pair_accuracy.directional_pair_accuracy] | none; descriptive | n/a | pooled pairwise ordering accuracy |
| [`common_beta_profile`][factrix.metrics.common_beta.common_beta_profile] | none; descriptive | n/a | positive-minus-negative beta mean spread |
| [`ic`][factrix.metrics.ic.ic] | test on per-period information coefficient (IC) series (non-overlapping `t` default; Newey-West HAC `t` or stationary-bootstrap empirical `p` if configured) | `p_value` | mean(IC) |
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
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | none — descriptive ([the D'Agostino skew test is withdrawn](inference-calibration.md#event-skewness-no-calibrated-test)) | — | Fisher skewness |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | none — descriptive | — | gains / \|losses\| |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | none — descriptive | — | mean bars per event |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | none — descriptive | — | mean leakage score |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | Patton-Timmermann (2010) MR (stationary-bootstrap p) | `p_value` | `mr_min_diff` (min adjacent bucket-return difference) |
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | non-overlapping `t` on top-bottom spread (NW HAC under `NEWEY_WEST`, empirical p under `STATIONARY_BOOTSTRAP`) | `p_value` | mean(spread) |
| [`k_spread`][factrix.metrics.k_spread.k_spread] | non-overlapping `t` on top-K−bottom-K spread (NW HAC under `NEWEY_WEST`, empirical p under `STATIONARY_BOOTSTRAP`) | `p_value` | mean(spread) |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | non-overlap `t` (default), NW HAC `t`, or bootstrap empirical p on vw spread | `p_value` | mean(vw spread) |
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | descriptive — no test | — | mean(eff_n) = mean(1/HHI) |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | none — descriptive | — | event-period Herfindahl-Hirschman index (HHI) |
| [`mfe_mae`][factrix.metrics.mfe_mae.mfe_mae] | none — descriptive | — | MFE_p50 / \|MAE_p75\| |
| [`oos_decay`][factrix.metrics.oos_decay.oos_decay] | none — descriptive | — | survival = \|mean_oos\| / \|mean_is\| |
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | HAR HAC `t` on α | `p_value` | spanning α |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | none — selection meta | — | (NaN; results in metadata) |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | Mann-Kendall `tau` on the index | `p_value` | Theil-Sen slope |
| [`predictive_beta`][factrix.metrics.predictive_beta.predictive_beta] | Newey-West HAC `t` on single-asset predictive slope | `p_value` | predictive beta |
| [`common_beta`][factrix.metrics.common_beta.common_beta] | cross-asset `t` on per-asset β | `p_value` | mean(β) |
| [`common_beta_sign_consistency`][factrix.metrics.common_beta.common_beta_sign_consistency] | none — descriptive | — | max(p, 1-p) on sign fraction |
| [`common_beta_r_squared`][factrix.metrics.common_beta.common_beta_r_squared] | none — descriptive | — | mean(R²) |
| [`common_asymmetry`][factrix.metrics.common_asymmetry.common_asymmetry] | Wald F (HAR HAC, finite-sample) on slope sum / equality | `p_value` | β_long + β_short |
| [`common_quantile_spread`][factrix.metrics.common_quantile.common_quantile_spread] | Wald F (HAR HAC, finite-sample) on bucket β contrast | `p_value` | top − bottom bucket β |
| [`rank_turnover`][factrix.metrics.tradability.rank_turnover] | none — descriptive | — | 1 − mean(rank-AC) |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | none — descriptive | — | replaced fraction |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | none — descriptive | — | breakeven one-way cost (bps) |
| [`net_spread`][factrix.metrics.tradability.net_spread] | none — descriptive | — | net spread (per-period return) |

### `alternative_requested` — every metric, degenerate path only

A metric that withholds its test on a degenerate sample returns
`p_value=None` and `alternative=None`, so the tail it would have run under
has nowhere left to live on the result. Every such result therefore carries
`metadata["alternative_requested"]`, and **only** such a result does: on a
normal path `MetricResult.alternative` already records it, so a second copy
would read as a degeneracy marker. Metrics that expose no `alternative`
knob record `"two-sided"` here, which is what they ran.

## Per-metric schemas

### `ic` family (`factrix.metrics.ic`)

#### `ic`

- *primary*: `p_value` — test on the per-period IC series, from the configured `inference`: a `t`-test on a non-overlapping stride of `overlap_periods` (default), a Newey-West HAC `t`-test, or a stationary-bootstrap empirical `p`.
- *descriptive*: `n_periods` (the sample the `value` / `stat` / `p_value` describe — the strided subsample under `NonOverlapping`, the full series under `NeweyWest` / `StationaryBootstrap`; equals `n_obs`), `n_periods_full` and `mean_ic_full` (the full per-period series, for reference), `overlap_periods`, `tie_ratio` (median across periods), `min_assets_per_period` / `warn_assets_per_period` when the upstream IC series carries per-period asset counts, `stat_type` (the test actually run: `"t"` under `NonOverlapping` / `NeweyWest`, `"bootstrap-mean"` under `StationaryBootstrap`), `h0` (`"mu=0"`), and `method`.
- *descriptive* (conditional, `NeweyWest`): `newey_west_lags` (resolved Bartlett bandwidth) and `hac_dof` (the effective degrees of freedom the `t` is read against; `None` when the sample is too short to run the kernel).
- *descriptive* (conditional, `StationaryBootstrap`): `n_resamples` and `seed` (the resolved seed, reported even when not supplied, so an unseeded run stays reproducible; `null` when a `numpy.random.Generator` was supplied — that stream is the caller's to reproduce), `p_value_mc_se` (Monte-Carlo SE of the empirical `p`), `block_length` (the resolved Politis-White mean block length) and `studentized`. See [Resampling knobs](statistical-methods.md#resampling-knobs).
- *warning*: `WarningCode.FEW_ASSETS` when retained per-period IC cross-sections are below `MIN_IC_ASSETS_WARN`; `WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED` under `NeweyWest` when the resolved bandwidth exceeds `n_periods / 5`.
- *short-circuit*: `reason` `insufficient_ic_periods` (too few periods) carries `min_required`; `insufficient_ic_assets` (every cross-section below `MIN_IC_ASSETS_HARD`, so no per-period IC survived — common on one-valid-pair panels) carries `min_assets_required`.

#### `ic_ir`

Descriptive metric — `MetricResult.stat` is `None` and no `p_value`
is emitted.

- *descriptive*: `mean_ic`, `std_ic`, `n_periods`, `tie_ratio`, `min_assets_per_period` / `warn_assets_per_period` when the upstream IC series carries per-period asset counts.
- *warning*: `WarningCode.FEW_ASSETS` when retained per-period IC cross-sections are below `MIN_IC_ASSETS_WARN`.
- *degenerate*: a zero IC-series standard deviation returns `value=NaN`
  with `WarningCode.DEGENERATE_VARIANCE`; it is not a hypothesis and does
  not enter multiple-testing families.

### `fm_beta` family (`factrix.metrics.fm_beta`)

#### `fm_beta` (emits `MetricResult.name = "fm_beta"`)

- *primary*: `p_value` — NW HAC `t` on per-period λ. With
  `is_estimated_factor=True` the Shanken EIV correction is applied
  post-hoc and the corrected `p_value` replaces the raw value — unless the
  correction is unavailable, in which case both are withheld (see the
  σ²_f ≈ 0 entry below).
- *secondary-test* (conditional, `is_estimated_factor=True`):
  `p_value_uncorrected`, `stat_uncorrected`. These remain descriptive-only
  diagnostics when the Shanken correction is unavailable.
- *descriptive*: `n_periods`, `newey_west_lags` (the resolved HAR
  bandwidth), `hac_dof` (the effective degrees of freedom the `t` is read
  against), `overlap_periods`,
  `is_estimated_factor`, `warning_codes` (conditional),
  `min_assets_per_period` / `warn_assets_per_period` when the upstream
  FM beta series carries per-period asset counts.
- *warning*: `WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED` when the resolved
  bandwidth exceeds `n_periods / 5`.
- *descriptive* (conditional, Shanken applied): `shanken_c`,
  `shanken_factor_return_var` (the caller-supplied σ²_f). The corrected
  `p_value` is read against the same `hac_dof` as `p_value_uncorrected`,
  so `p_value >= p_value_uncorrected` always.
- *descriptive* (conditional, σ²_f ≈ 0): `shanken_correction` =
  `"unavailable_zero_factor_variance"` — the correction is undefined when
  factor-return variance collapses. The mean beta remains, but `stat` /
  `p_value` are withheld with `WarningCode.DEGENERATE_VARIANCE`; the
  uncorrected test stays in the explicitly named secondary fields only.

#### `pooled_beta` (emits `MetricResult.name = "pooled_beta"`)

- *primary*: `p_value` — single- or two-way clustered OLS `t`. When
  the cluster count G < 3 the test is short-circuited with `stat =
  None`, `value = NaN`, and `p_value = 1.0`: with two clusters the sample
  cannot support a clustered estimate, so the algebraic slope is not
  exposed as a usable pooled beta. A non-PSD two-way covariance is the
  other withheld-inference regime and keeps `value` — see
  `variance_status` below.
- Sample size: `MetricResult.n_obs` (row count entering the test).
- *descriptive*: `overlap_periods` (the injected evaluation-grid horizon) and
  `overlap_adjustment_applied`. The latter is `False` for clustered covariance,
  which does not consume the horizon, and `True` for Driscoll-Kraay HAC.
- *descriptive*: `n_clusters` (one-way) or `n_clusters_a`,
  `n_clusters_b`, `n_clusters_intersection` (two-way).
- *descriptive* (conditional, short-circuit): `reason =
  "insufficient_clusters"`, `n_clusters` (smallest G — first-class
  `n_obs` carries the row count), `min_required` (always 3), plus
  `metric_unavailable` in `warning_codes`.
- *descriptive* (conditional): `variance_status` =
  `"non_psd_two_way_covariance"` when the requested two-way covariance cannot
  support a slope test. The pooled OLS slope remains available, but `stat` /
  `p_value` are withheld with `WarningCode.DEGENERATE_VARIANCE`; factrix does
  not substitute a one-way covariance.
- *descriptive* (Driscoll-Kraay path, `driscoll_kraay=True`):
  `se_method` (`"driscoll_kraay"`), `n_periods` (length of the
  cross-sectional score-sum series), and `driscoll_kraay_lags` (the
  resolved Bartlett bandwidth), `hac_scale` (`T / (T - L - 1)`),
  and `hac_dof` (the scalar HAR reference degrees of freedom). At 30 or more
  periods the DK path emits
  `WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED` when `L > T / 5`; below 30
  it emits the binding `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` without a
  redundant bandwidth echo. It short-circuits to `value = NaN` with
  `reason = "insufficient_periods"` below 3. An explicit
  `driscoll_kraay_lags` replaces the automatic base
  but cannot undercut the overlap floor or exceed the estimability cap.

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
  `EvaluationResult.forward_periods` for the cumulative quantity; do not use
  `overlap_periods`, which can be smaller on a coarse evaluation grid.
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
  `return_scale` (`per_period_simple` on the standard pipeline,
  `supplied_abnormal_return` for a caller-supplied abnormal-return column, or
  `unspecified` for a hand-built CAAR frame), `event_weighting` (`sign`,
  `factor_magnitude`, or `unspecified`), `abnormal_return_model` and
  `estimation_window_event_share` (carried up
  from `compute_caar`; the share of each tested event's estimation window
  that lies inside other events' forward windows — above
  `ESTIMATION_WINDOW_EVENT_SHARE_WARN` the mean-adjusted model's abnormal
  returns are negatively correlated and the test is conservative, which
  `ESTIMATION_WINDOW_CONTAMINATED` records; measured 4.7 / 2.3 / 5.0% size at
  h = 1 / 5 / 21 on 20 assets, 0.0% on one asset at h = 21),
  `warning_codes` (conditional, e.g. `FEW_EVENTS`; `SPARSE_MAGNITUDE_WEIGHTED`
  when the sparse factor is mixed-sign and not a clean ±1 ternary;
  `NON_FINITE_INPUT_DROPPED` when `compute_caar` dropped event rows).

#### `bmp_z`

Boehmer-Musumeci-Poulsen standardised-abnormal-return cross-sectional
`z` test, with optional Kolari-Pynnönen clustering adjustment.

- *primary*: `p_value`.
- *descriptive*: `n_events`, `n_event_periods` (distinct event periods —
  the effective sample once events cluster; `FEW_EVENTS` fires on it when
  below `MIN_EVENTS_WARN` and events share periods, and on the raw event
  count when it is below `MIN_EVENTS_WARN × overlap_periods`),
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
  `clustering_hhi`), `n_pairs_total` (finite return cells behind the ranks),
  `n_events_dropped_non_finite`, `n_events_dropped_no_estimation_window`
  (events whose asset had too little history for the abnormal-return
  estimate), `abnormal_return_model` (`"mean_adjusted"` or
  `"market_adjusted_supplied"`), `estimation_window`, `estimation_window_lag`,
  `estimation_window_source`, `estimation_window_event_share` (see `bmp_z`;
  `ESTIMATION_WINDOW_CONTAMINATED` above the advisory share),
  `overlap_periods` (the spacing stride),
  `n_events_overlapping` / `n_events_sampled` (removed by, and surviving,
  the spacing pass; a non-zero removal fires `EVENT_WINDOW_OVERLAP`).
- *degenerate*: a zero standard deviation of the event-period mean ranks
  returns `value=NaN` with `WarningCode.DEGENERATE_VARIANCE` and no test.

### `positive_rate` (`factrix.metrics.positive_rate`)

#### `positive_rate`

`MetricResult.stat` is the binomial hit count
(`stat_type="binomial_hits"`); the `p_value` is the exact two-sided
binomial test at every `n`.

- *primary*: `p_value` — exact binomial test on non-overlapping wins
  (stride `overlap_periods`).
- *descriptive*: `n_hits`. The trial count is the period-axis drop-stat
  `n_periods_out` (the surviving non-overlapping observations).
- *warning code* (conditional): `serial_correlation_detected` when the
  strided hit indicators remain persistent, violating the independent-trials
  assumption of the exact binomial reference.

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
- *degenerate*: one-signed predictions or realisations that collapse the
  Pesaran-Timmermann variance return `value=NaN` with
  `WarningCode.DEGENERATE_VARIANCE` and no test.

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

Descriptive; no test. `MetricResult.p_value`, `stat` and `alternative` are
`None` at every sample size, and no `stat_type` / `h0` / `method` key is
written.

- *descriptive*: `n_events` (post-spacing), `n_events_dropped_non_finite`,
  `n_events_dropped_no_estimation_window`, `abnormal_return_model` /
  `estimation_window` / `estimation_window_lag`, `sign_base_rate_up` /
  `sign_base_rate_source` / `n_base_rate_rows`,
  `n_events_overlapping` / `n_events_sampled`.

factrix used to publish D'Agostino's skew-test `z` beside the point
estimate whenever `n_events >= 20`. That test has no calibrated pooled form
here for two independent reasons: it assumes the signed abnormal returns
are normal under the null and over-rejects on the excess kurtosis a sampled
event panel carries with no clustering at all (19.0% and 23.3% at a nominal
5%), and it over-rejects again at 30.3% when same-period events share a
shock and a factor sign. The sibling clustering deflation repairs neither —
it moves the first to 17.7% / 22.7% and over-corrects the second to 0.0%.
The size table and both null definitions are in
[inference calibration](inference-calibration.md#event-skewness-has-no-calibrated-test).
Test the direction of an event payoff with `event_hit_rate` / `event_ic` /
`bmp_z`, which stay sized on both.

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
  possible score for a quantity never computed. An observed non-finite or
  non-positive price instead withdraws the entire curve with
  `reason=invalid_price_data`, `per_offset={}`, and
  `WarningCode.METRIC_UNAVAILABLE`; `n_invalid_prices` reports the offending
  row count. Null prices remain valid missing data on a ragged panel.
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
  binding step is visible), `mr_direction`, `n_resamples`,
  `seed` (resolved and reported when not supplied, so an
  unseeded run is still reproducible after the fact; `null` when a
  `numpy.random.Generator` was supplied), `p_value_mc_se`
  (Monte-Carlo SE of the empirical p, `sqrt(p(1-p)/n_resamples)` — how far
  the p would move on a re-run with a different seed, ~0.7pp at the default
  `n_resamples=999` and p near 0.05; `n_resamples` is floored at 199 — see
  [Resampling knobs](statistical-methods.md#resampling-knobs)).
- *descriptive Spearman shape*: `mean_abs_spearman` (magnitude, ≥ 0),
  `mean_signed` (direction consistency), `signed_spearman_t`,
  `signed_spearman_p_value`. A high magnitude with a near-zero signed mean
  still says the factor sorts returns but flips sign across dates.
- *descriptive*: `n_valid_periods`, `n_groups`, `tie_ratio`, `tie_policy`.

### `quantile` (`factrix.metrics.quantile`)
- `warning_codes` (conditional): `HIGH_TIE_RATIO` under `tie_policy="ordinal"`
  when the median per-period `tie_ratio` exceeds `TIE_RATIO_WARN_THRESHOLD`.

#### `quantile_spread`

- *primary*: `p_value` — non-overlapping `t`-test on the
  (top − bottom) spread series (Newey-West HAC on the full series under
  `inference=NEWEY_WEST`, a block-bootstrap empirical p on the full
  series under `inference=STATIONARY_BOOTSTRAP`). Small cross-sections
  (`median_cross_section < MIN_ASSETS_WARN`) attach `few_assets` and
  change nothing else; see the shared small-N note below.
- *secondary-test*: `long_alpha`, `long_stat`, `long_p_value` —
  long-leg attribution (mean excess and `t` / p-value), on
  `n_periods_long_leg` observations.
- *secondary-test*: `short_alpha`, `short_stat`, `short_p_value`,
  `short_significance` — short-leg attribution, on
  `n_periods_short_leg` observations.
- *secondary-test*: `leg_alternative` — always `"two-sided"` for both leg
  p-values, independent of the headline `alternative`. The headline tests the
  sum `long_alpha + short_alpha`; a directional hypothesis about that sum does
  not imply that each component has the same direction.
- *descriptive*: `n_periods` (**the sample the headline test ran on**,
  equal to `n_obs`: the strided series under `NON_OVERLAPPING`, the full
  overlapping series under `NEWEY_WEST` / `STATIONARY_BOOTSTRAP`),
  `n_periods_strided` (the non-overlap count, always present),
  `median_cross_section` (median per-period count of finite factor values
  behind the `few_assets` diagnostic), `tie_ratio`, `tie_policy`,
  `stat_type` (the test actually run: `"t"` under `NonOverlapping` /
  `NeweyWest`, `"bootstrap-mean"` under `StationaryBootstrap`), `method`.
- *descriptive*: `n_periods_in`, `n_periods_out`, `n_dropped`,
  `drop_rate`, `drop_reason` — the null/NaN-drop bookkeeping on the
  **strided** spread series (`n_periods_in` also appears on the
  no-surviving-periods short-circuit).
- *descriptive* (conditional, no-signal): `signal_status`
  (`"no_signal_zero_variance_factor"`) when the factor has observations
  but no cross-sectional variation. This is a valid `p_value = 1.0`
  result, not a short-circuit `reason`.
- `warning_codes` (conditional): `HIGH_TIE_RATIO` under `tie_policy="ordinal"`
  when the median per-period `tie_ratio` exceeds `TIE_RATIO_WARN_THRESHOLD`;
  `THIN_QUANTILE_GROUPS` when the median cross-section leaves fewer than
  `MIN_GROUP_ASSETS` names per bucket.

#### `quantile_spread_vw`

Value-weighted variant. Same metadata shape as `quantile_spread` —
including `median_cross_section`, `n_periods_strided`, `stat_type` (the test actually run: `"t"` under `NonOverlapping` / `NeweyWest`, `"bootstrap-mean"` under `StationaryBootstrap`)
and whatever the selected inference member contributes — plus a `weights_lagged` flag
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
- `warning_codes` (conditional): as `quantile_spread` — `HIGH_TIE_RATIO`,
  `THIN_QUANTILE_GROUPS`.

#### `k_spread`

Fixed-K (top-K − bottom-K) long-short spread; the small-N sibling of
`quantile_spread`.

- *primary*: `p_value` — non-overlapping `t`-test on the spread
  series (`method` records the inference member that ran, and
  `stat_type` (the test actually run: `"t"` under `NonOverlapping` / `NeweyWest`, `"bootstrap-mean"` under `StationaryBootstrap`) the statistic it reports).
- *descriptive*: `k` (names per leg), `tie_ratio` / `tie_policy` (the
  same tie diagnostics `quantile_spread` and `monotonicity` report —
  the leg ranking used to be hard-coded `"ordinal"` with no tie ratio at
  all, so a discrete signal's legs were filled by row order among tied
  names and the arbitrary split was reported as a spread),
  `cross_sectional_dispersion`
  (mean per-period cross-sectional return std), `top_return`,
  `bottom_return`, `n_periods` (**the sample the headline test ran on**,
  equal to `n_obs`: strided under `NON_OVERLAPPING`, full overlapping
  under `NEWEY_WEST` / `STATIONARY_BOOTSTRAP`), `n_periods_strided`,
  `median_cross_section` (median per-period count of usable names behind
  the `few_assets` diagnostic), `method`. The `k`-too-large short-circuit
  reports `max_assets_per_date`; the no-surviving-periods short-circuit
  reports `n_periods_in`.
- *descriptive* (conditional, no-signal): `signal_status`
  (`"no_signal_zero_variance_factor"`) when the factor has observations
  but no cross-sectional variation. This is a valid `p_value = 1.0`
  result, not a short-circuit `reason`.
- `warning_codes` (conditional): `HIGH_TIE_RATIO` under `tie_policy="ordinal"`
  when the median per-period `tie_ratio` exceeds `TIE_RATIO_WARN_THRESHOLD`.

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

**Descriptive; no hypothesis test.** `p_value`, `stat` and
`alternative` are `None`. The earlier one-sided `t` against
`H₀: ratio ≥ 0.5` is withdrawn: 0.5 sits far inside the null rather than
near it — the null diversification ratio is ~0.91 on a Gaussian factor
with `abs_factor` weights, and the measured rejection rate is 0 of 300
draws at a nominal 5% at both 10 and 48 tested periods, so the p could
not reject at any sample size. See
[inference-calibration reference](inference-calibration.md#coverage-contract).

- *descriptive*: `method`, `mean_n_top`, `ratio_eff_to_total`, `tie_ratio`,
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

- *primary*: `p_value` — `t` on α from the multivariate spanning
  regression. A single restriction, so the HAC SE, the `T/(T−L−1)` scale
  and the reference `ν` all come from `_resolve_scalar_wald_hac`; the
  regression residual count stays available as `_OLSResult.df_resid` but is
  not what the `t` is read against.
- Sample size: `MetricResult.n_obs` (length of the aligned
  candidate-series).
- *warning codes* (conditional): `serial_correlation_detected` (candidate
  spread persistent once strided at `overlap_periods`),
  `unreliable_se_short_periods` (fewer than 10 periods after that stride),
  `hac_bandwidth_ill_conditioned` when `n_periods >= 30` and `L > T / 5`.
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
  `overlap_periods` before anything is computed, so `value` is a slope
  *per sampled step* and `n_obs` is the survivor count;
- `p_value` comes from the Hamed-Rao (1998) variance-inflated normal
  statistic on the survivors, not from `scipy.stats.kendalltau`'s iid p.

Measured null rejection at nominal 5% on an overlapping series: raw
Mann-Kendall 0.37–0.68, strided + Hamed-Rao 0.02–0.05. The stride is
~99% of that fix — in every overlap cell "strided" and "strided +
Hamed-Rao" agree to within 1pp. Hamed-Rao handles residual persistence
*after* striding and is not calibrated on its own (Hamed-Rao alone on an
overlapping series: 0.16–0.32). Residual persistence that striding
cannot remove (an AR factor at `h = 1`) still leaves the test oversized
and carries `serial_correlation_detected`.

The stride costs power, and at `overlap_periods = 21` costs nearly all
of it. Against a drift of 2 sd over the sample, raw Mann-Kendall versus
strided + Hamed-Rao: 0.865 → 0.360 (`T=60, h=5`), 0.971 → 0.713
(`T=120, h=5`), 0.999 → 0.952 (`T=240, h=5`), 0.888 → 0.086
(`T=120, h=21`), 0.951 → 0.425 (`T=240, h=21`). At `h = 21` the metric
needs roughly 240+ periods before it can detect anything.

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
**Stambaugh-bias-corrected** slope in `forward_return ~ factor`;
`MetricResult.stat` is its `t` statistic for `H0: beta = 0`.

OLS on this regression is biased whenever the predictor is persistent
*and* its AR(1) innovation correlates with the return innovation
(Stambaugh 1999) — the classic dividend-yield artefact. The bias is the
**product** of the two, which is why the ADF screen alone cannot detect
it: ADF proxies persistence only. Measured against a true `beta = 0` at
`T = 60, phi = 0.95, rho = -0.9`, plain OLS averaged `+0.076` and
rejected 20.6% at a nominal 5%.

The fix is the Amihud-Hurvich (2004) augmented regression: the
bias-corrected AR(1) residual proxy enters as a second regressor, and
the slope on the predictor is the reduced-bias estimator. Its standard
error carries a generated-regressor term
(`Var_aug + gamma^2 Var(phi_c) c^2`) — without it the proxy absorbs the
correlated part of the return innovation and the test rejects ~50% of
true nulls.

Two departures from Amihud-Hurvich, both factrix choices. (1) The
covariance inside the augmented regression is horizon-dependent: at
`overlap_periods = 1` it is AH's own homoskedastic `s^2 (X'X)^-1` read
against `t_{m-3}`, because that regression is not overlapping and a
Bartlett kernel there is pure downward bias in the SE (it cost 4pp of size
at `T = 60`); at `h > 1` it is the Bartlett HAC covariance read against the
fixed-`b` effective df `_har_dof`, since a single-restriction slope test is
not the `K x K` Wald the narrow bandwidth rule exists for. (2) AH (2004) is
an `h = 1` method — summing the innovation proxy over `t+1..t+h` is a
factrix extension, and the design drops its last `h-1` rows rather than
carrying zero-padded truncated proxy sums.

Measured on 2000 draws per cell with a true `beta = 0`. The `rho = 0` rows
carry no Stambaugh channel at all and are there to separate the correction
from the inference:

| T    | φ    | ρ    | h | bias (OLS → AH)   | size (OLS → AH) |
|------|------|------|---|-------------------|-----------------|
| 60   | 0.50 |  0.0 | 1 | +0.0010 → +0.0012 | 0.086 → 0.043   |
| 60   | 0.90 |  0.0 | 1 | +0.0009 → −0.0007 | 0.091 → 0.050   |
| 60   | 0.95 |  0.0 | 1 | −0.0014 → −0.0022 | 0.102 → 0.055   |
| 60   | 0.99 |  0.0 | 1 | +0.0017 → +0.0015 | 0.084 → 0.050   |
| 60   | 0.95 | −0.9 | 1 | +0.0670 → +0.0114 | 0.183 → 0.075   |
| 120  | 0.95 | −0.9 | 1 | +0.0313 → +0.0026 | 0.125 → 0.083   |
| 240  | 0.95 | −0.9 | 1 | +0.0147 → +0.0007 | 0.088 → 0.062   |
| 60   | 0.50 |  0.0 | 5 | −0.0095 → −0.0137 | 0.144 → 0.121   |
| 60   | 0.90 |  0.0 | 5 | −0.0029 → −0.0115 | 0.214 → 0.145   |
| 120  | 0.50 |  0.0 | 5 | −0.0050 → −0.0059 | 0.107 → 0.097   |
| 240  | 0.95 |  0.0 | 5 | +0.0002 → −0.0000 | 0.130 → 0.102   |
| 60   | 0.95 | −0.9 | 5 | +0.2733 → −0.0031 | 0.370 → 0.058   |
| 120  | 0.95 | −0.9 | 5 | +0.1449 → +0.0058 | 0.262 → 0.068   |
| 1000 | 0.90 |  0.0 | 5 | −0.0005 → +0.0018 | 0.103 → 0.075   |

Read the two blocks separately. At `h = 1` the corrected test is
calibrated (4.3–5.5% at `rho = 0`, 6.2–8.3% in the strongest Stambaugh
cells). At `h > 1` it is not — 7.5–14.5%, present at `rho = 0` for every
`phi`, and plain OLS-NW carries the same excess. That is the
overlapping-regression HAC problem, not the Stambaugh channel; see the
[HAC family calibration](inference-calibration.md#hac-family-calibration).

The correction also costs power where OLS's apparent power was partly its
own bias: at `T = 60, phi = 0.95, rho = -0.9` the corrected test rejects
28.8% of a true alternative against OLS's 88.6% (`T = 120`: 44.4% against
90.7%). At `rho = 0`, where OLS is unbiased, the gap is small (63.2%
against 70.5% at `T = 60`).

Every key names the fit it belongs to. `alpha`, `residual_lag1_autocorr`,
`n_obs`, `n_periods` and `n_periods_effective` describe the **corrected**
fit — the model `value` came from, on the rows it was estimated on.
`beta_ols_uncorrected` and `r_squared_ols_uncorrected` are the
pre-correction OLS reference on the full finite-pair sample
(`n_periods_finite`). There is deliberately no corrected `r_squared`: the
Amihud-Hurvich slope does not minimise the sum of squares, so `1 - SSR/SST`
off its residual can go negative and has no standing in this literature,
which reports `R²` for the OLS regression.

- *primary*: `p_value` — two-sided bias-corrected slope test.
- *primary sample*: `n_obs` — the rows the headline test ran on, which is
  `n_periods_finite - overlap_periods`. The augmented design spends the first
  observation on the AR(1) lag and, at `overlap_periods > 1`, the last
  `overlap_periods - 1` windows on the horizon-summed innovation proxy. If
  that design cannot be formed, the metric short-circuits; its `n_obs` then
  records the finite-pair count available before bailing.
- *descriptive*: `n_periods` (= `n_obs`), `n_periods_finite` (the finite
  `(factor, return)` pairs available before the augmented design trimmed its
  rows), `n_periods_effective` (`n_periods // overlap_periods` — the
  non-overlapping observations the short-sample gate reads),
  `residual_lag1_autocorr` (lag-1 autocorrelation of the reported model's
  residuals, strided at `overlap_periods`), `newey_west_lags`,
  `overlap_periods`, `hac_applied` (whether the headline corrected test used
  HAC), `har_lags` (the headline HAR bandwidth at `overlap_periods > 1`;
  `None` at `h = 1`, where the corrected test is homoskedastic;
  `newey_west_lags` stays the narrow covariance bandwidth resolved for the
  uncorrected OLS reference), `alpha` (the intercept the corrected slope implies
  on the `n_obs` rows), `r_squared_ols_uncorrected`, `factor_std`,
  `adf_stat`, `adf_p`, `adf_threshold`, `unit_root_suspected`.
- *descriptive* (Stambaugh correction): `stambaugh_adjusted` (True on every
  computed result), `beta_ols_uncorrected`, `stambaugh_bias_estimate`
  (`beta_ols_uncorrected - value`), `ar1_phi`,
  `ar1_phi_corrected`, `innovation_corr` (`rho_hat`),
  `stambaugh_bias_channel` (`|rho_hat * phi_corrected|`),
  `ar1_phi_corrected_explosive` (`ar1_phi_corrected >= 1` — the corrected
  AR(1) coefficient is deliberately not clamped, because clamping it
  re-opens the bias the correction exists to close).
- *warning*: `WarningCode.PERSISTENT_REGRESSOR` when the ADF p-value exceeds
  `adf_threshold`, **or** `stambaugh_bias_channel` exceeds `0.7`, **or**
  `ar1_phi_corrected_explosive`. The channel trigger is the one that
  matters: it reads the actual bias channel rather than a unit-root verdict
  on the regressor alone. ADF fired on 1% of runs at `phi = 0.5` where the
  bias is already present, and stayed silent on 62% of biased runs at
  `T = 240`. The threshold is 0.7 rather than the earlier 0.3 because a
  size sweep over `T` in {60, 120, 240} x `phi` in {0.5, 0.9, 0.95, 0.99} x
  `rho` in {-0.5, -0.9} puts every cell with a channel at or below 0.5 at
  4.1–6.6% — calibrated — while cells above 0.8 sit at 5.7–9.1%. At 0.3 the
  code fired on the whole `rho = -0.5` column and read "oversized here"
  where it is not. The code means "the corrected test is itself somewhat
  oversized in this regime", not "beta may carry Stambaugh bias" — the bias
  is corrected. It is about the *regressor*, not overlapping windows.
- *warning*: `WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE` whenever
  `overlap_periods > 1`. The corrected test remains 7.5-14.5% oversized
  across the measured null cells, including independent-regressor designs;
  read the returned p-value against a raised hurdle and compare `h = 1` or
  a genuinely non-overlapping design.
- *warning*: `WarningCode.SERIAL_CORRELATION_DETECTED` when the **reported**
  model's residuals — `y - alpha - value * factor` over the `n_obs` rows,
  strided at `overlap_periods` — have a lag-1 autocorrelation above
  `PERSISTENT_SERIES_AUTOCORR`. Reading the uncorrected OLS residuals here
  flipped the verdict on draws where the two slopes are far apart.
- *warning*: `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` when
  `n_periods_effective` is below `MIN_PERIODS_WARN`.
- *warning*: `WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED` when the scalar HAR
  bandwidth satisfies `L > T / 5` on at least 30 finite periods. Below 30,
  `UNRELIABLE_SE_SHORT_PERIODS` owns the regime without a duplicate warning.
- *degenerate*: zero factor variance returns `value=NaN` with
  `WarningCode.DEGENERATE_VARIANCE` and no test.
- *short-circuit*: `reason` `insufficient_predictive_periods`,
  `no_amihud_hurvich_fit`, `no_factor_column`, or `no_return_column`. The
  `insufficient_predictive_periods` floor is a pre-flight gate on
  `n_periods_finite`, before the augmented design trims its rows.
  `no_amihud_hurvich_fit` retains `beta_ols_uncorrected` as a descriptive
  reference but does not substitute its estimate or inference; it also
  reports `stambaugh_adjusted=False`.

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
  panel behind the frame), `shared_period_count_unavailable`,
  `too_few_shared_periods`, or `invalid_calendar_time_variance`. Only the
  hand-built-frame case reports the iid `t`; producer failures withhold the
  test.
- *warning*: `WarningCode.FEW_ASSETS` below `MIN_ASSETS_WARN`.
- *warning*: `WarningCode.COMMON_BETA_IID_FALLBACK` when a hand-built beta
  table has no calendar-time variance; `WarningCode.DEGENERATE_VARIANCE`
  when producer-supplied calendar-time inference fails.

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

Both are **single-restriction** contrasts, so both take the scalar HAR
reference (`_resolve_scalar_wald_hac`): bandwidth, `T/(T−L−1)` variance
scale and the kernel's fixed-`b` effective `ν`.

- **Method A** (always): Wald F (finite-sample `F_{1, ν}`) on
  `H₀: β_long + β_short = 0` with HAR HAC SE.
- **Method B** (conditional, ≥ 2 distinct values per side):
  Wald F (finite-sample `F_{1, ν}`) on `H₀: β_pos = β_neg`.

- *primary*: `p_value` — Method A. `stat` / `p_value` are `None` with
  `degenerate_variance` when the contrast's HAC variance collapses (e.g.
  a constant return); `value` is kept.
- *secondary-test* (conditional, Method B ran):
  `method_b`, `stat_type_method_b`, `beta_pos`, `beta_neg`,
  `p_wald_slopes`.
- *warning codes* (conditional): `serial_correlation_detected` (per-period
  factor persistent once strided at `overlap_periods`),
  `unreliable_se_short_periods` (fewer than 10 periods after that stride),
  `hac_bandwidth_ill_conditioned` when `n_periods >= 30` and `L > T / 5`.
- *descriptive*: `beta_long`, `beta_short`, `abs_short_over_long`,
  `n_pos`, `n_neg`, `n_zero`, `n_periods`, `newey_west_lags_used`,
  `method_b_skipped` (conditional), `intercept` (conditional),
  `beta_zero` (conditional).

### `common_quantile` (`factrix.metrics.common_quantile`)

#### `common_quantile_spread`

- *primary*: `p_value` — Wald F on `H₀: β_top = β_bottom` from an OLS fit
  on bucket dummies. A single restriction, so it takes the scalar HAR
  reference (`_resolve_scalar_wald_hac`): finite-sample `F_{1, ν}` at the
  kernel's fixed-`b` effective `ν`. `stat` /
  `p_value` are `None` with `degenerate_variance` when the contrast's HAC
  variance collapses (e.g. identical bucket means every period); `value`
  is kept.
- *secondary-test*: `spearman_rho`, `spearman_p` — small-sample
  Spearman of (bucket-idx, mean-return) for monotonicity diagnostic.
- *warning codes* (conditional): `thin_quantile_periods`,
  `serial_correlation_detected` (per-period factor persistent once strided
  at `overlap_periods`), `unreliable_se_short_periods` (fewer than 10
  periods after that stride), `hac_bandwidth_ill_conditioned` when
  `n_periods >= 30` and `L > T / 5`.
- *descriptive*: `n_groups`, `n_periods`, `n_distinct_factor`,
  `newey_west_lags_used`, `buckets` (list of `{idx, mean_return, n}`).

### `tradability` (`factrix.metrics.tradability`)

All four are descriptive — `MetricResult.stat = None` and no
`p_value` is emitted. They feed cost/benefit arithmetic, not
inference.
- `warning_codes` (conditional): `THIN_QUANTILE_PERIODS` when the historical
  buckets average fewer than 5 periods each.

#### `rank_turnover`

- *descriptive*: `mean_rank_autocorrelation`,
  `std_rank_autocorrelation`, `n_pairs`, `overlap_periods`,
  `rebalance_lag`, `quantile`, `n_cross_section_mean`.

#### `notional_turnover`

- *descriptive*: `n_rebalances`, `n_groups`, `overlap_periods`,
  `rebalance_lag`, `mean_top_turnover`, `mean_bottom_turnover`
  (each leg's mean replaced fraction; `value` is their mean —
  `mean_top_turnover` is the matched proxy for an equal-weight top-quantile
  long-only book), `mean_tail_size`, `mean_top_tail_size`,
  `mean_bottom_tail_size`.

Both turnover metrics report two strides. `overlap_periods` is the panel's
evaluation-grid overlap stamp — an inference quantity, injected, never a user
knob. `rebalance_lag` is the stride the metric actually paired consecutive
observations at, counted in evaluation-grid observations; it equals
`overlap_periods` unless the caller passed `rebalance_lag=`.

#### `breakeven_cost`

Scalar-input metric (consumes pre-aggregated scalars rather than a
date-keyed DataFrame).

- *descriptive*: `gross_spread`, `turnover`, `holding_periods`.

#### `net_spread`

Scalar-input metric.

- *descriptive*: `gross_spread`, `cost_drag`, `estimated_cost_bps`,
  `turnover`, `holding_periods`.

`holding_periods` is the cost-amortisation interval: the number of
**underlying return periods** between rebalances, the same unit
`compute_forward_return` normalises `gross_spread` to. It is neither the
evaluation-grid `overlap_periods` nor the turnover metrics' `rebalance_lag`.
Both helpers also record `n_groups` and `pairing_checked` when handed the
producing `MetricResult`s rather than bare floats.

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
