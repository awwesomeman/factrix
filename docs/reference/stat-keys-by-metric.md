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
| [`ic`][factrix.metrics.ic.ic] | `t` on per-date information coefficient (IC) series (non-overlapping default, Newey-West HAC if configured) | `p_value` | mean(IC) |
| [`ic_ir`][factrix.metrics.ic.ic_ir] | none — descriptive | — | mean(IC) / std(IC) |
| [`fm_beta`][factrix.metrics.fm_beta.fm_beta] | NW HAC `t` on per-date λ | `p_value` | mean(β) |
| [`pooled_beta`][factrix.metrics.fm_beta.pooled_beta] | clustered ordinary least squares (OLS) `t` (or `None` if G < 3) | `p_value` | pooled β |
| [`fm_beta_sign_consistency`][factrix.metrics.fm_beta.fm_beta_sign_consistency] | none — descriptive | — | fraction with expected sign |
| [`caar`][factrix.metrics.caar.caar] | non-overlapping `t` on event-date CAAR | `p_value` | mean(CAAR) |
| [`bmp_z`][factrix.metrics.caar.bmp_z] | BMP cross-sectional `z` on SAR | `p_value` | mean(SAR) |
| [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank] | nonparametric rank `z` | `p_value` | mean(U × sign(factor)) |
| [`positive_rate`][factrix.metrics.positive_rate.positive_rate] | binomial test (or normal `z`) | `p_value` | hit rate ∈ [0, 1] |
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
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | one-sided `t` on diversity ratio | `p_value` | mean(eff_n) = mean(1/HHI) |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | none — descriptive | — | event-date Herfindahl-Hirschman index (HHI) |
| [`mfe_mae`][factrix.metrics.mfe_mae.mfe_mae] | none — descriptive | — | MFE_p50 / \|MAE_p75\| |
| [`oos_decay`][factrix.metrics.oos_decay.oos_decay] | none — descriptive | — | survival = \|mean_oos\| / \|mean_is\| |
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | OLS `t` on α | `p_value` | spanning α |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | none — selection meta | — | (NaN; results in metadata) |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | Theil-Sen slope `t` (CI-based) | `p_value` | Theil-Sen slope |
| [`predictive_beta`][factrix.metrics.predictive_beta.predictive_beta] | Newey-West HAC `t` on single-asset predictive slope | `p_value` | predictive beta |
| [`common_beta`][factrix.metrics.common_beta.common_beta] | cross-asset `t` on per-asset β | `p_value` | mean(β) |
| [`common_beta_sign_consistency`][factrix.metrics.common_beta.common_beta_sign_consistency] | none — descriptive | — | max(p, 1-p) on sign fraction |
| [`common_beta_r_squared`][factrix.metrics.common_beta.common_beta_r_squared] | none — descriptive | — | mean(R²) |
| [`common_asymmetry`][factrix.metrics.common_asymmetry.common_asymmetry] | Wald F (NW HAC, finite-sample) on slope sum / equality | `p_value` | β_long + β_short |
| [`common_quantile_spread`][factrix.metrics.common_quantile.common_quantile_spread] | Wald F (NW HAC, finite-sample) on bucket β contrast | `p_value` | top − bottom bucket β |
| [`rank_turnover`][factrix.metrics.tradability.rank_turnover] | none — descriptive | — | 1 − mean(rank-AC) |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | none — descriptive | — | replaced fraction |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | none — descriptive | — | breakeven single-leg cost (bps) |
| [`net_spread`][factrix.metrics.tradability.net_spread] | none — descriptive | — | net spread (per-period return) |

## Per-metric schemas

### `ic` family (`factrix.metrics.ic`)

#### `ic`

- *primary*: `p_value` — `t`-test on the per-date IC series (non-overlapping stride with stride `forward_periods` by default, or Newey-West HAC if configured).
- *descriptive*: `n_periods`, `forward_periods`, `tie_ratio` (median across dates), `min_assets_per_period` / `warn_assets_per_period` when the upstream IC series carries per-date asset counts, `stat_type` (`"t"`), `h0` (`"mu=0"`), `method`.
- *warning*: `WarningCode.FEW_ASSETS` when retained per-date IC cross-sections are below `MIN_IC_ASSETS_WARN`; suppressed under `evaluate(..., expect_few_assets=True)`, which stamps `few_assets_expected` (conditional — the thin regime engaged under a declared `evaluate(..., expect_few_assets=True)` study; replaces the warning, not the record).
- *short-circuit*: `reason` `insufficient_ic_periods` (too few dates) carries `min_required`; `insufficient_ic_assets` (every cross-section below `MIN_IC_ASSETS_HARD`, so no per-date IC survived — common on one-valid-pair panels) carries `min_assets_required`.

#### `ic_ir`

Descriptive metric — `MetricResult.stat` is `None` and no `p_value`
is emitted.

- *descriptive*: `mean_ic`, `std_ic`, `n_periods`, `tie_ratio`, `min_assets_per_period` / `warn_assets_per_period` when the upstream IC series carries per-date asset counts.
- *warning*: `WarningCode.FEW_ASSETS` when retained per-date IC cross-sections are below `MIN_IC_ASSETS_WARN`; suppressed under `evaluate(..., expect_few_assets=True)`, which stamps `few_assets_expected` (conditional — the thin regime engaged under a declared `evaluate(..., expect_few_assets=True)` study; replaces the warning, not the record).

### `fm_beta` family (`factrix.metrics.fm_beta`)

#### `fm_beta` (emits `MetricResult.name = "fm_beta"`)

- *primary*: `p_value` — NW HAC `t` on per-date λ. With
  `is_estimated_factor=True` the Shanken EIV correction is applied
  post-hoc and the corrected `p_value` replaces the raw value.
- *secondary-test* (conditional, Shanken applied):
  `p_value_uncorrected`, `stat_uncorrected`.
- *descriptive*: `n_periods`, `newey_west_lags`, `forward_periods`,
  `is_estimated_factor`, `warning_codes` (conditional),
  `min_assets_per_period` / `warn_assets_per_period` when the upstream
  FM beta series carries per-date asset counts; `few_assets_expected` (conditional — the thin regime engaged under a declared `evaluate(..., expect_few_assets=True)` study; replaces the warning, not the record).
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
  FM beta series carries per-date asset counts; `few_assets_expected` (conditional — the thin regime engaged under a declared `evaluate(..., expect_few_assets=True)` study; replaces the warning, not the record).

### `caar` family (`factrix.metrics.caar`)

#### `caar`

- *primary*: `p_value` — non-overlapping `t` on per-event-date CAAR.
- *descriptive*: `n_event_periods` (number of periods with an event),
  `total_events` (underlying events behind the portfolio),
  `n_event_periods_sampled`,
  `warning_codes` (conditional, e.g. `FEW_EVENTS`).

#### `bmp_z`

Boehmer-Musumeci-Poulsen standardised-abnormal-return cross-sectional
`z` test, with optional Kolari-Pynnönen clustering adjustment.

- *primary*: `p_value`.
- *descriptive*: `n_events`, `n_dropped`, `std_sar`,
  `estimation_window`, `include_prediction_error_variance`,
  `vol_source` (`"price"` or `"forward_return"`), `vol_estimation_lag`
  (rows the fallback std is lagged so its window ends before the event;
  `0` on the price path).
- *descriptive* (conditional, KP applied): `kolari_pynnonen_r`,
  `kolari_pynnonen_n_eff`, `kolari_pynnonen_r_source`,
  `kolari_pynnonen_applied`, `kolari_pynnonen_scaling`,
  `stat_uncorrected`.

### `corrado` (`factrix.metrics.corrado_rank`)

#### `corrado_rank` (emits `MetricResult.name = "corrado_rank"`)

- *primary*: `p_value` — Corrado nonparametric rank `z`.
- *descriptive*: `n_events`, `n_total_obs`.

### `positive_rate` (`factrix.metrics.positive_rate`)

#### `positive_rate`

`MetricResult.stat` is the binomial hit count when the exact branch
runs, the normal `z` when the approximation branch runs;
`stat_type` discriminates (`"binomial_hits"` vs `"z"`).

- *primary*: `p_value` — binomial / normal-approximation test on
  non-overlapping wins (stride `forward_periods`).
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
  `kolari_pynnonen_r` (within-date ICC of the sign-hit indicator, `None`
  on a single-asset series), `kolari_pynnonen_n_eff` (mean assets-per-date),
  `kolari_pynnonen_applied` (whether the Kolari-Pynnönen deflation fired).
- *descriptive* (conditional, adjustment applied): `stat_uncorrected`
  (the raw `S_n` before the cross-sectional-correlation deflation).

### `directional_pair_accuracy` (`factrix.metrics.directional_pair_accuracy`)

#### `directional_pair_accuracy`

Descriptive small-N ordering diagnostic. `MetricResult.value` is pooled
comparable-pair accuracy. `p_value` and `stat` are `None` because same-date
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

Same shape as `positive_rate` (binomial / normal-approx branches).

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

When `n_events < 20`, `MetricResult.stat = None` and `p_value` / `stat_type`
/ `h0` / `method` are omitted — the metric reports the Fisher
skewness in `value` only.

#### `profit_factor`

Descriptive; no test.

- *descriptive*: `total_gains`, `total_losses`, `n_events`, `n_wins`,
  `n_losses`, `no_gains`, `no_losses`, `profit_factor_status`.
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

- *descriptive*: `per_offset` (dict `offset → {mean, median, p25, p75,
  hit_rate, n}`), `interpretation`.
- `p_value` is `None` — no hypothesis test runs; the headline `value` is
  the pre-event leakage score, and per-horizon `hit_rate` is a raw
  fraction.

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
- *descriptive* (conditional, no-signal): `signal_status`
  (`"no_signal_zero_variance_factor"`) when the factor has observations
  but no cross-sectional variation. This is a valid `p_value = 1.0`
  result, not a short-circuit `reason`.

#### `quantile_spread_vw`

Value-weighted variant. Same metadata shape as `quantile_spread`
plus a `weights_lagged` flag indicating whether the weighting input
was lagged before the join (descriptive). This includes the conditional
no-signal `signal_status` (`"no_signal_zero_variance_factor"`, a valid
`p_value = 1.0` result) when the factor has no cross-sectional variation.

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
- *descriptive* (conditional, no-signal): `signal_status`
  (`"no_signal_zero_variance_factor"`) when the factor has observations
  but no cross-sectional variation. This is a valid `p_value = 1.0`
  result, not a short-circuit `reason`.

#### Shared small-N significance keys

Both `quantile_spread` and `k_spread` switch the headline test to a
block-bootstrap CI when `n_assets < MIN_ASSETS_WARN`. In that branch
they additionally emit `p_value_t` (the parametric `t` p-value kept
for reference), `bootstrap_block_length`, `bootstrap_n_resamples`,
and `bootstrap_seed`. The switch is **not** silent: the single
cross-section code (`few_assets`) is attached to `warning_codes`,
so the method change surfaces as a `Warning` on the result. Under a
declared `evaluate(..., expect_few_assets=True)` study the code is
not attached — the branch instead stamps `few_assets_expected` so
the acknowledged switch stays readable from `metadata` (`method`
still names the bootstrap path).

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

- *descriptive*: `n_events`, `n_event_periods`, `effective_n_periods`,
  `hhi_normalized`, `cluster_window`.

### `mfe_mae` (`factrix.metrics.mfe_mae`)

#### `mfe_mae` (emits `MetricResult.name = "mfe_mae"`)

Descriptive; no test.

- *descriptive*: `mfe_p50`, `mae_p75`, `mfe_mae_ratio`, `n_events`.
- *descriptive* (conditional, when σ-normalised inputs available):
  `mfe_z_p50`, `mae_z_p75`, `mfe_mae_ratio_z`, `n_events_z`.
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

Theil-Sen median slope on the IC series. The reported `MetricResult.stat`
is the slope-`t` derived from the rank-based confidence interval.

- *primary*: `p_value` — slope significance from the Theil-Sen CI.
- *descriptive*: `n_periods`, `ci_low`, `ci_high`,
  `ci_excludes_zero`, `intercept`.
- *descriptive* (conditional, augmented Dickey-Fuller (ADF) run): `adf_stat`, `adf_p`,
  `unit_root_suspected`.

### `predictive_beta` (`factrix.metrics.predictive_beta`)

#### `predictive_beta`

Single-asset dense predictive regression. `MetricResult.value` is the
slope in `forward_return ~ factor`; `MetricResult.stat` is the
Newey-West HAC `t` statistic for `H0: beta = 0`.

- *primary*: `p_value` — two-sided HAC slope test.
- *descriptive*: `n_periods`, `newey_west_lags`, `forward_periods`,
  `alpha`, `r_squared`, `factor_std`, `adf_stat`, `adf_p`,
  `adf_threshold`, `unit_root_suspected`.
- *warning*: `WarningCode.PERSISTENT_REGRESSOR` when the ADF p-value exceeds
  `adf_threshold`; the HAC slope is still returned, but the predictive
  regression may carry persistent-regressor bias.
- *short-circuit*: `reason` `insufficient_predictive_periods`,
  `degenerate_factor_variance`, `no_factor_column`, or
  `no_return_column`.

### `common_beta` (`factrix.metrics.common_beta`)

#### `common_beta`

- *primary*: `p_value` — cross-asset `t` on the per-asset OLS β
  distribution.
- *descriptive*: `n_assets`, `beta_std`, `median_beta`.

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

- *primary*: `p_value` — Method A.
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
  `H₀: β_top = β_bottom` from an OLS fit on bucket dummies.
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
