# Standalone metric pipelines

Cross-module index of every module under `factrix/metrics/`. Use this
page to pick the existing aggregation pattern a new metric should
match, or to mechanically check that a candidate metric satisfies the
[NW HAC discipline](statistical-methods.md) the rest of the suite
follows.

The matrix is a curated index, complementing the auto-generated
[API reference](api/datasets.md): the API pages document the call
surface, this page documents the *aggregation pattern* — the order in
which the cross-section step, the time-series step, and inference SE
compose.

## Aggregation vocabulary

- **CS-first** — aggregate cross-section per date first, then aggregate
  the resulting time series.
- **TS-first** — aggregate time-series per asset first, then aggregate
  across assets.
- **TS-only** — single-series time-series operation; no cross-section
  step.
- **Static CS** — single cross-section, no time-axis aggregation.
- **Per-event** — aggregation centred on event dates (per-event-date
  step), then cross-event aggregation.

## Matrix

| Module | Public functions | Cell scope | Aggregation order | Inference SE | Reuses primitives |
|---|---|---|---|---|---|
| [`metrics/caar.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/caar.py) | `compute_caar`, `caar`, `bmp_test` | `(*, SPARSE, *, PANEL)` | per-event | non-overlapping t / z | `_calc_t_stat`, `_p_value_from_t`, `_p_value_from_z`, `_significance_marker`, `_sample_non_overlapping`, `_short_circuit_output` |
| [`metrics/clustering.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/clustering.py) | `clustering_diagnostic` | `(*, SPARSE, *, PANEL)` | static CS | no formal H₀ | `_short_circuit_output` |
| [`metrics/concentration.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/concentration.py) | `top_concentration` | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | across-time t (one-sided H₀: ratio ≥ 0.5) | `_calc_t_stat`, `_p_value_from_t`, `_significance_marker`, `_sample_non_overlapping`, `_short_circuit_output`, `_compute_tie_ratio` |
| [`metrics/corrado.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/corrado.py) | `corrado_rank_test` | `(*, SPARSE, *, PANEL)` | per-event | nonparametric rank | `_calc_t_stat`, `_p_value_from_z`, `_significance_marker`, `_short_circuit_output` |
| [`metrics/event_horizon.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/event_horizon.py) | `compute_event_returns`, `event_around_return`, `multi_horizon_hit_rate` | `(*, SPARSE, *, PANEL)` | per-event | binomial | `_short_circuit_output`, `_significance_marker` |
| [`metrics/event_quality.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/event_quality.py) | `event_hit_rate`, `event_ic`, `profit_factor`, `event_skewness`, `signal_density` | `(*, SPARSE, *, PANEL)` | per-event | binomial / nonparametric rank | `_binomial_two_sided_p`, `_significance_marker`, `_short_circuit_output`, `_signed_car` |
| [`metrics/fama_macbeth.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/fama_macbeth.py) | `compute_fm_betas`, `fama_macbeth`, `pooled_ols`, `beta_sign_consistency` | `(INDIVIDUAL, CONTINUOUS, FM, PANEL)` | CS-first | NW HAC / clustered t | `_newey_west_t_test`, `_p_value_from_t`, `_significance_marker`, `_short_circuit_output` |
| [`metrics/hit_rate.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/hit_rate.py) | `hit_rate` | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | binomial | `_binomial_two_sided_p`, `_significance_marker`, `_short_circuit_output`, `_sample_non_overlapping` |
| [`metrics/ic.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ic.py) | `compute_ic`, `ic`, `ic_newey_west`, `ic_ir`, `regime_ic`, `multi_horizon_ic` | `(INDIVIDUAL, CONTINUOUS, IC, PANEL)` | CS-first | NW HAC / cross-asset t | `_newey_west_t_test`, `_calc_t_stat`, `_p_value_from_t`, `_significance_marker`, `_sample_non_overlapping`, `_short_circuit_output` |
| [`metrics/mfe_mae.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/mfe_mae.py) | `compute_mfe_mae`, `mfe_mae_summary` | `(*, SPARSE, *, PANEL)` | per-event | no formal H₀ | `_short_circuit_output` |
| [`metrics/monotonicity.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/monotonicity.py) | `monotonicity` | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | cross-asset t | `_calc_t_stat`, `_p_value_from_t`, `_significance_marker`, `_sample_non_overlapping`, `_short_circuit_output`, `_assign_quantile_groups`, `_compute_tie_ratio` |
| [`metrics/oos.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/oos.py) | `multi_split_oos_decay` | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | no formal H₀ | `_short_circuit_output` |
| [`metrics/quantile.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/quantile.py) | `compute_spread_series`, `quantile_spread`, `quantile_spread_vw`, `compute_group_returns` | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | cross-asset t | `_calc_t_stat`, `_p_value_from_t`, `_significance_marker`, `_sample_non_overlapping`, `_short_circuit_output`, `_assign_quantile_groups`, `_compute_tie_ratio`, `_lag_within_asset` |
| [`metrics/spanning.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/spanning.py) | `spanning_alpha`, `greedy_forward_selection` | factor-return-series consumer (post-PANEL pipeline) | TS-only | NW HAC / OLS t | `_p_value_from_t`, `_significance_marker`, `_short_circuit_output`, `_ols_alpha` |
| [`metrics/tradability.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/tradability.py) `notional_turnover`, `breakeven_cost`, `net_spread` | `notional_turnover`, `breakeven_cost`, `net_spread` | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first (per-date Q1/Qn membership delta → time mean) | no formal H₀ | `_sample_non_overlapping`, `_short_circuit_output`, `_assign_quantile_groups` |
| [`metrics/tradability.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/tradability.py) `turnover` | `turnover` | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | TS-only (rank autocorrelation across consecutive dates) | no formal H₀ | `_short_circuit_output` |
| [`metrics/trend.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/trend.py) | `ic_trend` | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | Theil-Sen rank-based CI | `_significance_marker`, `_short_circuit_output`, `_adf`, `_p_value_from_t` |
| [`metrics/ts_asymmetry.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ts_asymmetry.py) | `ts_asymmetry` | `(COMMON, CONTINUOUS, *, PANEL)` | CS-first | NW HAC Wald | `_significance_marker`, `_short_circuit_output`, `_aggregate_to_per_date`, `_ols_nw_multivariate`, `_wald_p_linear` |
| [`metrics/ts_beta.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ts_beta.py) | `compute_ts_betas`, `ts_beta`, `mean_r_squared`, `compute_rolling_mean_beta`, `ts_beta_sign_consistency`, `ts_beta_single_asset_fallback` | `(COMMON, CONTINUOUS, *, PANEL)` | TS-first | cross-asset t | `_calc_t_stat`, `_p_value_from_t`, `_significance_marker`, `_short_circuit_output` |
| [`metrics/ts_quantile.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ts_quantile.py) | `ts_quantile_spread` | `(COMMON, CONTINUOUS, *, PANEL)` | CS-first | NW HAC Wald | `_significance_marker`, `_short_circuit_output`, `_aggregate_to_per_date`, `_ols_nw_multivariate`, `_wald_p_linear` |

For per-module formula derivations, read each module's top-level
docstring (linked above); for the underlying paper references and
inference-SE rationale, see [Statistical methods](statistical-methods.md);
for `n_obs` / `n_assets` thresholds per metric, see
[Metric applicability](metric-applicability.md).
