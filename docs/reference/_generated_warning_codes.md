| WarningCode | Trigger / meaning |
|---|---|
| `unreliable_se_short_periods` | n_periods is below the WARN floor (~30); NW HAC SE may be biased. Reused across panel time-series guards (MIN_PERIODS_WARN) and primitive inference (MIN_FM_PERIODS_WARN); both default to 30. |
| `event_window_overlap` | Adjacent events sit within forward_periods; AR windows overlap. |
| `persistent_regressor` | ADF p > 0.10 on the continuous factor; β may carry Stambaugh bias. |
| `serial_correlation_detected` | Ljung-Box p < 0.05 on residuals; NW lag may be under-set. |
| `few_assets` | PANEL cross-asset t-test with n_assets < MIN_ASSETS_WARN (30); df=n_assets-1 inflates t_crit relative to the asymptotic 1.96 (≈4.30 at n_assets=3, +119%; 5–15% near 30). Severity scales with n_assets — read the n_assets metadata. |
| `sparse_common_few_events` | (COMMON, SPARSE, PANEL) broadcast dummy has MIN_BROADCAST_EVENTS_HARD ≤ n_events < MIN_BROADCAST_EVENTS_WARN (5..19); per-asset β estimable but cross-event averaging too thin for asymptotic t. |
| `sparse_magnitude_weighted` | Sparse factor column is mixed-sign and not a clean ±1 ternary; statistic is magnitude-weighted (Sefcik-Thompson) rather than textbook MacKinlay signed CAAR — apply .sign() before calling for sign-flip semantics. |
| `few_events` | CAAR significance test with MIN_EVENTS_HARD ≤ n_event_periods < MIN_EVENTS_WARN (4..29). caar is an equal-weight calendar-time portfolio across event periods, so this counts the number of periods with an event, not events; a sub-30 series is power-thin for the asymptotic t-distribution — read borderline p-values cautiously. |
| `borderline_portfolio_periods` | top_concentration with MIN_PORTFOLIO_PERIODS_HARD ≤ n_periods < MIN_PORTFOLIO_PERIODS_WARN (3..19); one-sided t-test on the per-date diversification ratio is returned but df=n-1 inflates t_crit relative to the asymptotic cutoff. |
| `rect_kernel_negative_variance` | Rectangular-kernel HAC variance-of-mean came out negative (no PSD guarantee, Andrews 1991); clamped to 0 → SE=0, t=0, p=1.0. Fires only on short / mildly anti-correlated samples. |
| `upstream_unavailable` | DAG-executor consumer skipped because an upstream producer short-circuited. The downstream MetricResult carries metadata['upstream'] / ['upstream_reason'] for the original cause. |
| `cross_factor_density_mismatch` | Factor columns carry inconsistent FactorDensity (dense and sparse mixed). |
| `cross_factor_scope_mismatch` | Factor columns carry inconsistent FactorScope (individual and common mixed). |
| `excessive_period_drops` | An upstream PANEL→SERIES primitive dropped more than DROP_RATE_WARN_THRESHOLD of dates at its cross-sectional filter; the metric was computed on a shortened sample. Exact counts are in MetricResult.metadata (n_periods_in / n_periods_out / dropped_periods / drop_rate / drop_reason). |
| `excessive_asset_drops` | An upstream primitive dropped more than DROP_RATE_WARN_THRESHOLD of assets at its per-asset filter (e.g. compute_ts_betas dropping assets with insufficient history or zero factor variance); the cross-asset aggregate was computed on a shortened sample. Exact counts are in MetricResult.metadata (n_assets_in / n_assets_out / dropped_assets / drop_rate / drop_reason). |
