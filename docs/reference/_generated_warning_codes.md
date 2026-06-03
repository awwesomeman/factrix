| WarningCode | Trigger / meaning |
|---|---|
| `unreliable_se_short_periods` | n_periods is below the WARN floor (~30); NW HAC SE may be biased. Reused across panel time-series guards (MIN_PERIODS_WARN) and primitive inference (MIN_FM_PERIODS_WARN); both default to 30. |
| `event_window_overlap` | Adjacent events sit within forward_periods; AR windows overlap. |
| `persistent_regressor` | ADF p > 0.10 on the continuous factor; β may carry Stambaugh bias. |
| `serial_correlation_detected` | Ljung-Box p < 0.05 on residuals; NW lag may be under-set. |
| `small_cross_section_n` | PANEL cross-asset t-test with n_assets < MIN_ASSETS (10); df=n_assets-1 too low — t_crit at n_assets=3 ≈ 4.30 (+119% vs asymptotic 1.96). |
| `borderline_cross_section_n` | PANEL cross-asset t-test with MIN_ASSETS ≤ n_assets < MIN_ASSETS_WARN (10..29); residual t_crit inflation 5–15% — read borderline p-values cautiously. |
| `sparse_common_few_events` | (COMMON, SPARSE, PANEL) broadcast dummy has MIN_BROADCAST_EVENTS_HARD ≤ n_events < MIN_BROADCAST_EVENTS_WARN (5..19); per-asset β estimable but cross-event averaging too thin for asymptotic t. |
| `sparse_magnitude_weighted` | Sparse factor column is mixed-sign and not a clean ±1 ternary; statistic is magnitude-weighted (Sefcik-Thompson) rather than textbook MacKinlay signed CAAR — apply .sign() before calling for sign-flip semantics. |
| `few_events` | CAAR significance test with MIN_EVENTS_HARD ≤ n_event_dates < MIN_EVENTS_WARN (4..29); t-stat returned but Brown-Warner (1985) convention treats sub-30 events as power-thin for the asymptotic t-distribution — read borderline p-values cautiously. |
| `borderline_portfolio_periods` | top_concentration with MIN_PORTFOLIO_PERIODS_HARD ≤ n_periods < MIN_PORTFOLIO_PERIODS_WARN (3..19); one-sided t-test on the per-date diversification ratio is returned but df=n-1 inflates t_crit relative to the asymptotic cutoff. |
| `rect_kernel_negative_variance` | Rectangular-kernel HAC variance-of-mean came out negative (no PSD guarantee, Andrews 1991); clamped to 0 → SE=0, t=0, p=1.0. Fires only on short / mildly anti-correlated samples. |
| `singular_weight_matrix` | GMM long-run covariance Ŝ was numerically singular; J-statistic was computed via Moore-Penrose pseudo-inverse rather than a true inverse. Fires on rank-deficient or strongly collinear moment matrices. |
| `upstream_unavailable` | DAG-executor consumer skipped because an upstream producer short-circuited. The downstream MetricResult carries metadata['upstream'] / ['upstream_reason'] for the original cause. |
