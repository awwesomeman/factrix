| Module | Cell scope | Aggregation order | Inference SE |
|---|---|---|---|
| [`metrics.caar`][factrix.metrics.caar] | `(*, SPARSE, PANEL)` | event_time | t (hac) |
| [`metrics.clustering_hhi`][factrix.metrics.clustering_hhi] | `(*, SPARSE, PANEL)` | cs_snapshot | descriptive (none) |
| [`metrics.concentration`][factrix.metrics.concentration] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (ols) |
| [`metrics.corrado_rank`][factrix.metrics.corrado_rank] | `(*, SPARSE, PANEL)` | event_time | rank (built_in) |
| [`metrics.directional_hit_rate`][factrix.metrics.directional_hit_rate] | `(*, DENSE, *)` | ts_only | t (built_in) |
| [`metrics.event_horizon`][factrix.metrics.event_horizon] | `(*, SPARSE, PANEL)` | event_time | binomial (built_in) |
| [`metrics.event_quality`][factrix.metrics.event_quality] | `(*, SPARSE, PANEL)` | event_time | rank (built_in) |
| [`metrics.fm_beta`][factrix.metrics.fm_beta] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (hac) |
| [`metrics.hit_rate`][factrix.metrics.hit_rate] | `(*, DENSE, TIMESERIES)` | ts_only | binomial (built_in) |
| [`metrics.ic`][factrix.metrics.ic] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (ols) |
| [`metrics.ic`][factrix.metrics.ic] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (hac) |
| [`metrics.k_spread`][factrix.metrics.k_spread] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (ols) |
| [`metrics.mfe_mae`][factrix.metrics.mfe_mae] | `(*, SPARSE, PANEL)` | event_time | descriptive (none) |
| [`metrics.monotonicity`][factrix.metrics.monotonicity] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (ols) |
| [`metrics.oos_decay`][factrix.metrics.oos_decay] | `(*, DENSE, TIMESERIES)` | ts_only | descriptive (none) |
| [`metrics.quantile`][factrix.metrics.quantile] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | t (ols) |
| [`metrics.spanning`][factrix.metrics.spanning] | `factor-return-series consumer (post-PANEL pipeline)` | ts_only | t (hac) |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts | descriptive (none) |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, DENSE, PANEL)` | ts_only | descriptive (none) |
| [`metrics.trend`][factrix.metrics.trend] | `(*, DENSE, TIMESERIES)` | ts_only | rank (built_in) |
| [`metrics.ts_asymmetry`][factrix.metrics.ts_asymmetry] | `(COMMON, DENSE, PANEL)` | cs_then_ts | chi2 (hac) |
| [`metrics.ts_beta`][factrix.metrics.ts_beta] | `(COMMON, DENSE, PANEL)` | ts_then_cs | t (ols) |
| [`metrics.ts_quantile`][factrix.metrics.ts_quantile] | `(COMMON, DENSE, PANEL)` | cs_then_ts | chi2 (hac) |
