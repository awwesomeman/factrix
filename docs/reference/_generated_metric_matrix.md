| Module | Cell scope | Aggregation order |
|---|---|---|
| [`metrics.caar`][factrix.metrics.caar] | `(*, SPARSE, *)` | event_time |
| [`metrics.clustering_hhi`][factrix.metrics.clustering_hhi] | `(*, SPARSE, PANEL)` | cs_snapshot |
| [`metrics.common_asymmetry`][factrix.metrics.common_asymmetry] | `(COMMON, DENSE, PANEL)` | cs_then_ts |
| [`metrics.common_beta`][factrix.metrics.common_beta] | `(COMMON, DENSE, PANEL)` | ts_then_cs |
| [`metrics.common_quantile`][factrix.metrics.common_quantile] | `(COMMON, DENSE, PANEL)` | cs_then_ts |
| [`metrics.concentration`][factrix.metrics.concentration] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.corrado_rank`][factrix.metrics.corrado_rank] | `(*, SPARSE, *)` | event_time |
| [`metrics.directional_hit_rate`][factrix.metrics.directional_hit_rate] | `(*, DENSE, *)` | ts_only |
| [`metrics.directional_pair_accuracy`][factrix.metrics.directional_pair_accuracy] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.event_horizon`][factrix.metrics.event_horizon] | `(*, SPARSE, *)` | event_time |
| [`metrics.event_quality`][factrix.metrics.event_quality] | `(*, SPARSE, *)` | event_time |
| [`metrics.fm_beta`][factrix.metrics.fm_beta] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.ic`][factrix.metrics.ic] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.k_spread`][factrix.metrics.k_spread] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.mfe_mae`][factrix.metrics.mfe_mae] | `(*, SPARSE, *)` | event_time |
| [`metrics.monotonicity`][factrix.metrics.monotonicity] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.oos_decay`][factrix.metrics.oos_decay] | `(INDIVIDUAL, DENSE, PANEL)` | ts_only |
| [`metrics.positive_rate`][factrix.metrics.positive_rate] | `(INDIVIDUAL, DENSE, PANEL)` | ts_only |
| [`metrics.predictive_beta`][factrix.metrics.predictive_beta] | `(*, DENSE, TIMESERIES)` | ts_only |
| [`metrics.quantile`][factrix.metrics.quantile] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.spanning`][factrix.metrics.spanning] | `factor-return-series consumer (post-PANEL pipeline)` | ts_only |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, DENSE, PANEL)` | cs_then_ts |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, DENSE, PANEL)` | ts_only |
| [`metrics.trend`][factrix.metrics.trend] | `(INDIVIDUAL, DENSE, PANEL)` | ts_only |
