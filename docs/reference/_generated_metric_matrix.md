| Module | Cell scope | Aggregation order | Inference SE |
|---|---|---|---|
| [`metrics.caar`][factrix.metrics.caar] | `(*, SPARSE, *, PANEL)` | per-event | non-overlapping t / z |
| [`metrics.clustering`][factrix.metrics.clustering] | `(*, SPARSE, *, PANEL)` | static-cs | no formal H₀ |
| [`metrics.concentration`][factrix.metrics.concentration] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | cs-first | across-time t (one-sided H₀: ratio ≥ 0.5) |
| [`metrics.corrado`][factrix.metrics.corrado] | `(*, SPARSE, *, PANEL)` | per-event | nonparametric rank |
| [`metrics.event_horizon`][factrix.metrics.event_horizon] | `(*, SPARSE, *, PANEL)` | per-event | binomial |
| [`metrics.event_quality`][factrix.metrics.event_quality] | `(*, SPARSE, *, PANEL)` | per-event | binomial / nonparametric rank |
| [`metrics.fama_macbeth`][factrix.metrics.fama_macbeth] | `(INDIVIDUAL, CONTINUOUS, FM, PANEL)` | cs-first | NW HAC / clustered t |
| [`metrics.hit_rate`][factrix.metrics.hit_rate] | `(*, CONTINUOUS, *, TIMESERIES)` | ts-only | binomial |
| [`metrics.ic`][factrix.metrics.ic] | `(INDIVIDUAL, CONTINUOUS, IC, PANEL)` | cs-first | NW HAC / cross-asset t |
| [`metrics.mfe_mae`][factrix.metrics.mfe_mae] | `(*, SPARSE, *, PANEL)` | per-event | no formal H₀ |
| [`metrics.monotonicity`][factrix.metrics.monotonicity] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | cs-first | cross-asset t |
| [`metrics.oos`][factrix.metrics.oos] | `(*, CONTINUOUS, *, TIMESERIES)` | ts-only | no formal H₀ |
| [`metrics.quantile`][factrix.metrics.quantile] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | cs-first | cross-asset t |
| [`metrics.regime`][factrix.metrics.regime] | `(*, *, *, *)` | dispatcher | none (no cross-regime test) |
| [`metrics.spanning`][factrix.metrics.spanning] | `factor-return-series consumer (post-PANEL pipeline)` | ts-only | NW HAC / OLS t |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | cs-first | no formal H₀ |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | ts-only (rank autocorrelation across consecutive dates) | no formal H₀ |
| [`metrics.trend`][factrix.metrics.trend] | `(*, CONTINUOUS, *, TIMESERIES)` | ts-only | Theil-Sen rank-based CI |
| [`metrics.ts_asymmetry`][factrix.metrics.ts_asymmetry] | `(COMMON, CONTINUOUS, *, PANEL)` | cs-first | NW HAC Wald |
| [`metrics.ts_beta`][factrix.metrics.ts_beta] | `(COMMON, CONTINUOUS, *, PANEL)` | ts-first | cross-asset t |
| [`metrics.ts_quantile`][factrix.metrics.ts_quantile] | `(COMMON, CONTINUOUS, *, PANEL)` | cs-first | NW HAC Wald |
