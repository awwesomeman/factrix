| Module | Cell scope | Aggregation order | Inference SE |
|---|---|---|---|
| [`metrics.caar`][factrix.metrics.caar] | `(*, SPARSE, *, PANEL)` | per-event | non-overlapping t / z |
| [`metrics.clustering`][factrix.metrics.clustering] | `(*, SPARSE, *, PANEL)` | static CS | no formal H₀ |
| [`metrics.concentration`][factrix.metrics.concentration] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | across-time t (one-sided H₀: ratio ≥ 0.5) |
| [`metrics.corrado`][factrix.metrics.corrado] | `(*, SPARSE, *, PANEL)` | per-event | nonparametric rank |
| [`metrics.event_horizon`][factrix.metrics.event_horizon] | `(*, SPARSE, *, PANEL)` | per-event | binomial |
| [`metrics.event_quality`][factrix.metrics.event_quality] | `(*, SPARSE, *, PANEL)` | per-event | binomial / nonparametric rank |
| [`metrics.fama_macbeth`][factrix.metrics.fama_macbeth] | `(INDIVIDUAL, CONTINUOUS, FM, PANEL)` | CS-first | NW HAC / clustered t |
| [`metrics.hit_rate`][factrix.metrics.hit_rate] | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | binomial |
| [`metrics.ic`][factrix.metrics.ic] | `(INDIVIDUAL, CONTINUOUS, IC, PANEL)` | CS-first | NW HAC / cross-asset t |
| [`metrics.mfe_mae`][factrix.metrics.mfe_mae] | `(*, SPARSE, *, PANEL)` | per-event | no formal H₀ |
| [`metrics.monotonicity`][factrix.metrics.monotonicity] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | cross-asset t |
| [`metrics.oos`][factrix.metrics.oos] | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | no formal H₀ |
| [`metrics.quantile`][factrix.metrics.quantile] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | cross-asset t |
| [`metrics.spanning`][factrix.metrics.spanning] | `factor-return-series consumer (post-PANEL pipeline)` | TS-only | NW HAC / OLS t |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | no formal H₀ |
| [`metrics.tradability`][factrix.metrics.tradability] | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | TS-only (rank autocorrelation across consecutive dates) | no formal H₀ |
| [`metrics.trend`][factrix.metrics.trend] | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | Theil-Sen rank-based CI |
| [`metrics.ts_asymmetry`][factrix.metrics.ts_asymmetry] | `(COMMON, CONTINUOUS, *, PANEL)` | CS-first | NW HAC Wald |
| [`metrics.ts_beta`][factrix.metrics.ts_beta] | `(COMMON, CONTINUOUS, *, PANEL)` | TS-first | cross-asset t |
| [`metrics.ts_quantile`][factrix.metrics.ts_quantile] | `(COMMON, CONTINUOUS, *, PANEL)` | CS-first | NW HAC Wald |
