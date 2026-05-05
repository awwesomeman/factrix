| Module | Cell scope | Aggregation order | Inference SE |
|---|---|---|---|
| [`metrics/caar.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/caar.py) | `(*, SPARSE, *, PANEL)` | per-event | non-overlapping t / z |
| [`metrics/clustering.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/clustering.py) | `(*, SPARSE, *, PANEL)` | static CS | no formal H₀ |
| [`metrics/concentration.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/concentration.py) | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | across-time t (one-sided H₀: ratio ≥ 0.5) |
| [`metrics/corrado.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/corrado.py) | `(*, SPARSE, *, PANEL)` | per-event | nonparametric rank |
| [`metrics/event_horizon.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/event_horizon.py) | `(*, SPARSE, *, PANEL)` | per-event | binomial |
| [`metrics/event_quality.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/event_quality.py) | `(*, SPARSE, *, PANEL)` | per-event | binomial / nonparametric rank |
| [`metrics/fama_macbeth.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/fama_macbeth.py) | `(INDIVIDUAL, CONTINUOUS, FM, PANEL)` | CS-first | NW HAC / clustered t |
| [`metrics/hit_rate.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/hit_rate.py) | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | binomial |
| [`metrics/ic.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ic.py) | `(INDIVIDUAL, CONTINUOUS, IC, PANEL)` | CS-first | NW HAC / cross-asset t |
| [`metrics/mfe_mae.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/mfe_mae.py) | `(*, SPARSE, *, PANEL)` | per-event | no formal H₀ |
| [`metrics/monotonicity.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/monotonicity.py) | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | cross-asset t |
| [`metrics/oos.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/oos.py) | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | no formal H₀ |
| [`metrics/quantile.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/quantile.py) | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | cross-asset t |
| [`metrics/spanning.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/spanning.py) | `factor-return-series consumer (post-PANEL pipeline)` | TS-only | NW HAC / OLS t |
| [`metrics/tradability.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/tradability.py) | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | CS-first | no formal H₀ |
| [`metrics/tradability.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/tradability.py) | `(INDIVIDUAL, CONTINUOUS, *, PANEL)` | TS-only (rank autocorrelation across consecutive dates) | no formal H₀ |
| [`metrics/trend.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/trend.py) | `(*, CONTINUOUS, *, TIMESERIES)` | TS-only | Theil-Sen rank-based CI |
| [`metrics/ts_asymmetry.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ts_asymmetry.py) | `(COMMON, CONTINUOUS, *, PANEL)` | CS-first | NW HAC Wald |
| [`metrics/ts_beta.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ts_beta.py) | `(COMMON, CONTINUOUS, *, PANEL)` | TS-first | cross-asset t |
| [`metrics/ts_quantile.py`](https://github.com/awwesomeman/factrix/blob/main/factrix/metrics/ts_quantile.py) | `(COMMON, CONTINUOUS, *, PANEL)` | CS-first | NW HAC Wald |
