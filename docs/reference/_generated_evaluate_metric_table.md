| Dispatch key `(scope, signal, metric, mode)` | Run by `evaluate()` | Procedure summary | Literature |
|---|---|---|---|
| `(INDIVIDUAL, CONTINUOUS, IC, PANEL)` | [`ic`][factrix.metrics.ic.ic] | Per-date Spearman IC across the asset cross-section. | Grinold (1989), Newey & West (1987) |
| `(INDIVIDUAL, CONTINUOUS, FM, PANEL)` | [`fama_macbeth`][factrix.metrics.fama_macbeth.fama_macbeth] | Fama-MacBeth λ on per-date OLS slope. | Fama & MacBeth (1973), Petersen (2009) |
| `(INDIVIDUAL, SPARSE, *, PANEL)` | [`caar`][factrix.metrics.caar.caar] | Cross-event CAAR with t-test on per-event AR aggregate. | Brown & Warner (1985), MacKinlay (1997) |
| `(COMMON, CONTINUOUS, *, PANEL)` | [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | Per-asset β on broadcast factor + cross-asset t on E[β]. | Black, Jensen & Scholes (1972), Fama & French (1993) |
| `(COMMON, SPARSE, *, PANEL)` | [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | Per-asset β on broadcast event dummy + cross-asset t. | MacKinlay (1997) |
| `(COMMON, CONTINUOUS, *, TIMESERIES)` | [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | Single-asset OLS β on broadcast factor + NW HAC SE. | Newey & West (1987, 1994), Stambaugh (1999) |
| `(*, SPARSE, *, TIMESERIES)` | [`ts_beta`][factrix.metrics.ts_beta.ts_beta] | Single-asset calendar-time TS dummy regression + NW HAC SE. | Newey & West (1994), Ljung & Box (1978) |
