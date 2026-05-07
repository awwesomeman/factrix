# ts_asymmetry

Long-side / short-side asymmetry test for COMMON / single-asset cells.
Two methods — conditional means and piecewise slopes — both fit by OLS
with NW HAC and tested by Wald χ², so cross-method *p*-values stay
comparable under overlapping forward returns.

!!! info "TIMESERIES-mode conventions"
    `FACTOR_ADF_P` persistence diagnostic, plain stage-1 SE rationale,
    and the `forward_periods` vs `signal_horizon` bias framing apply
    here as for the rest of the TS-mode family. See
    [TIMESERIES-mode conventions](../../reference/ts-mode-conventions.md).

::: factrix.metrics.ts_asymmetry

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Common continuous landing page](common-continuous.md) — adjacent macro-factor metrics.
