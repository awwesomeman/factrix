# ts_asymmetry

Long-side / short-side asymmetry test for COMMON / single-asset cells.
Two methods — conditional means and piecewise slopes — both fit by OLS
with NW HAC and tested by Wald χ², so cross-method *p*-values stay
comparable under overlapping forward returns.

::: factrix.metrics.ts_asymmetry

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Common continuous landing page](common-continuous.md) — adjacent macro-factor metrics.
