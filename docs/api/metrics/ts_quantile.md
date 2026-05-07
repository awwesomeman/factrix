# ts_quantile

Quantile-bucketed NW HAC OLS on the per-date `(_f, _r)` series for
COMMON / single-asset cells; Wald χ² on the top-bottom bucket spread.
Catches U-shape / extreme-only signals that linear β assumes away.

!!! info "TIMESERIES-mode conventions"
    `FACTOR_ADF_P` persistence diagnostic, plain stage-1 SE rationale,
    and the `forward_periods` vs `signal_horizon` bias framing apply
    here as for the rest of the TS-mode family. See
    [TIMESERIES-mode conventions](../../reference/ts-mode-conventions.md).

::: factrix.metrics.ts_quantile

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Common continuous landing page](common-continuous.md) — adjacent macro-factor metrics.
