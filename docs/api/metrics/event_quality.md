# event_quality

Per-event quality summaries computed on `signed_car`: hit rate, IC,
profit factor, skewness, signal density. Inference is binomial or
nonparametric — descriptive elsewhere.

::: factrix.metrics.event_quality
    options:
      members:
        - event_hit_rate
        - event_ic
        - profit_factor
        - event_skewness
        - signal_density

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual sparse landing page](individual-sparse.md) — adjacent event-study metrics.
