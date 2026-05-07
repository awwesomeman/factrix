# corrado

Corrado (1989) nonparametric rank test on event abnormal returns.
Robust to extreme returns, non-normality, and cross-asset
heteroscedasticity. Direction-adjusted for two-sided signals.

!!! info "Event-study contracts"
    Corrado's primitive is `signed_rank = uniform_rank(forward_return) ×
    sign(factor)` — note that the direction adjustment is on the rank,
    not on the return itself. The
    [Event-study contracts table](../../reference/metric-applicability.md#abnormal-return-definition-per-metric)
    contrasts this with `caar` (magnitude-weighted) and `event_quality`
    (sign-only on the raw return). The
    [`estimation_window`](../../reference/metric-applicability.md#estimation_window)
    section specifies the per-asset baseline this test ranks against.

::: factrix.metrics.corrado

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual sparse landing page](individual-sparse.md) — adjacent event-study metrics.
