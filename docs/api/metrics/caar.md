# caar

Cumulative Average Abnormal Return tests for event signals. CAAR
*t*-test (parametric) and BMP standardised AR *z*-test (robust to
event-induced variance).

→ Formula and references:
[CAAR cross-event t](../../reference/statistical-methods.md#caar-cross-event-t)
(`caar`),
[BMP standardised AR](../../reference/statistical-methods.md#bmp-standardised-ar)
(`bmp_test`).

!!! info "Event-study contracts"
    `signed_car`, the `estimation_window` consumed by `bmp_test`, and
    factrix's confounded-event handling are documented in
    [Metric applicability § Event-study contracts](../../reference/metric-applicability.md#event-study-contracts).
    factrix computes **CAR** (sum of per-period abnormal returns), not
    BHAR; see the same section for the distinction.

::: factrix.metrics.caar

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual sparse landing page](individual-sparse.md) — adjacent event-study metrics.
