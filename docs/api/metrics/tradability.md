# tradability

Implementation-feasibility diagnostics: turnover (rank stability),
notional turnover (Novy-Marx & Velikov τ), breakeven cost, and net
spread. Profile-level — not gating.

::: factrix.metrics.tradability
    options:
      members:
        - notional_turnover
        - turnover
        - breakeven_cost
        - net_spread

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Individual continuous landing page](individual-continuous.md) — adjacent metrics in the same cell.
