---
title: factrix.metrics.tradability
---

::: factrix.metrics.tradability
    options:
      show_root_members_full_path: true
      members:
        - notional_turnover
        - turnover
        - breakeven_cost
        - net_spread

<hr>

!!! warning "Two flavours of turnover — do not mix them"
    `notional_turnover` is the Novy-Marx & Velikov (2016) $\tau$:
    fraction of top-and-bottom quantile members replaced per rebalance.
    This is the quantity whose units are compatible with `breakeven_cost`
    and `net_spread`. `turnover` is `1 - mean(rank autocorrelation)`,
    a *rank-stability diagnostic* over the full cross-section — mid-rank
    churn that triggers no Q1/Qn rebalance still counts. Feeding
    `turnover()` into the cost formulas will mis-state the result by a
    factor that grows with mid-rank churn.

## Use cases

<div class="grid cards" markdown>

-   __Portfolio rebalance cost driver__

    ---

    `notional_turnover` — per-rebalance fraction of the equal-weight
    Q1/Qn long-short portfolio that must be traded. Drop-in input for
    `breakeven_cost` / `net_spread`. Matches the Novy-Marx & Velikov
    (2016) $\tau$ used in their anomaly-cost taxonomy.

-   __Cross-factor rank-stability comparison__

    ---

    `turnover` — $1 - \overline{\rho}$ on per-date rank
    autocorrelation, optionally restricted to the top/bottom-$q$
    tail union. Use for stability rankings across factors;
    **not** for cost arithmetic.

-   __Breakeven cost in bps__

    ---

    `breakeven_cost = gross_spread \cdot h / (2 \cdot \tau) \cdot 10^4`.
    If the venue's actual round-trip cost is below this, the factor's
    alpha survives. The `\cdot h` lift puts per-period spread onto the
    per-rebalance scale of $\tau$.

-   __Net spread after estimated costs__

    ---

    `net_spread = gross_spread - 2 \cdot (cost_{bps} / 10^4) \cdot \tau / h`.
    The cost is paid once per $h$-period rebalance, so dividing by $h$
    amortises it back to the per-period scale of `gross_spread` — without
    that, any factor with $h \geq 2$ would be artificially killed.

</div>

## Choosing a function

| Goal                                                                            | Function             |
|---------------------------------------------------------------------------------|----------------------|
| Per-rebalance Q1/Qn membership churn — feeds the cost formulas (default $\tau$) | `notional_turnover`  |
| Rank-stability diagnostic across the full cross-section (or tail-union)         | `turnover`           |
| Breakeven trading cost in bps, given a gross spread and $\tau$                  | `breakeven_cost`     |
| Net per-period spread after a venue-specific cost estimate                      | `net_spread`         |

## Worked example — turnover then breakeven and net spread

!!! example "quantile_spread → notional_turnover → breakeven_cost / net_spread"

    ```python
    import factrix as fx
    from factrix.metrics.quantile import quantile_spread
    from factrix.metrics.tradability import (
        notional_turnover, breakeven_cost, net_spread,
    )
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=500, n_dates=500, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    spread = quantile_spread(panel, forward_periods=5, n_groups=10)
    tau    = notional_turnover(panel, n_groups=10, forward_periods=5)
    print(spread.value, tau.value)
    # 0.0021  0.18   (approximate)

    be  = breakeven_cost(spread.value, tau.value, forward_periods=5)
    net = net_spread(spread.value, tau.value,
                     estimated_cost_bps=30.0, forward_periods=5)
    print(be.value, net.value)
    # 291.7   0.00189   (approximate; bps and per-period spread)
    ```

## See also

<div class="grid cards" markdown>

-   __`quantile_spread` / `quantile_spread_vw`__

    ---

    Source of `gross_spread` for the cost formulas; pairs naturally
    with `notional_turnover` on the same Q1/Qn buckets.

    [api/metrics/quantile →](quantile.md)

-   __`top_concentration`__

    ---

    Long-leg concentration on the same top bucket — combine with
    turnover for a feasibility picture.

    [api/metrics/concentration →](concentration.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice turnover / breakeven
    summaries.

    [api/by-slice →](../by-slice.md)

-   __Metric applicability reference__

    ---

    Implementation-feasibility framing, not a factor-quality significance test.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
