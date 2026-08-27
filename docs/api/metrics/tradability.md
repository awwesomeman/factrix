---
title: factrix.metrics.tradability
---

::: factrix.metrics.tradability
    options:
      show_root_members_full_path: true
      members:
        - notional_turnover
        - rank_turnover
        - breakeven_cost
        - net_spread

<hr>

!!! warning "Two flavours of turnover — do not mix them"
    `notional_turnover` is the Novy-Marx & Velikov (2016) $\tau$:
    fraction of top-and-bottom quantile members replaced per rebalance.
    This is the quantity whose units are compatible with `breakeven_cost`
    and `net_spread`. `rank_turnover` is `1 - mean(rank autocorrelation)`,
    a *rank-stability diagnostic* over the full cross-section — mid-rank
    churn that triggers no Q1/Qn rebalance still counts. Feeding
    `rank_turnover()` into the cost formulas will mis-state the result by a
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

    `rank_turnover` — $1 - \overline{\rho}$ on per-date rank
    autocorrelation, optionally restricted to the top/bottom-$q$
    tail union. Use for stability rankings across factors;
    **not** for cost arithmetic. Because $\rho \in [-1, +1]$, the value
    lies in $[0, 2]$ — not $[0, 1]$: a stable ranking gives 0, an
    independent re-draw gives $\approx 1$, a reversed ranking up to 2.
    `n_obs` counts adjacent-period transitions (`n_obs_axis="periods"`).

-   __Breakeven cost in bps__

    ---

    `breakeven_cost = gross_spread \cdot h / (4 \cdot \tau) \cdot 10^4`.
    If the venue's actual **one-way** cost is below this, the factor's
    alpha survives. The `\cdot h` lift puts per-period spread onto the
    per-rebalance scale of $\tau$. The $4\tau$ is two legs times two
    trades (sell the leaver, buy the joiner) per unit of per-leg
    turnover; halve a round-trip quote before comparing it to this
    number. See the `breakeven_cost` Notes for the full derivation.

-   __Net spread after estimated costs__

    ---

    `net_spread = gross_spread - 4 \cdot (cost_{bps} / 10^4) \cdot \tau / h`,
    with `cost_bps` quoted **one-way**. The cost is paid once per
    $h$-period rebalance, so dividing by $h$ amortises it back to the
    per-period scale of `gross_spread` — without that, any factor with
    $h \geq 2$ would be artificially killed. `breakeven_cost` inverts
    this same $4\tau$ coefficient, so the two stay consistent.

</div>

## Choosing a function

| Goal                                                                            | Function             |
|---------------------------------------------------------------------------------|----------------------|
| Per-rebalance Q1/Qn membership churn — feeds the cost formulas (default $\tau$) | `notional_turnover`  |
| Rank-stability diagnostic across the full cross-section (or tail-union)         | `rank_turnover`           |
| Breakeven trading cost in bps, given a gross spread and $\tau$                  | `breakeven_cost`     |
| Net per-period spread after a venue-specific cost estimate                      | `net_spread`         |

## Worked example — notional turnover then breakeven and net spread

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
    # Stamps both horizons; every standalone call below reads the overlap.
    panel = compute_forward_return(raw, forward_periods=5)

    # quantile_spread returns {factor_name: MetricResult}; notional_turnover
    # returns a single MetricResult.
    spread = quantile_spread(panel, n_groups=10)["factor"]
    tau    = notional_turnover(panel, n_groups=10)
    print(spread.value, tau.value)
    # 0.00258  0.897   (approximate)

    # The scalar helpers take the gross spread positionally and every other
    # parameter by keyword. Pass the MetricResults, not their .value: the
    # helper then verifies the two describe the same portfolio.
    be  = breakeven_cost(spread, turnover=tau, forward_periods=5)
    net = net_spread(spread, turnover=tau,
                     estimated_cost_bps=30.0, forward_periods=5)
    print(be.value, net.value)
    # 36.0   0.00043   (approximate; one-way bps and per-period spread)
    ```

!!! warning "The spread and the turnover must price the same portfolio"
    The cost algebra is a statement about *one* book, so a τ measured on
    decile membership churn does not price a quintile spread, and a τ per bar
    does not price a 5-period holding. `quantile_spread` and
    `notional_turnover` used to ship incompatible defaults (`n_groups` 5 vs
    10, `forward_periods` 5 vs 1). On a 60-name, 400-period panel at
    `gross_spread = 0.001`, the matched pair gives breakeven **15.7 bps** and
    net **−9.14 bps**; each function at its own default gave **2.8 bps** and
    **−98.02 bps** — breakeven understated 5.6×, drag overstated 10.7×.

    They now share one constant (`DEFAULT_N_GROUPS = 5`,
    `DEFAULT_FORWARD_PERIODS = 5`), so the defaults pair by construction. And
    when handed the producing `MetricResult`s rather than bare floats,
    `breakeven_cost` / `net_spread` cross-check `n_groups` and
    `forward_periods` and raise `UserInputError` on a mismatch, recording
    `pairing_checked` in metadata otherwise. Bare floats carry no provenance,
    so nothing can be verified — prefer passing the results.

    `monotonicity` deliberately keeps its own `n_groups=10`: a decile curve is
    the shape it is calibrated to read, not a long-short leg.

`breakeven_cost` and `net_spread` are scalar post-processing helpers, not
panel-evaluation metrics. Both are `@metric` classes, so the
gross spread is their call-time data argument and everything else must be passed
by keyword — a second positional argument raises `TypeError`. `inspect_data()` marks these helpers as
standalone so they are not included in `inspect_data().usable.to_metrics_dict()`.

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
    rank turnover for a feasibility picture.

    [api/metrics/concentration →](concentration.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice rank turnover / breakeven
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
