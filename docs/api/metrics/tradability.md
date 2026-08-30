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
    `n_obs` counts transitions at the rebalance lag
    (`n_obs_axis="periods"`).

-   __Breakeven cost in bps__

    ---

    `breakeven_cost = gross_spread \cdot H / (4 \cdot \tau) \cdot 10^4`,
    where $H$ is `holding_periods`. If the venue's actual **one-way**
    cost is below this, the factor's alpha survives. The $\cdot H$ lift
    puts the per-underlying-period spread onto the per-rebalance scale
    of $\tau$. The $4\tau$ is two legs times two trades (sell the
    leaver, buy the joiner) per unit of per-leg turnover; halve a
    round-trip quote before comparing it to this number. See the
    `breakeven_cost` Notes for the full derivation.

-   __Net spread after estimated costs__

    ---

    `net_spread = gross_spread - 4 \cdot (cost_{bps} / 10^4) \cdot \tau / H`,
    with `cost_bps` quoted **one-way**. The cost is paid once per
    rebalance, i.e. once per $H$ underlying return periods, so dividing
    by $H$ amortises it back to the scale of `gross_spread` — without
    that, any multi-period holding would be artificially killed.
    `breakeven_cost` inverts this same $4\tau$ coefficient, so the two
    stay consistent.

</div>

## Four period counts, four questions

The tradability surface touches all four of the period counts factrix keeps
apart. They coincide on a full evaluation grid and come apart the moment
`compute_forward_return(..., dates=)` puts the evaluation grid on a coarser
spacing than the return horizon.

| Quantity | Question it answers | Unit | Who declares it |
|---|---|---|---|
| `forward_periods` | Over how many periods was the return measured? | Underlying period grid | `compute_forward_return`; stamped |
| `overlap_periods` | How many adjacent evaluation observations share future periods? | Evaluation-grid observations | Derived and stamped; injected into metrics |
| `rebalance_lag` | How far apart are the rankings / memberships being compared? | Evaluation-grid observations | The user, on `rank_turnover` / `notional_turnover` |
| `holding_periods` | How many return periods pass between paying trading cost? | Underlying period grid | The user, on `breakeven_cost` / `net_spread` |

!!! warning "Do not substitute one for another"
    `rebalance_lag` defaults to the injected `overlap_periods`, which
    reproduces the horizon-aligned turnover the metrics have always reported.
    Pass `rebalance_lag=1` when the evaluation grid *is* the rebalance
    schedule. `holding_periods` has no such default relationship to the stamp:
    it must be the rebalance interval measured in **underlying return
    periods**, because `gross_spread` is normalised to that unit
    (`compute_forward_return` divides by `forward_periods`).

!!! example "Worked numbers — the 10x cost-drag error"
    A signal holding 20 underlying return periods per rebalance, evaluated on
    a coarse grid whose derived `overlap_periods` is 2. At
    `gross_spread = 0.001`, `turnover = 0.20` and a one-way cost of 30 bps:

    ```text
    holding_periods=20  ->  drag = 4 * 0.003 * 0.20 / 20 = 0.00012, net =  0.00088
    overlap_periods=2   ->  drag = 4 * 0.003 * 0.20 /  2 = 0.00120, net = -0.00020
    ```

    Breakeven is 250 bps at 20 underlying periods and 25 bps at overlap 2 —
    a 10x error that flips the sign of the net spread.

### Migration — the `holding_periods` rename

`breakeven_cost` and `net_spread` take `holding_periods=`. The keyword was
`forward_periods=` up to `v0.22.0` and `overlap_periods=` in `v0.23.0`; there
is no deprecation shim, so a call using either older name raises `TypeError`.

- **From `forward_periods=`** — pass the same number as `holding_periods=`.
  The unit is unchanged: underlying return periods between rebalances.
- **From `overlap_periods=`** — pass the rebalance interval in underlying
  return periods, *not* the panel's derived evaluation-grid overlap. On a full
  grid the two coincide and the number is the same; on a panel built with
  `compute_forward_return(..., dates=)` they differ, and substituting the
  derived overlap is exactly the unit error above.
- **The stride pairing check is gone.** `breakeven_cost` / `net_spread` used
  to reject a call whose keyword disagreed with the upstream producer's
  `overlap_periods`. That check compared two different units once the
  evaluation grid could differ from the horizon, so it is removed;
  `holding_periods` is recorded in metadata instead. The `n_groups` bucketing
  check is unchanged.
- **`rank_turnover` / `notional_turnover` are unaffected by the rename.** They
  keep the injected `overlap_periods` as their default stride and gain the
  optional `rebalance_lag=`.

## Choosing a function

| Goal                                                                            | Function             |
|---------------------------------------------------------------------------------|----------------------|
| Per-rebalance Q1/Qn membership churn — feeds the cost formulas (default $\tau$) | `notional_turnover`  |
| Top-leg-only churn — matched proxy for an equal-weight top-quantile long-only book (not a cost model) | `notional_turnover` → `metadata["mean_top_turnover"]` |
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
        n_assets=500, n_dates=500, ic_target=0.08, rng=2024,
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
    # holding_periods is the rebalance interval in underlying return periods.
    # On this full grid that is the forward_periods the return was built at.
    be  = breakeven_cost(spread, turnover=tau, holding_periods=5)
    net = net_spread(spread, turnover=tau,
                     estimated_cost_bps=30.0, holding_periods=5)
    print(be.value, net.value)
    # 36.0   0.00043   (approximate; one-way bps and per-period spread)
    ```

!!! warning "The spread and the turnover must price the same portfolio"
    The cost algebra is a statement about *one* book, so a τ measured on
    decile membership churn does not price a quintile spread, and a τ per
    period does not price a multi-period holding. `quantile_spread` and
    `notional_turnover` used to ship incompatible defaults (`n_groups` 5 vs
    10, stride 5 vs 1). On a 60-name, 400-period panel at
    `gross_spread = 0.001`, the matched pair gives breakeven **15.7 bps** and
    net **−9.14 bps**; each function at its own default gave **2.8 bps** and
    **−98.02 bps** — breakeven understated 5.6×, drag overstated 10.7×.

    They now share one constant (`DEFAULT_N_GROUPS = 5`,
    `DEFAULT_FORWARD_PERIODS = 5`), so the defaults pair by construction. And
    when handed the producing `MetricResult`s rather than bare floats,
    `breakeven_cost` / `net_spread` cross-check `n_groups` and raise
    `UserInputError` on a mismatch, recording `pairing_checked` in metadata
    otherwise. Bare floats carry no provenance, so nothing can be verified —
    prefer passing the results.

    `holding_periods` is **not** cross-checked against the producers. It
    describes the trading schedule in underlying return periods, which no
    upstream metadata records; the stride a producer *does* record is an
    evaluation-grid count, so equality between the two would not have been
    evidence of a correctly paired book.

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
