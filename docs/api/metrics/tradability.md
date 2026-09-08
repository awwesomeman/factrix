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
    `notional_turnover` is the Novy-Marx & Velikov (2016) $\tau$: the
    fraction of each top/bottom quantile leg's notional traded per
    rebalance, averaged over the two legs.
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

## The turnover convention

`notional_turnover` reports **one-way** turnover on each equal-weight leg,

$$
\tau_{\text{leg}}(t) = \tfrac{1}{2} \sum_{i} \bigl| w_t(i) - w_{t-1}(i) \bigr| ,
\qquad
w_t(i) = \frac{\mathbb{1}\left[i \in Q(t)\right]}{\left|Q(t)\right|} ,
$$

the sum running over the **union** of the leg's prior and current holdings. An
unchanged book gives 0 and a full rotation gives 1; `value` is the average of
the top and bottom legs, which is why the cost helpers multiply it back up by
$4\tau$ (2 legs $\times$ 2 trades). Halve a two-way (round-trip) turnover quote
before comparing it to this number, exactly as for `estimated_cost_bps`.

Writing $k = |Q(t)|$, $j = |Q(t-1)|$ and $m = |Q(t) \cap Q(t-1)|$, that sum has
the exact closed form

$$
\tau_{\text{leg}}(t) = 1 - \frac{m}{\max(j,\, k)} ,
$$

because the survivors' resizing term $m \cdot |1/k - 1/j|$ cancels against the
smaller side's entries or exits. The metric evaluates this form, so the
membership definition and the weight-change definition are the same statement,
not two approximations of each other.

!!! warning "A shrinking leg books its liquidations"
    Denominating by today's leg size $k$ alone counts only the buys. That is
    the same number whenever the leg grows or holds its size, and it
    understates a leg that **shrinks** — a holding delisted, a narrowing
    universe, a thinner bucket. A four-name universe cut to two, with both
    survivors keeping their leg, forces each equal-weight leg to sell the
    departed name and double the survivor: $\tau = 0.5$, where the $1 - m/k$
    reading was $0$.

    A leg's churn is skipped only when that leg is empty on either date. Its
    valid sample is reported as `metadata["n_top_rebalances"]` or
    `metadata["n_bottom_rebalances"]`; an empty opposite leg does not erase a
    well-defined long-only diagnostic. The headline long-short `value` still
    needs both legs and `metadata["n_rebalances"]` counts that joint sample.
    Consequently, `value` equals the arithmetic mean of the two published
    per-leg means only when their samples coincide. `mean_tail_size` /
    `mean_top_tail_size` / `mean_bottom_tail_size` report the **current** leg
    sizes at $t$ — they equal the turnover denominator only while the legs do
    not shrink.

## The cost algebra's domain

`breakeven_cost` and `net_spread` solve one portfolio's arithmetic, so they
police the economic domain of what they are handed rather than pushing any
float through the formula.

| Input | Domain | Why |
|---|---|---|
| `gross_spread` | finite real numeric scalar; `bool` / strings rejected | A non-finite spread has no reading as a per-period return. |
| `turnover` | finite real numeric scalar, $0 \le \tau \le 1$; `bool` / strings rejected | The one-way per-leg replaced fraction above. `rank_turnover` lives in $[0, 2]$ and does not belong here. |
| `estimated_cost_bps` | finite real numeric scalar, $\ge 0$; `None`, `bool` and strings rejected | A one-way per-trade cost. Omit the argument to use `net_spread`'s 30 bps default; `None` is not a default sentinel. A negative cost would make trading a source of return. |
| `holding_periods` | integer $\ge 1$ | A rebalance interval in underlying return periods. |

A violation raises `UserInputError`, with the same bounds applied to a bare
scalar and to an *available* `MetricResult`'s `value`. Before this was
enforced, `breakeven_cost(0.001, turnover=-0.2, holding_periods=1)` returned
$+\infty$ and `net_spread(..., turnover=-0.2, estimated_cost_bps=30)`
*increased* the alpha it was meant to charge.

!!! note "Unavailable inputs propagate; they are not priced"
    A `MetricResult` whose producer short-circuited — it carries
    `METRIC_UNAVAILABLE`, or simply a non-finite `value` — comes back as the
    consumer's own short circuit: `reason` is `no_gross_spread` or
    `no_turnover`, the producer's `reason` travels under `upstream_reason`,
    and its advisory codes are carried along. `gross_spread` is inspected
    first, so when both are unavailable the spread's reason is the one
    reported. Running the algebra on a NaN instead yields a NaN breakeven that
    reads exactly like a computed one.

### Zero turnover — three different questions

A book that trades nothing pays nothing, so `breakeven_cost` reads the limit
off the sign of the numerator rather than evaluating the ratio.

| `gross_spread` | Result | Reading |
|---|---|---|
| $> 0$ | $+\infty$ | The alpha is free to keep; no finite one-way cost takes it to zero. |
| $< 0$ | $-\infty$ | The book already loses before costs; no cost $\ge 0$ makes it break even. |
| $= 0$ | short circuit, `reason="no_unique_breakeven_cost"` | `net` is zero at *every* cost, so no single cost is the boundary. |

Returning $+\infty$ for all three — as this function did before — says a
losing book can bear an unlimited cost.

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

[](){ #migration--the-holding_periods-rename }
### Migrating to `holding_periods`

`breakeven_cost` and `net_spread` now take `holding_periods=`; the former
`forward_periods=` and `overlap_periods=` keywords raise `TypeError`. Pass the
rebalance interval in underlying return periods. Do not substitute the panel's
derived `overlap_periods` on a coarse evaluation grid, where the units differ.
This rename does not affect `rank_turnover` or `notional_turnover`, whose stride
is controlled by `rebalance_lag`.

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
