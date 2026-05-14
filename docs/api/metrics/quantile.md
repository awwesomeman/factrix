---
title: factrix.metrics.quantile
---

::: factrix.metrics.quantile
    options:
      show_root_members_full_path: true
      members:
        - compute_spread_series
        - compute_group_returns
        - quantile_spread
        - quantile_spread_vw

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Compute per-date long-short spread series__

    ---

    Build the per-date `spread = top_return - bottom_return` series
    (with `top_return`, `bottom_return`, `universe_return`) on a
    non-overlap-sampled panel. Pre-step for `quantile_spread`; also
    feeds `spanning_alpha` and any `series/` tool.

-   __Mean-spread significance, equal-weighted__

    ---

    Test $H_0: \mathbb{E}[\text{spread}] = 0$ on the non-overlap
    spread series, with the long-vs-short alpha decomposition
    (`top - universe`, `universe - bottom`) attached so callers can
    attribute the spread to long-side vs short-side excess.

-   __Value-weighted spread for capacity diagnostics__

    ---

    `quantile_spread_vw` weights each bucket by lagged `market_cap`
    (or any caller-supplied `weight_col=`). When the VW spread is much
    smaller than the EW spread, the alpha is concentrated in small
    names and may not survive capacity / liquidity constraints —
    Hou-Xue-Zhang (2020) found ~65% of factors disappear under VW.

-   __Per-bucket mean returns for monotonicity charts__

    ---

    `compute_group_returns` returns the pooled mean forward return per
    quantile bucket — the chart input that shows whether returns rise
    monotonically across deciles, before any formal monotonicity test.

</div>

## Choosing a function

| Goal                                                                            | Function                |
|---------------------------------------------------------------------------------|-------------------------|
| Per-date long-short spread table for downstream inspection / slicing            | `compute_spread_series` |
| Per-bucket pooled mean return (decile-curve chart input)                        | `compute_group_returns` |
| Mean-spread significance, equal-weighted, non-overlap $t$ (default)             | `quantile_spread`       |
| Mean-spread significance, value-weighted (capacity / size-concentration check)  | `quantile_spread_vw`    |

## Worked example — per-date spread then EW vs VW significance

!!! example "compute_spread_series → quantile_spread → quantile_spread_vw on a synthetic cross-sectional panel"

    ```python
    import factrix as fx
    from factrix.metrics.quantile import (
        compute_spread_series, quantile_spread, quantile_spread_vw,
    )
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=200, n_dates=500, ic_target=0.08,
        with_market_cap=True, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    spread_df = compute_spread_series(panel, forward_periods=5, n_groups=5)
    print(spread_df.head())
    # ┌────────────┬──────────┬───────────────┬─────────────────┬──────────────────┐
    # │ date       ┆ spread   ┆ top_return    ┆ bottom_return   ┆ universe_return  │
    # ├────────────┼──────────┼───────────────┼─────────────────┼──────────────────┤
    # │ 2024-01-01 ┆  0.0042  ┆  0.0061       ┆  0.0019         ┆  0.0040          │
    # │  ...       ┆  ...     ┆  ...          ┆  ...            ┆  ...             │
    # └────────────┴──────────┴───────────────┴─────────────────┴──────────────────┘

    ew = quantile_spread(panel, forward_periods=5, n_groups=5,
                         _precomputed_series=spread_df)
    print(ew.value, ew.stat, ew.metadata["long_alpha"], ew.metadata["short_alpha"])
    # 0.0041  4.92  0.0019  0.0022   (approximate)

    vw = quantile_spread_vw(panel, forward_periods=5, n_groups=5,
                            weight_col="market_cap")
    print(vw.value, vw.stat)
    # 0.0017  2.10   (approximate — VW < EW signals small-cap concentration)
    ```

## See also

<div class="grid cards" markdown>

-   __`monotonicity`__

    ---

    Decile-curve direction-of-monotonicity test on the same buckets.

    [api/metrics/monotonicity →](monotonicity.md)

-   __`spanning_alpha`__

    ---

    Does this spread series carry alpha after controlling for base
    factor spreads? Consumes `compute_spread_series` output directly.

    [api/metrics/spanning →](spanning.md)

-   __`notional_turnover` / `breakeven_cost` / `net_spread`__

    ---

    Implementation feasibility on the same Q1/Qn long-short portfolio.

    [api/metrics/tradability →](tradability.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice spread summaries.

    [api/by-slice →](../by-slice.md)

-   __Statistical methods__

    ---

    Non-overlap $t$ vs Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) on overlapping spreads, DDOF convention.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
