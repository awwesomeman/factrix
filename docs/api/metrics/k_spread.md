---
title: factrix.metrics.k_spread
---

::: factrix.metrics.k_spread
    options:
      show_root_members_full_path: true
      members:
        - k_spread

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Small-N sibling of `quantile_spread`__

    ---

    `k_spread` is the fixed-**count** long-short spread: per non-overlapping
    date it longs the `k` highest-factor names and shorts the `k` lowest,
    then tests the spread across time. Quintile bucketing (`quantile_spread`,
    `n_groups=5`) degrades when `N < 30` — each bucket holds only a few names
    and the breakpoints are unstable. Fixing the count `k` keeps each leg's
    composition stable regardless of `N`. Both name the selection convention,
    not a leg: `quantile_spread` (fraction) ↔ `k_spread` (count).

-   __Cross-sectional dispersion reported alongside__

    ---

    `metadata["cross_sectional_dispersion"]` carries the mean per-date
    cross-sectional standard deviation of returns, so the headline spread can
    be read relative to the typical spread of returns that period rather than
    in isolation.

-   __Cross-section-aware significance__

    ---

    The headline test follows the shared small-cross-section policy
    (`_spread_significance`): with `n_assets < MIN_ASSETS_WARN` the per-date
    spread is heavy-tailed (few names per leg), so the non-overlapping `t` is
    replaced by a block-bootstrap CI; `metadata["method"]` records which ran.
    Dates with fewer than `2·k` names are dropped; if none qualify the metric
    short-circuits with `max_assets_per_date`.

</div>

## Choosing a function

| Goal                                                            | Function          |
|----------------------------------------------------------------|-------------------|
| Long-short spread by extreme **quantiles** (large universe)    | `quantile_spread` |
| Long-short spread by fixed **count** per leg (small universe)  | `k_spread`        |

## Worked example — fixed-K spread on a small universe

!!! example "k_spread on N=20"

    ```python
    import factrix as fx
    from factrix.metrics.k_spread import k_spread
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(n_assets=20, n_dates=180, seed=2024)
    panel = compute_forward_return(raw, forward_periods=5)

    out = k_spread(panel, forward_periods=5, k=3)
    print(out.value, out.metadata["method"], out.metadata["cross_sectional_dispersion"])
    # 0.0018  block-bootstrap CI  0.041   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`quantile_spread`__

    ---

    The fraction-based sibling: long-short spread by extreme quantiles, the
    idiomatic choice for large cross-sections.

    [api/metrics/quantile →](quantile.md)

-   __`top_concentration`__

    ---

    Long-leg diversification (HHI effective bets) on the top bucket.

    [api/metrics/concentration →](concentration.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

</div>
