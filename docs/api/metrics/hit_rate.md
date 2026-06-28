---
title: factrix.metrics.hit_rate
---

::: factrix.metrics.hit_rate
    options:
      show_root_members_full_path: true
      members:
        - hit_rate

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Sign-significance on a factor-return series__

    ---

    `hit_rate` is a `(*, CONTINUOUS, *, TIMESERIES)` diagnostic — input
    is a 1-D series with `(date, value)`, not the raw panel. Typical
    pipes: per-date information coefficient (IC) from `compute_ic`, quantile spread from
    `compute_spread_series`, or any other factor-mimicking-portfolio
    return series. Reports the fraction of periods with `value > 0`
    against $H_0: p = 0.5$.

-   __Two-branch test under one $p$-value__

    ---

    Below `_BINOMIAL_EXACT_CUTOFF` the test is the exact two-sided
    binomial; above the cutoff it switches to the normal-approximation
    $z$. `stat` / `stat_type` track the branch actually taken, so a
    reader can never see `stat=z` paired with an exact-binomial $p$.

-   __Non-overlap stride matches the IC pipeline__

    ---

    The series is sub-sampled at stride `forward_periods` before the
    test so the MA dependence induced by overlapping forward returns
    does not leak in; same convention as `quantile_spread` and the
    other non-overlap inference paths.

</div>

## Choosing a function

| Goal                                                                       | Function           |
|----------------------------------------------------------------------------|--------------------|
| Hit-rate significance vs $p = 0.5$ on a `(date, value)` series             | `hit_rate`         |
| Per-date hit indicator (`value > 0` cast to float) for slice-test plumbing | `per_date_series`  |

## Worked example — IC series fed into hit_rate

!!! example "compute_ic → hit_rate"

    ```python
    import factrix as fx
    from factrix.metrics.ic import compute_ic
    from factrix.metrics.hit_rate import hit_rate
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=500, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    # The series diagnostic consumes (date, value); the value column on
    # the compute_ic output is named ``ic``.
    ic_df = compute_ic(panel)["factor"]
    out   = hit_rate(ic_df, value_col="ic", forward_periods=5)
    print(out.value, out.stat, out.p_value, out.metadata["method"])
    # 0.62  19  0.011   exact-binomial   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`compute_ic` / `compute_spread_series`__

    ---

    Canonical producers of the `(date, value)` series this diagnostic
    consumes. `compute_ic` emits `(date, ic, tie_ratio)`;
    `compute_spread_series` emits `(date, spread, ...)`.

    [api/metrics/ic →](ic.md)

-   __`trend` / `oos`__

    ---

    Sibling series diagnostics on the same input shape — slope
    detection and IS/out-of-sample (OOS) persistence.

    [api/metrics/trend →](trend.md)

-   __`by_slice`__

    ---

    Per-slice hit-rate summaries; uses `per_date_series` as the
    slice-test capability hook.

    [api/by-slice →](../by-slice.md)

-   __Statistical methods__

    ---

    Two-sided binomial branching, non-overlap stride discipline, and
    the Hansen-Hodrick autocorrelation floor that motivates it.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_IC_PERIODS` floor on the sampled series).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Series diagnostics landing__

    ---

    Adjacent axis-agnostic series diagnostics.

    [api/metrics/series-tools →](series-tools.md)

</div>
