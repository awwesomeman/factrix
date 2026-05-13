---
title: factrix.metrics.monotonicity
---

::: factrix.metrics.monotonicity
    options:
      show_root_heading: true
      show_root_full_path: true
      show_root_toc_entry: true
      show_root_members_full_path: true
      heading_level: 1
      members:
        - monotonicity

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Decile-curve direction test__

    ---

    Per-date Spearman correlation between quantile-group index and
    group mean return; cross-asset $t$ on the signed series asks
    whether the bucket ordering is *consistently* increasing (or
    decreasing) across dates — a stronger requirement than a positive
    long-short spread.

-   __Magnitude vs direction separation__

    ---

    `value` is mean $|\text{Spearman}|$ (magnitude, $\geq 0$); `stat`
    is a $t$ on the signed series. High `value` with insignificant
    `stat` flags a factor that monotonically sorts returns but whose
    direction flips across dates — Patton-Timmermann (2010) territory
    that a single signed average would hide.

</div>

!!! info "Per-bucket cardinality floor"
    `min_assets_per_group = 50` (Patton-Timmermann 2010) — the slice-
    test function downscales `n_groups` automatically so each bucket meets
    the floor; below it, per-date bucket means are noise-dominated and
    the rank statistic is unreliable. Defaults: `n_groups=10` for
    universes around 2000 stocks; drop to 5 for $n_{assets} < 1000$
    and 3 for $n_{assets} < 200$.

## Worked example — Spearman direction test on quantile-bucket returns

!!! example "monotonicity on a synthetic cross-sectional panel"

    ```python
    import factrix as fx
    from factrix.metrics.monotonicity import monotonicity
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=2000, n_dates=500, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    out = monotonicity(panel, forward_periods=5, n_groups=10)
    print(out.value, out.stat,
          out.metadata["mean_signed"], out.metadata["p_value"])
    # 0.41  5.62  0.39  2.1e-08   (approximate)
    # value ≈ mean|Spearman|; mean_signed ≈ direction; stat t on signed series
    ```

## See also

<div class="grid cards" markdown>

-   __`quantile_spread` / `compute_group_returns`__

    ---

    The decile-curve chart input and the long-short headline spread
    over the same buckets.

    [api/metrics/quantile →](quantile.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice monotonicity summaries.

    [api/by-slice →](../by-slice.md)

-   __`slice_pairwise_test` / `slice_joint_test`__

    ---

    Cross-slice inference (Wald $\chi^2$ + Holm / Romano-Wolf adjusted $p$).

    [api/slice-test →](../slice-test.md)

-   __Statistical methods__

    ---

    Cross-asset $t$ on the per-date signed Spearman series, DDOF
    convention.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    Per-bucket sample-size floor (Patton-Timmermann 2010) and the
    `n_groups` downscaling contract.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
