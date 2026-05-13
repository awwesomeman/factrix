---
title: factrix.metrics.oos
---

::: factrix.metrics.oos
    options:
      show_root_members_full_path: true
      members:
        - multi_split_oos_decay
        - SplitDetail

<hr>

!!! info "Descriptive only — no formal $H_0$"
    `multi_split_oos_decay` emits a survival ratio + sign-flip detail;
    no `p_value` is attached and `stat` is `None`. A $t$-test at the
    `MIN_OOS_PERIODS` floor would have power $\approx 0$ and would
    invite mis-reading the diagnostic as a significance test. Callers
    routing this output into BHY / gate logic must read `status`
    (`"PASS"` / `"VETOED"`) and `sign_flipped`, not a probability.

## Use cases

<div class="grid cards" markdown>

-   __Persistence read on a factor-return series__

    ---

    `multi_split_oos_decay` is a `(*, CONTINUOUS, *, TIMESERIES)`
    diagnostic — input is a 1-D `(date, value)` series, typically IC
    from `compute_ic`, spread from `compute_spread_series`, or any
    other factor-mimicking-portfolio return series. Reports
    $|\mathrm{mean}_{\text{OOS}}| / |\mathrm{mean}_{\text{IS}}|$ across
    multiple `(IS_fraction, OOS_fraction)` splits.

-   __Sign-flip veto__

    ---

    Any split with opposite-signed IS and OOS means flips
    `sign_flipped = True` and forces `status = "VETOED"` — IC
    sign-flip OOS means the factor predicts the wrong direction, not
    just a weaker one. McLean & Pontiff (2016) report average OOS
    decay around 32 %; factrix's default `survival_threshold = 0.5`
    sits inside that window.

-   __Median across splits, not mean__

    ---

    Headline is `median_f s_f` over the default splits
    `[(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]`. A single regime change
    landing inside one split distorts the mean disproportionately;
    the median absorbs it.

</div>

## Choosing a function

| Goal                                                                          | Function                |
|-------------------------------------------------------------------------------|-------------------------|
| Multi-split OOS survival + sign-flip gate on a `(date, value)` series         | `multi_split_oos_decay` |
| Typed accessor for an individual split's `(is_ratio, mean_is, mean_oos, ...)` | `SplitDetail`           |

## Worked example — IC series fed into multi_split_oos_decay

!!! example "compute_ic → multi_split_oos_decay"

    ```python
    import factrix as fx
    from factrix.metrics.ic import compute_ic
    from factrix.metrics.oos import multi_split_oos_decay
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=1000, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    # The series diagnostic consumes (date, value); the value column on
    # the compute_ic output is named ``ic``.
    ic_df = compute_ic(panel)
    out   = multi_split_oos_decay(ic_df, value_col="ic")
    print(out.value, out.metadata["status"], out.metadata["sign_flipped"])
    # 0.94   PASS   False   (approximate)
    for split in out.metadata["per_split"]:
        print(split)
    # {"is_ratio": 0.6, "mean_is": 0.080, "mean_oos": 0.077,
    #  "survival_ratio": 0.96, "sign_flipped": False}
    # ...
    ```

## See also

<div class="grid cards" markdown>

-   __`compute_ic` / `compute_spread_series`__

    ---

    Canonical producers of the `(date, value)` series this diagnostic
    consumes.

    [api/metrics/ic →](ic.md)

-   __`hit_rate` / `trend`__

    ---

    Sibling series diagnostics on the same input shape — sign
    significance and slope detection. Pair with `oos` when both
    in-sample magnitude and out-of-sample persistence matter.

    [api/metrics/hit_rate →](hit_rate.md)

-   __`by_slice`__

    ---

    Per-slice survival summaries (regime / universe / sector).

    [api/by-slice →](../by-slice.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_OOS_PERIODS * 2` floor; per-split `MIN_OOS_PERIODS` on each
    side).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Series diagnostics landing__

    ---

    Adjacent axis-agnostic series diagnostics.

    [api/metrics/series-tools →](series-tools.md)

</div>
