---
title: factrix.metrics.trend
---

::: factrix.metrics.trend
    options:
      show_root_heading: true
      show_root_full_path: true
      show_root_toc_entry: true
      show_root_members_full_path: true
      heading_level: 1
      members:
        - ic_trend

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Detect alpha decay / crowding on an IC series__

    ---

    `ic_trend` is a `(*, CONTINUOUS, *, TIMESERIES)` diagnostic — input
    is a 1-D `(date, value)` series, typically the per-date IC from
    `compute_ic`. Slope $\approx 0$ = stable; slope significantly $< 0$
    = decaying (crowding / alpha erosion in the Lou-Polk 2022 sense);
    slope significantly $> 0$ = improving.

-   __Outlier-robust slope via Theil-Sen__

    ---

    Median pairwise slope $\mathrm{median}\{(y_j - y_i)/(j - i): i <
    j\}$ has a 29.3 % breakdown point — absorbs IC outliers (e.g. COVID-
    era spikes) that would dominate an OLS slope. The trade-off is the
    SE recovered from the rank-CI is approximate, not asymptotically
    exact.

-   __Unit-root pre-check on the input__

    ---

    `adf_threshold` (default 0.10, the Stock-Watson cutoff) drives an
    ADF persistence diagnostic on the input series. Above the cutoff
    the slope null is rejected at inflated rates regardless of the
    true trend; the slope value is still returned but
    `metadata["unit_root_suspected"] = True` flags it for sceptical
    reading. Pass `adf_threshold=None` to skip the check entirely.

-   __Reusable on any post-PANEL series__

    ---

    The `name=` argument lets `EventFactor.caar_trend` /
    `MacroPanelFactor.beta_trend` pass their own primitive names so
    method / cache key / primitive name stay three-point unified —
    the same Theil-Sen primitive serves caar series, rolling-$\beta$
    series, and spread series, not just IC.

</div>

## Worked example — IC series fed into ic_trend

!!! example "compute_ic → ic_trend with ADF persistence check"

    ```python
    import factrix as fx
    from factrix.metrics.ic import compute_ic
    from factrix.metrics.trend import ic_trend
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=1000, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    # The series diagnostic consumes (date, value); the value column on
    # the compute_ic output is named ``ic``.
    ic_df = compute_ic(panel)
    out   = ic_trend(ic_df, value_col="ic")
    print(out.value, out.stat, out.metadata["p_value"])
    # -3.2e-05  -0.91  0.36   (approximate; flat slope)
    print(out.metadata["ci_low"], out.metadata["ci_high"],
          out.metadata["ci_excludes_zero"])
    # -1.0e-04  3.5e-05  False
    print(out.metadata["adf_p"], out.metadata["unit_root_suspected"])
    # 0.003  False   (stationary; slope read is trustworthy)
    ```

## See also

<div class="grid cards" markdown>

-   __`compute_ic` / `compute_spread_series`__

    ---

    Canonical producers of the `(date, value)` series this diagnostic
    consumes — IC series, spread series, or any other factor-mimicking-
    portfolio return series.

    [api/metrics/ic →](ic.md)

-   __`hit_rate` / `oos`__

    ---

    Sibling series diagnostics on the same input shape — sign
    significance and IS/OOS persistence.

    [api/metrics/hit_rate →](hit_rate.md)

-   __`compute_rolling_mean_beta`__

    ---

    Common-factor source of a rolling-$\beta$ series for stability
    trend analysis via the same primitive.

    [api/metrics/ts_beta →](ts_beta.md)

-   __Statistical methods__

    ---

    Theil-Sen breakdown point, ADF persistence diagnostic (MacKinnon
    response surface), and the rank-CI to approximate-$t$ recovery.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    ($n \geq 10$).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Series diagnostics landing__

    ---

    Adjacent axis-agnostic series diagnostics.

    [api/metrics/series-tools →](series-tools.md)

</div>
