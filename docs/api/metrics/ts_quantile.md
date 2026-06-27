---
title: factrix.metrics.ts_quantile
---

::: factrix.metrics.ts_quantile
    options:
      show_root_members_full_path: true
      members:
        - ts_quantile_spread

<hr>

!!! info "Timeseries-mode conventions"
    Plain stage-1 SE rationale and the `forward_periods` vs
    `signal_horizon` bias framing apply here as for the rest of the
    Common × Continuous family. See
    [Timeseries-mode conventions](../../reference/ts-mode-conventions.md).

## Use cases

<div class="grid cards" markdown>

-   __Detect non-linear factor → return shape__

    ---

    Linear ordinary least squares (OLS) $\beta$ reports a single slope and fails on
    U-shape / inverted-U / extreme-only signals. `ts_quantile_spread`
    aggregates the panel to a per-date $(\_f, \_r)$ series, buckets
    `_f` into $K$ historical quantiles, and reads the conditional
    mean return per bucket — preserves whatever shape the relationship
    has.

-   __Top-bottom spread significance with Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC)__

    ---

    Headline is $\beta_{K-1} - \beta_0$ — the conditional mean
    difference between the top and bottom quantile of factor history.
    Inference is a Wald $\chi^2$ on $H_0: \beta_{K-1} = \beta_0$ with
    Newey-West HAC covariance; kernel choice is consistent with
    `ts_asymmetry` so cross-method $p$-values stay comparable under
    overlapping forward returns.

-   __Rank-monotonicity diagnostic across buckets__

    ---

    `metadata["spearman_rho"]` / `spearman_p` give a non-parametric
    Spearman rank check on the bucket-mean sequence
    $(\beta_0, \ldots, \beta_{K-1})$. Catches a monotone shape that
    the top-bottom Wald would conflate with a wider U.

</div>

## Worked example — quantile-bucketed conditional means on a common-factor panel

!!! example "broadcast common factor → ts_quantile_spread"

    ```python
    import factrix as fx
    import polars as pl
    from factrix.metrics.ts_quantile import ts_quantile_spread
    from factrix.preprocess import compute_forward_return

    # Build a panel whose ``factor`` is broadcast (one value per date,
    # shared across all assets) — VIX / USD-index style.
    raw = fx.datasets.make_cs_panel(
        n_assets=50, n_dates=1000, ic_target=0.08, seed=2024,
    )
    common = raw.group_by("date").agg(pl.col("factor").mean().alias("factor"))
    panel  = raw.drop("factor").join(common, on="date")
    panel  = compute_forward_return(panel, forward_periods=5)

    out = ts_quantile_spread(panel, n_groups=5, forward_periods=5)
    print(out.value, out.stat, out.p_value)
    # 0.0018  3.20  0.0014   (approximate)
    print(out.metadata["spearman_rho"], out.metadata["spearman_p"])
    # 0.90  0.037   (approximate; positive ⇒ monotone shape)
    for b in out.metadata["buckets"]:
        print(b)
    # {"idx": 0, "mean_return": -0.00091, "n": 199}
    # ...
    # {"idx": 4, "mean_return":  0.00091, "n": 199}
    ```

## See also

<div class="grid cards" markdown>

-   __`ts_beta` / `ts_asymmetry`__

    ---

    Linear $\beta$ and signed-slope asymmetry diagnostics on the same
    broadcast-factor panel. Use `ts_quantile_spread` when a Spearman
    rank check or U-shape detection matters; the others assume a
    monotone or piecewise-linear response.

    [api/metrics/ts_beta →](ts_beta.md)

-   __`event_quality` family__

    ---

    Where the input gate redirects when the factor cannot sustain
    quantile cuts (binary / sparse signals): `event_hit_rate` /
    `event_ic` / `profit_factor` / `event_skewness`.

    [api/metrics/event_quality →](event_quality.md)

-   __`by_slice` / `slice_pairwise_test`__

    ---

    Axis-agnostic slice dispatcher and pairwise Wald (Holm /
    Romano-Wolf adjusted $p$) for per-slice spread comparisons.

    [api/by-slice →](../by-slice.md)

-   __Statistical methods__

    ---

    NW HAC SE, Newey-West (1994) auto bandwidth, Hansen-Hodrick overlap floor, and
    the Wald-on-linear-restriction framing.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Timeseries-mode conventions__

    ---

    Plain stage-1 SE rationale, `forward_periods` vs `signal_horizon`
    bias framing.

    [reference/ts-mode-conventions →](../../reference/ts-mode-conventions.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_PORTFOLIO_PERIODS_HARD`, `n_distinct(factor) >= n_groups * 2`).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Common × Continuous landing__

    ---

    Adjacent macro-common metrics in the same cell.

    [api/metrics/common-continuous →](common-continuous.md)

</div>
