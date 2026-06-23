---
title: factrix.metrics.ts_asymmetry
---

::: factrix.metrics.ts_asymmetry
    options:
      show_root_members_full_path: true
      members:
        - ts_asymmetry

<hr>

!!! info "Timeseries-mode conventions"
    `FACTOR_ADF_P` persistence diagnostic, plain stage-1 SE rationale,
    and the `forward_periods` vs `signal_horizon` bias framing apply
    here as for the rest of the TS-mode family. See
    [Timeseries-mode conventions](../../reference/ts-mode-conventions.md).

## Use cases

<div class="grid cards" markdown>

-   __Decompose $\beta$ into long-side vs short-side response__

    ---

    Ordinary least squares (OLS) $\beta$ reports one slope and assumes a symmetric response —
    $\beta > 0$ could be "rises more on positive factor" *or* "falls
    less on negative factor". `ts_asymmetry` runs Method A (conditional
    means on $\mathrm{sign}(f)$ dummies) so the long and short legs
    are recoverable separately.

-   __Symmetric-magnitude Wald test__

    ---

    Headline is $\beta_{\text{long}} + \beta_{\text{short}}$ — 0 under
    perfect symmetry, positive when the long side dominates. Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC)
    Wald on $H_0: \beta_{\text{long}} + \beta_{\text{short}} = 0$
    handles the autocorrelation induced by overlapping forward
    returns; Welch $t$ is intentionally avoided because its iid
    assumption breaks under `forward_periods > 1`.

-   __Piecewise-slope check (Method B)__

    ---

    When each side has $\geq 2$ distinct factor values, Method B fits
    $r = \alpha + \beta_{\text{pos}} \max(f, 0) + \beta_{\text{neg}}
    \min(f, 0)$ and tests $H_0: \beta_{\text{pos}} = \beta_{\text{neg}}$.
    Distinguishes a magnitude asymmetry (Method A) from a slope
    asymmetry. Gate C below the cardinality floor records
    `method_b_skipped` and the reason; Method A already carries the
    full information for categorical / binary signals.

</div>

## Worked example — sign-asymmetric slopes on a common-factor panel

!!! example "broadcast common factor → ts_asymmetry (Methods A + B)"

    ```python
    import factrix as fx
    import polars as pl
    from factrix.metrics.ts_asymmetry import ts_asymmetry
    from factrix.preprocess import compute_forward_return

    # Build a panel whose ``factor`` is broadcast (one value per date,
    # shared across all assets) — VIX / NFP-surprise style.
    raw = fx.datasets.make_cs_panel(
        n_assets=50, n_dates=1000, ic_target=0.08, seed=2024,
    )
    common = raw.group_by("date").agg(pl.col("factor").mean().alias("factor"))
    panel  = raw.drop("factor").join(common, on="date")
    panel  = compute_forward_return(panel, forward_periods=5)

    out = ts_asymmetry(panel, forward_periods=5)
    print(out.value, out.stat, out.p_value)
    # 0.00021  1.83  0.067   (approximate; method A magnitude)
    print(out.metadata["beta_long"], out.metadata["beta_short"],
          out.metadata["abs_short_over_long"])
    # 0.00088  -0.00067  0.76
    print(out.metadata.get("beta_pos"), out.metadata.get("beta_neg"),
          out.metadata.get("p_wald_slopes"))
    # 0.00121  -0.00102  0.18   (Method B; ~null if Gate C passed)
    ```

## See also

<div class="grid cards" markdown>

-   __`ts_beta`__

    ---

    Symmetric linear $\beta$ on the same panel; pair when both
    direction and magnitude asymmetry matter.

    [api/metrics/ts_beta →](ts_beta.md)

-   __`ts_quantile_spread`__

    ---

    Bucketed conditional means — strictly more flexible than the
    two-flavour split, at the cost of $K$ free parameters and
    the `n_distinct(factor) >= n_groups * 2` gate.

    [api/metrics/ts_quantile →](ts_quantile.md)

-   __`event_quality` family__

    ---

    Where the input gate redirects when the factor is single-sided
    (no positive or no negative observations): `event_hit_rate` /
    `event_ic` / `profit_factor`.

    [api/metrics/event_quality →](event_quality.md)

-   __Statistical methods__

    ---

    NW HAC SE, Andrews bandwidth, Hansen-Hodrick overlap floor, and
    the Wald-on-linear-restriction framing for both Method A and
    Method B.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Timeseries-mode conventions__

    ---

    `FACTOR_ADF_P`, plain stage-1 SE rationale, `forward_periods` vs
    `signal_horizon` bias framing.

    [reference/ts-mode-conventions →](../../reference/ts-mode-conventions.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the gates (Gate B: two-sided factor;
    Gate C: $\geq 2$ distinct values per side for Method B).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Common × Continuous landing__

    ---

    Adjacent macro-common metrics in the same cell.

    [api/metrics/common-continuous →](common-continuous.md)

</div>
