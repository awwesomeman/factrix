---
title: factrix.metrics.ts_beta
---

::: factrix.metrics.ts_beta
    options:
      show_root_members_full_path: true
      members:
        - compute_ts_betas
        - ts_beta
        - mean_r_squared
        - ts_beta_sign_consistency
        - compute_rolling_mean_beta

<hr>

!!! info "Timeseries-mode conventions"
    Stage-1 per-asset ordinary least squares (OLS) uses **plain SE**, not heteroskedasticity-and-autocorrelation-consistent (HAC) — the dominant
    bias under a persistent predictor is Stambaugh coefficient bias,
    which HAC does not address. `FACTOR_ADF_P` is emitted on the input
    series; non-overlap resampling is **not** applied. See
    [Timeseries-mode conventions](../../reference/ts-mode-conventions.md)
    for the full rationale.

## Use cases

<div class="grid cards" markdown>

-   __Compute per-asset TS betas__

    ---

    Stage 1 of the Black-Jensen-Scholes aggregation: per-asset OLS
    $R_{i,t} = \alpha_i + \beta_i \cdot F_t + \varepsilon$ over each
    asset's full sample. Pre-step for `ts_beta` / `mean_r_squared` /
    `ts_beta_sign_consistency`. Assets with fewer than `MIN_TS_OBS`
    rows or a singular design are dropped.

-   __Cross-asset mean-$\beta$ significance__

    ---

    Stage 2 of BJS: $t = \overline{\beta} / (\mathrm{std}(\beta) /
    \sqrt{N})$ with $H_0: \mathbb{E}[\beta] = 0$ across assets.
    Cross-asset iid $t$ is used because the per-asset betas come from
    non-overlapping time-series fits and are approximately independent
    unless a strong latent common factor links them.

-   __Explanatory power across the cross-section__

    ---

    `mean_r_squared` reports $\overline{R^2}$ and `median_r_squared`
    on the per-asset fits. Low values ($< 0.05$) say the factor is
    too weak or noisy to drive individual-asset returns even when
    its cross-asset mean $\beta$ looks nonzero; large mean-vs-median
    gaps say the factor explains a small subset of assets rather than
    the cross-section as a whole.

-   __Rolling-window stability__

    ---

    `compute_rolling_mean_beta` emits a `(date, value)` series of
    rolling cross-asset mean $\beta$ at stride `window`. Output schema
    matches the time-series tools so callers can pipe rolling betas
    into `trend` / `oos`.

</div>

## Choosing a function

| Goal                                                                    | Function                          |
|-------------------------------------------------------------------------|-----------------------------------|
| Per-asset TS beta table for downstream inspection / slicing             | `compute_ts_betas`                |
| Mean-$\beta$ significance across assets (Stage 2 of BJS)                | `ts_beta`                         |
| Average explanatory power $\overline{R^2}$ across assets                | `mean_r_squared`                  |
| Direction-agnostic sign agreement on per-asset $\beta$                  | `ts_beta_sign_consistency`        |
| Rolling cross-asset mean $\beta$ series for trend / out-of-sample (OOS) pipes | `compute_rolling_mean_beta`       |
| $N=1$ degenerate-case fallback used by Profile / Factor entry points    | `ts_beta_single_asset_fallback`   |

## Worked example — per-asset TS betas then cross-asset $t$

!!! example "compute_ts_betas → ts_beta on a broadcast common-factor panel"

    ```python
    import factrix as fx
    import polars as pl
    from factrix.metrics.ts_beta import compute_ts_betas, ts_beta, mean_r_squared
    from factrix.preprocess import compute_forward_return

    # Build a panel where ``factor`` is broadcast (one value per date,
    # shared across all assets) — VIX / USD-index / NFP surprise style.
    raw = fx.datasets.make_cs_panel(
        n_assets=50, n_dates=500, ic_target=0.08, seed=2024,
    )
    common = (
        raw.group_by("date").agg(pl.col("factor").mean().alias("factor"))
    )
    panel = (
        raw.drop("factor").join(common, on="date")
    )
    panel = compute_forward_return(panel, forward_periods=5)

    betas_df = compute_ts_betas(panel)
    print(betas_df.head())
    # ┌──────────┬─────────┬─────────┬────────┬───────────┬───────┐
    # │ asset_id ┆ beta    ┆ alpha   ┆ t_stat ┆ r_squared ┆ n_obs │
    # ├──────────┼─────────┼─────────┼────────┼───────────┼───────┤
    # │ A00      ┆  0.082  ┆ 0.0001  ┆  1.91  ┆  0.018    ┆  494  │
    # │ ...      ┆  ...    ┆ ...     ┆  ...   ┆  ...      ┆  ...  │
    # └──────────┴─────────┴─────────┴────────┴───────────┴───────┘

    out = ts_beta(betas_df)
    r2  = mean_r_squared(betas_df)
    print(out.value, out.stat, out.metadata["p_value"], r2.value)
    # 0.078  6.84  4.1e-09  0.021   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`trend` / `oos`__

    ---

    Pipe `compute_rolling_mean_beta` into the series diagnostics for
    $\beta$-stability and OOS-survival reads.

    [api/metrics/trend →](trend.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice $\beta$ summaries.

    [api/by-slice →](../by-slice.md)

-   __Timeseries-mode conventions__

    ---

    Stambaugh bias, plain stage-1 SE rationale, augmented Dickey-Fuller (ADF) persistence
    discipline.

    [reference/ts-mode-conventions →](../../reference/ts-mode-conventions.md)

-   __Statistical methods__

    ---

    Cross-asset $t$, the BJS aggregation pattern, and unit-root
    discipline on the common factor.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_TS_OBS`, $N \geq 3$ for cross-asset $t$, $N \geq 2$ for sign
    consistency).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Common × Continuous landing__

    ---

    Adjacent macro-common metrics in the same cell.

    [api/metrics/common-continuous →](common-continuous.md)

</div>
