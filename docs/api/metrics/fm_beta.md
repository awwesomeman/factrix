---
title: factrix.metrics.fm_beta
---

::: factrix.metrics.fm_beta
    options:
      show_root_members_full_path: true
      members:
        - compute_fm_betas
        - fm_beta
        - pooled_beta
        - beta_sign_consistency

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Compute per-date FM beta series__

    ---

    Stage 1 of Fama-MacBeth: per-date cross-sectional ordinary least squares (OLS) slope
    $\beta_t$ in $R_{i,t} = \alpha_t + \beta_t \cdot \text{FactorSignal}_{i,t} + \varepsilon_{i,t}$.
    Pre-step for `fm_beta` and the descriptive
    `beta_sign_consistency` check.

-   __Mean-$\beta$ significance, Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC)__

    ---

    Stage 2 of Fama-MacBeth: $t$-test on $\mathbb{E}[\beta_t] = 0$
    with Newey-West HAC SE, bandwidth $\max(\lfloor T^{1/3} \rfloor,
    h-1)$. Default inferential test for the Individual x Continuous
    cell.

-   __Errors-in-variables correction for estimated signals__

    ---

    Set `is_estimated_factor=True` (with `factor_return_var=` where the
    factor-mimicking-portfolio return series is available) to apply the
    Shanken (1992) single-factor EIV correction (the multi-factor
    multiplicative term collapses to $1 + \hat\lambda^2/\sigma^2_f$).
    Required when the FactorSignal column is itself estimated — rolling beta,
    PCA score, ML prediction.

-   __Pooled OLS robustness check__

    ---

    `pooled_beta` runs a single regression across the stacked panel with
    cluster-robust SE (one-way on `date`, or two-way with
    `two_way_cluster_col`). When pooled $\hat\beta$ and FM
    $\hat\lambda$ disagree in sign, `profile.diagnose()` flags a
    misspecification red flag.

</div>

## Choosing a function

| Goal                                                                         | Function                |
|------------------------------------------------------------------------------|-------------------------|
| Per-date FM beta table for downstream inspection / slicing                   | `compute_fm_betas`      |
| Mean-$\beta$ significance with NW HAC SE (default Stage 2)                   | `fm_beta`          |
| Pooled OLS with cluster-robust SE (one-way on date, or two-way)              | `pooled_beta`            |
| Directional stability — fraction of periods with the expected $\beta$ sign   | `beta_sign_consistency` |

## Worked example — per-date FM beta then NW HAC significance

!!! example "compute_fm_betas → fm_beta on a synthetic cross-sectional panel"

    ```python
    import factrix as fx
    from factrix.metrics.fm_beta import compute_fm_betas, fm_beta
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=500, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    beta_df = compute_fm_betas(panel)
    print(beta_df.head())
    # ┌────────────┬───────────┐
    # │ date       ┆ beta      │
    # ├────────────┼───────────┤
    # │ 2024-01-01 ┆  0.0091   │
    # │ 2024-01-02 ┆  0.0077   │
    # │ ...        ┆  ...      │
    # └────────────┴───────────┘

    out = fm_beta(beta_df, forward_periods=5)
    print(out.value, out.stat, out.metadata["p_value"])
    # 0.0084  6.10  1.3e-09   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice FM beta summaries.

    [api/by-slice →](../by-slice.md)

-   __`slice_pairwise_test` / `slice_joint_test`__

    ---

    Cross-slice inference (Wald $\chi^2$ + Holm / Romano-Wolf adjusted $p$).

    [api/slice-test →](../slice-test.md)

-   __Statistical methods__

    ---

    NW HAC SE, Andrews bandwidth, Hansen-Hodrick overlap floor, and the
    Shanken (1992) single-factor EIV correction.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it
    (`MIN_FM_PERIODS_HARD` / `MIN_FM_PERIODS_WARN`).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
