---
title: factrix.metrics.predictive_beta
---

::: factrix.metrics.predictive_beta
    options:
      show_root_members_full_path: true
      members:
        - predictive_beta

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Single-asset dense predictive slope__

    ---

    `predictive_beta` is the `(*, DENSE, TIMESERIES)` metric for a
    one-asset panel. It fits `forward_return ~ factor` and tests whether
    the slope differs from zero.

-   __Not a `common_beta` fallback__

    ---

    `common_beta` tests the cross-asset mean of per-asset betas and remains
    `PANEL`-only. `predictive_beta` is the explicit single-series
    predictive regression.

-   __HAC inference for overlapping returns__

    ---

    At `overlap_periods > 1`, the headline bias-corrected slope test uses
    the project's scalar HAR bandwidth and effective degrees of freedom.
    Its Amihud-Hurvich generated-regressor covariance does not apply the
    separate finite-sample variance scale used by scalar Wald consumers.
    At `overlap_periods = 1` it uses the original homoskedastic covariance.
    The raw OLS reference retained in metadata stays on the narrower
    Newey-West bandwidth and is labelled as uncorrected.
    Metadata reports the headline branch directly: `hac_applied=False` and
    `har_lags=None` at `h = 1`; both switch to the HAR path at `h > 1`. The
    `method` string names the covariance that produced the headline test, and
    `har_lags` is the bandwidth the kernel ran at — the augmented design is
    `n_periods` rows, so a long horizon can leave it narrower than the
    bandwidth resolved from the full series.

-   __Persistent predictor diagnostic__

    ---

    The metric also runs a lightweight ADF check on the factor series.
    When `adf_p` exceeds `adf_threshold`, metadata sets
    `unit_root_suspected=True` and the result carries
    `WarningCode.PERSISTENT_REGRESSOR`. The beta is still returned; the
    warning tells you to read the slope as a persistent-regressor risk.
    The Stambaugh bias correction still applies; this diagnostic describes
    the remaining inference regime, not whether the correction ran.

    Read it as a verdict on the **regressor**, not on this test's size.
    Under a classic Stambaugh design (AR(1) `phi = 0.99`,
    `corr(u_x, eps_r) = -0.9`, true beta = 0, 300 seeds) the flag and the
    distortion move apart as the sample grows: at `T = 500` it fires on ~75%
    of draws while the test rejects 13% (`h = 1`) / 35% (`h = 21`) at a
    nominal 5%; at `T = 2500` the ADF test can reject the unit root, the flag
    goes silent (~0% of draws) and the test still rejects 9% / 14%. A silent
    flag is not evidence of an unbiased slope. The Stambaugh bias itself is
    corrected unconditionally, so the flag is about the *regressor*, not
    about whether `value` still carries the bias.

-   __Overlap and residual persistence__

    ---

    Three further screens read the sample the standard error actually has.
    `WarningCode.OVERLAPPING_PREDICTIVE_INFERENCE` is present whenever
    `overlap_periods > 1`: the corrected test remains 7.5-14.5% oversized
    across the measured null cells, including independent-regressor designs.
    The estimate and p-value remain available, but the code makes the need
    for a raised hurdle and an `h = 1` or genuinely non-overlapping
    sensitivity check machine-readable. This is separate from
    `EVENT_WINDOW_OVERLAP`, which belongs to event studies and removes
    overlapping events before testing.

    `WarningCode.SERIAL_CORRELATION_DETECTED` fires when the **reported**
    model's residuals — `y - alpha - value * factor` over the `n_obs` rows,
    so the bias-corrected fit's residuals whenever the correction applies —
    have a lag-1 autocorrelation above `PERSISTENT_SERIES_AUTOCORR` (0.3) —
    the same rule `fm_beta` and the series-mean inference members
    apply. `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` reads
    `n_periods_effective = n_periods // overlap_periods`, not the raw row
    count: `n` overlapping rows carry about `n / h` independent observations
    while the HAC lag floor rises with `h`. At `T = 120`, `h = 21` the
    regression runs on 98 rows with a Bartlett lag of 20 and rejects 17.5% of
    null draws at a nominal 5%, a regime the raw-`n` gate could not see.

-   __Stability is a workflow__

    ---

    `predictive_beta` returns the full-sample HAC slope test. Rolling or
    expanding beta checks should be treated as descriptive stability
    diagnostics with pre-declared windows, not as a second family of
    p-values to rank or feed into multiple-testing correction.

</div>

## Worked example

!!! example "single-asset panel -> predictive_beta"

    ```python
    import polars as pl
    import factrix as fx
    from factrix.metrics import predictive_beta
    from factrix.preprocess import compute_forward_return

    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=180, rng=0)
    asset = raw["asset_id"].unique().sort()[0]
    panel = compute_forward_return(
        raw.filter(pl.col("asset_id") == asset),
        forward_periods=5,
    )

    out = fx.evaluate(
        panel,
        metrics={"predictive_beta": predictive_beta()},
        factor_cols=["factor"],
        forward_periods=5,
    )["factor"].metrics["predictive_beta"]

    print(out.value, out.stat, out.p_value, out.metadata["unit_root_suspected"])
    ```

## Stability workflow

For a single-asset dense factor, the first question is still the full-sample
HAC slope: does `factor_t` have a statistically legible relation to
`forward_return_{t+h}`? Stability checks come after that and should answer a
different question: whether the slope is broadly persistent through time or
mostly supported by one segment of the sample.

Use pre-declared windows and read the rolling betas descriptively. The windowed
calls below reuse the same `predictive_beta` estimator, but the resulting
`t_stat` / `p_value` values are not independent tests because overlapping
windows share observations.

!!! example "rolling beta series as a diagnostic"

    ```python
    import polars as pl
    import factrix as fx
    from factrix.metrics import directional_hit_rate, predictive_beta
    from factrix.preprocess import compute_forward_return

    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=180, rng=0)
    asset = raw["asset_id"].unique().sort()[0]
    panel = compute_forward_return(
        raw.filter(pl.col("asset_id") == asset),
        forward_periods=5,
    )

    full = predictive_beta(panel, overlap_periods=5)
    hit = directional_hit_rate(panel, overlap_periods=5)

    dates = panel.select("date").unique().sort("date")["date"].to_list()
    window_periods = 60
    step_periods = 20
    rows = []
    for end_idx in range(window_periods, len(dates) + 1, step_periods):
        window_dates = dates[end_idx - window_periods : end_idx]
        window = panel.filter(
            pl.col("date").is_between(window_dates[0], window_dates[-1])
        )
        result = predictive_beta(window, overlap_periods=5)
        rows.append(
            {
                "date": window_dates[-1],
                "window_start": window_dates[0],
                "window_end": window_dates[-1],
                "beta": result.value,
                "t_stat": result.stat,
                "n_periods": result.metadata["n_periods"],
                "r_squared_ols": result.metadata["r_squared_ols_uncorrected"],
            }
        )

    beta_series = pl.DataFrame(rows)
    reference_sign = 1.0 if full.value >= 0.0 else -1.0
    stability = beta_series.select(
        ((pl.col("beta") * reference_sign) > 0).mean().alias("sign_consistency"),
        pl.col("beta").median().alias("median_beta"),
        pl.col("beta").last().alias("recent_beta"),
        pl.col("beta").min().alias("min_beta"),
        pl.col("beta").max().alias("max_beta"),
    )

    print("full beta:", full.value, "p:", full.p_value)
    print("directional hit:", hit.value, "p:", hit.p_value)
    print(stability)
    ```

Read this output as a stability profile:

- `full.value` / `full.p_value` is the canonical single-asset dense inference.
- `directional_hit_rate` checks whether the factor gets the return sign right.
- `sign_consistency` asks how often rolling betas keep the full-sample sign.
- `recent_beta`, `median_beta`, `min_beta`, and `max_beta` show whether the
  signal is decaying, flipping, or concentrated in one segment.

Avoid turning this workflow into automatic model selection. Do not choose the
window after looking at the strongest beta, do not feed overlapping-window
`p_value` values into `bhy`, and do not interpret `sign_consistency` as a
formal structural-break test. If the stability profile points to a regime
change, split that regime hypothesis explicitly and test it as a separate
research design.
