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

    The slope test uses Newey-West HAC covariance. The lag defaults to
    the Newey-West automatic bandwidth, floored at `forward_periods - 1`
    so overlapping forward-return windows do not understate standard
    errors.

-   __Persistent predictor diagnostic__

    ---

    The metric also runs a lightweight ADF check on the factor series.
    When `adf_p` exceeds `adf_threshold`, metadata sets
    `unit_root_suspected=True` and the result carries
    `WarningCode.PERSISTENT_REGRESSOR`. The beta is still returned; the
    warning tells you to read the slope as a persistent-regressor risk,
    not as an automatically corrected estimate.

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

    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=180, seed=0)
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

    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=180, seed=0)
    asset = raw["asset_id"].unique().sort()[0]
    panel = compute_forward_return(
        raw.filter(pl.col("asset_id") == asset),
        forward_periods=5,
    )

    full = predictive_beta(panel, forward_periods=5)
    hit = directional_hit_rate(panel, forward_periods=5)

    dates = panel.select("date").unique().sort("date")["date"].to_list()
    window_periods = 60
    step_periods = 20
    rows = []
    for end_idx in range(window_periods, len(dates) + 1, step_periods):
        window_dates = dates[end_idx - window_periods : end_idx]
        window = panel.filter(
            pl.col("date").is_between(window_dates[0], window_dates[-1])
        )
        result = predictive_beta(window, forward_periods=5)
        rows.append(
            {
                "date": window_dates[-1],
                "window_start": window_dates[0],
                "window_end": window_dates[-1],
                "beta": result.value,
                "t_stat": result.stat,
                "n_periods": result.metadata["n_periods"],
                "r_squared": result.metadata["r_squared"],
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
