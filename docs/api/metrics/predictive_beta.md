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
