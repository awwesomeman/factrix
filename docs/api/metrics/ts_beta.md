# ts_beta

Per-asset time-series β to a macro common factor (VIX, USD index,
etc.), then cross-asset *t* on the β distribution. Includes mean R²,
rolling-window β stability, and sign consistency.

!!! info "TS-regression conventions"
    PANEL stage-1 per-asset OLS uses **plain SE**, not HAC — the
    dominant bias under a persistent predictor is Stambaugh
    coefficient bias, which HAC does not address. TIMESERIES dispatch
    (`N == 1`) instead applies NW HAC directly on the single series.
    `FACTOR_ADF_P` is emitted in both modes; non-overlap resampling
    is not applied in either. See
    [TS-regression conventions](../../reference/ts-mode-conventions.md)
    for the full rationale.

::: factrix.metrics.ts_beta
    options:
      members:
        - ts_beta
        - mean_r_squared
        - ts_beta_sign_consistency
        - compute_rolling_mean_beta
        - compute_ts_betas

## See also

- [Reference § Metric applicability](../../reference/metric-applicability.md) — when this metric applies and sample-size guards.
- [Reference § Statistical methods](../../reference/statistical-methods.md) — HAC SE, FDR, robust-scale, unit-root disciplines that govern the inference.
- [Common continuous landing page](common-continuous.md) — adjacent macro-factor metrics.
