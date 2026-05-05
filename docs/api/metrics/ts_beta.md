# ts_beta

Per-asset time-series β to a macro common factor (VIX, USD index,
etc.), then cross-asset *t* on the β distribution. Includes mean R²,
rolling-window β stability, and sign consistency.

::: factrix.metrics.ts_beta
    options:
      members:
        - ts_beta
        - mean_r_squared
        - ts_beta_sign_consistency
        - compute_rolling_mean_beta
        - compute_ts_betas
