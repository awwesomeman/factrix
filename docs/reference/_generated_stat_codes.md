| StatCode | Trigger / meaning |
|---|---|
| `mean` | Cell primary point estimate (interpretation per `profile.config.metric`: IC mean, FM λ mean, CAAR event-only mean, or TS β / E[β]). |
| `t_nw` | Newey-West HAC t-stat on the cell primary estimate. Implementation convention lives in `factrix.stats.NeweyWest`. |
| `p_nw` | Two-sided p-value from the Newey-West HAC t-test on the cell primary estimate. Sibling of `T_NW`. |
| `t_hh` | Hansen-Hodrick (1980) rectangular-kernel HAC t-stat on the cell primary estimate. Sibling of `T_NW`; uses `Var(mean) = (γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n` instead of NW's Bartlett kernel. |
| `p_hh` | Two-sided p-value from the Hansen-Hodrick (1980) rectangular-kernel HAC t-test on the cell primary estimate. Implementation convention lives in `factrix.stats.HansenHodrick`. |
| `j_gmm` | Hansen (1982) GMM J-statistic for over-identifying moment restrictions; chi-square distributed under H₀ with `df = n_moments - n_params`. Implementation convention lives in `factrix.stats.GMM`. |
| `p_gmm` | Right-tail p-value from the Hansen (1982) GMM J-test (`1 - χ²_df.cdf(J_GMM)`). Sibling under the (J_GMM, P_GMM) algorithm-pair convention; computed by `factrix.stats.GMM`. |
| `wald_nwcl` | Wald χ² statistic for a linear restriction on slice contrasts / joint coefficients, computed under NW Bartlett HAC plus one-way cluster on the slice grouping. Implementation convention lives in `factrix.stats.WaldNWCluster`. |
| `p_wald_nwcl` | P-value from `WALD_NWCL`. Sibling under the (WALD_NWCL, P_WALD_NWCL) algorithm-pair convention. |
| `wald_twoway` | Wald χ² statistic for a linear restriction on a panel coefficient vector, computed under two-way cluster on (date, asset) (Cameron-Gelbach-Miller 2011). Implementation convention lives in `factrix.stats.WaldTwoWayCluster`. |
| `p_wald_twoway` | P-value from `WALD_TWOWAY`. Sibling under the (WALD_TWOWAY, P_WALD_TWOWAY) algorithm-pair convention. |
| `p_boot` | Empirical two-sided p-value from a block-bootstrap resample of a paired-diff statistic. Implementation convention lives in `factrix.stats.BlockBootstrap` (Politis-Romano stationary or Künsch fixed scheme; Politis-White auto block length). Single key for both schemes — scheme choice is metadata, not StatCode. |
| `factor_adf_tau` | ADF τ statistic on the factor input series (constant-only specification); fed to the MacKinnon 1996 response-surface for `FACTOR_ADF_P`. |
| `factor_adf_p` | ADF unit-root test p-value on the factor input series (MacKinnon 1996 response-surface; constant-only specification). p > 0.05 flags persistent regressor regime. |
| `resid_ljung_box_q` | Ljung-Box Q statistic on regression residuals (TS-dummy single-asset path); compared against χ²(h) for `RESID_LJUNG_BOX_P`. |
| `resid_ljung_box_p` | Ljung-Box p-value on residual autocorrelation (TS-dummy single-asset path); p < 0.05 flags under-set NW lag. |
| `event_hhi_value` | Herfindahl concentration of event counts across equal-width period bins on the panel's time axis; high values flag time-axis clumping. Does not measure within-asset event clustering. |
