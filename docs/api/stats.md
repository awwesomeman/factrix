---
title: factrix.stats
---

::: factrix.stats

Inference-method instances + standalone statistical helpers under `factrix.stats`.

## Estimator catalogue

The classes registered in the estimator registry include:

| Class | Algorithm family | Emits | Applicable to |
|---|---|---|---|
| `NeweyWest` | Newey-West (NW) Bartlett HAC | `(t, p_value)` | every cell |
| `HansenHodrick` | Hansen-Hodrick (HH) rectangular HAC | `(t, p_value)` | continuous cells |
| `WaldNWCluster` | Cluster-Wald $\chi^2$ (1-way cluster on slice) | `(wald, p_value)` | continuous cells |
| `WaldTwoWayCluster` | Cluster-Wald $\chi^2$ (two-way cluster on (date, asset)) | `(wald, p_value)` | continuous cells |
| `BlockBootstrap` | Politis-Romano stationary or Künsch fixed block bootstrap | `(p_value)` | continuous cells |
| `DriscollKraay` | Driscoll-Kraay cross-section-robust HAC SE | `(t, p_value)` | continuous cells |

Use `list_estimators()` to discover all registered estimators.

---

## StatCode pairs

StatCodes identify the specific statistics populated in `MetricResult.metadata` or returned by the estimators:

- `(T_NW, P_NW)`: Newey-West HAC t-statistic and p-value.
- `(T_HH, P_HH)`: Hansen-Hodrick rectangular-kernel HAC t-statistic and p-value.
- `(WALD_NWCL, P_WALD_NWCL)`: Cluster-Wald $\chi^2$ and p-value under NW HAC + 1-way slice cluster.
- `(WALD_TWOWAY, P_WALD_TWOWAY)`: Cluster-Wald $\chi^2$ and p-value under two-way cluster.
- `(P_BOOT,)`: Block-bootstrap empirical p-value.
- `(J_GMM, P_GMM)`: Hansen GMM J-statistic and p-value.

---

## FDR / Bootstrap utilities

- **`bhy_adjust(p_values, fdr=0.05, *, n_tests=None)`**: Benjamini-Yekutieli step-up rejection mask. Returns a boolean array.
- **`bhy_adjusted_p(p_values, *, n_tests=None)`**: BHY-adjusted p-values (clipped at 1.0).
- **`stationary_bootstrap_resamples(values, n_bootstrap, ...)`**: Politis-Romano (1994) bootstrap resamples.
- **`bootstrap_mean_ci(values, *, n_bootstrap, ci, ...)`**: Stationary-bootstrap CI for a statistic.
