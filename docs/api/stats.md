---
title: factrix.stats
---

::: factrix.stats

Inference-method instances + standalone statistical helpers under `factrix.stats`.

## Estimator catalogue

The selection-only estimator classes under `factrix.stats`:

| Class | Algorithm family | Description |
|---|---|---|
| `WaldNWCluster` | Cluster-Wald $\chi^2$ (1-way cluster on slice) | 1-way cluster-robust Wald contrast test |
| `WaldTwoWayCluster` | Cluster-Wald $\chi^2$ (two-way cluster on (date, asset)) | 2-way cluster-robust Wald contrast test |
| `BlockBootstrap` | Politis-Romano stationary or Künsch fixed block bootstrap | Empirical p-value block bootstrap for paired contrast |
| `DriscollKraay` | Driscoll-Kraay cross-section-robust HAC SE | Selection-only base Estimator for Driscoll-Kraay robust standard errors |

---

## Multiplicity / bootstrap utilities

- **`bhy_adjust(p_values, fdr=0.05, *, n_tests=None)`**: Benjamini-Yekutieli step-up rejection mask. Returns a boolean array.
- **`bhy_adjusted_p(p_values, *, n_tests=None)`**: BHY-adjusted p-values (clipped at 1.0).
- **`holm_adjusted_p(p_values, *, n_tests=None)`**: Holm step-down FWER-adjusted p-values under arbitrary dependence. Use FWER when a search selects one winner or every retained hypothesis must avoid any false positive; use FDR when retaining a batch and controlling its expected false-discovery proportion.
- **`romano_wolf_adjusted_p(statistics, bootstrap_statistics, *, one_sided=False)`**: Dependence-aware step-down max-t FWER-adjusted p-values. This expert primitive requires every searched hypothesis and each bootstrap row to be a joint, null-centred draw with the same studentization as the observed statistics; independently resampled or omitted columns are invalid. Its empirical p-value resolution is `1 / (B + 1)`. Use Holm when a valid joint bootstrap family is unavailable.
- **`stationary_bootstrap_resamples(values, n_bootstrap, ...)`**: Politis-Romano (1994) bootstrap resamples. An aligned `(T, m)` per-period statistic matrix is resampled with common row indices and returns `(B, T, m)`, preserving cross-hypothesis dependence for Romano-Wolf; separate per-column calls do not.
- **`bootstrap_mean_ci(values, *, n_bootstrap, ci, ...)`**: Stationary-bootstrap CI for a statistic.
