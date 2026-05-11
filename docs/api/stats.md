# factrix.stats

Inference-method instances + standalone statistical helpers. The
public surface is what family verbs (`bhy` / `bhy_hierarchical`) and
the upcoming Layer-B slice-test verbs accept on their `estimator=`
kwarg, plus a small set of FDR / bootstrap utilities for callers who
want to drive inference outside the dispatch chain.

The numerical implementations live in the private `factrix._stats`
package; nothing under `_stats` is part of the public API.

## Estimator catalogue

`Estimator` is a [selection-only `Protocol`](#estimator-protocol) —
each instance names *which inference path* downstream code should
read from `FactorProfile.stats`, not how to compute it. Default-
constructed instances live in `factrix.stats._ESTIMATOR_REGISTRY`
and surface through `list_estimators(scope, signal)` (top-level
factrix export).

| Class | Algorithm family | Emits | Applicable to | Use when |
|---|---|---|---|---|
| `NeweyWest` | NW Bartlett HAC | `(T_NW, P_NW)` | every cell | Default — drives `primary_p` on every PANEL / TIMESERIES procedure. |
| `HansenHodrick` | HH rectangular HAC | `(T_HH, P_HH)` | `(INDIVIDUAL, CONTINUOUS)` only | Overlapping forward returns on IC PANEL / FM PANEL — the MA(h-1) overlap structure has a closed-form rectangular-kernel SE. |
| `WaldNWCluster` | Cluster-Wald χ² (NW HAC + 1-way cluster on slice) | `(WALD_NWCL, P_WALD_NWCL)` | `(INDIVIDUAL, CONTINUOUS)` | Layer-B slice test on a stacked per-date metric panel (#176 verbs). |
| `WaldDoubleCluster` | Cluster-Wald χ² (Cameron-Gelbach-Miller two-way cluster on (date, asset)) | `(WALD_DC, P_WALD_DC)` | `(INDIVIDUAL, CONTINUOUS)` | Reserved interface — raw asset-date panel path. No verb consumes it until `factor_decomposition` lands later. |
| `BlockBootstrap` | Politis-Romano stationary or Künsch fixed block bootstrap; Politis-White auto block length | `(P_BOOT,)` | `(INDIVIDUAL, CONTINUOUS)` | Layer-B paired-diff slice test when distributional assumptions of the cluster-Wald path are uncomfortable (heavy tails, persistent shocks). |

Pass an instance to a family verb to override the default
`primary_p` lookup:

```python
from factrix.stats import HansenHodrick

# IC PANEL with overlapping forward_periods=5 → use HH instead of NW.
survivors = fx.multi_factor.bhy(profiles, estimator=HansenHodrick())
```

`BlockBootstrap` is the only Estimator whose constructor takes
configuration:

```python
from factrix.stats import BlockBootstrap

# Stationary scheme, B=999, automatic block length, fixed seed.
est = BlockBootstrap(
    block_length="auto",   # Politis-White (2004) spectral plug-in
    n_resamples=999,
    scheme="stationary",   # or "fixed" for Künsch (1989) deterministic blocks
    rng_seed=42,           # None → system entropy; realised seed written to metadata
)
```

The `scheme` is metadata, not a separate `StatCode` — both schemes
emit `P_BOOT`. Two `BlockBootstrap` instances with different `scheme`
are distinct Estimators from a verb's perspective; the verb writes
the resolved scheme + block length + seed into
`FactorProfile.metadata[StatCode.P_BOOT]`.

## StatCode pairs

`StatCode` is the canonical naming for the scalar statistics the
procedures populate on `profile.stats`. The shape is
`<KIND>_<ALGO>` — KIND names the test statistic (`T` for Student-t /
asymptotic normal, `J` for Hansen J / χ², `WALD` for Wald χ²); ALGO
names the inference algorithm or SE family (`NW`, `HH`, `NWCL`,
`DC`, `GMM`, …).

| Pair | What it is |
|---|---|
| `(T_NW, P_NW)` | Newey-West HAC t-statistic + p — the `primary_p` source the metric `evaluate()` runs populates by default. |
| `(T_HH, P_HH)` | Hansen-Hodrick rectangular-kernel HAC t + p — emitted only when `forward_periods > 1`. |
| `(WALD_NWCL, P_WALD_NWCL)` | Cluster-Wald χ² + p under NW HAC + 1-way slice cluster — emitted by Layer-B slice-test verbs. |
| `(WALD_DC, P_WALD_DC)` | Cluster-Wald χ² + p under two-way cluster on (date, asset) — reserved. |
| `(P_BOOT,)` | Block-bootstrap empirical p — singleton, no parametric test statistic to publish. |
| `(P_GMM,)` | GMM J-test p — reserved (#191). |

Diagnostic StatCodes (`FACTOR_ADF_*`, `RESID_LJUNG_BOX_*`,
`EVENT_HHI_VALUE`) follow a different naming axis; see
[`StatCode`](../reference/warning-codes.md#statcode) for the full
enum.

## FDR / bootstrap utilities

Standalone helpers that don't go through the Estimator dispatch
chain:

- **`bhy_adjust(p_values, fdr=0.05, *, n_total=None)`** —
  Benjamini-Yekutieli step-up rejection mask. Returns
  `np.ndarray[bool]` aligned to input order.
- **`bhy_adjusted_p(p_values, *, n_total=None)`** — per-hypothesis
  BHY-adjusted p-values (clipped at 1).
- **`stationary_bootstrap_resamples(values, n_bootstrap, …)`** —
  Politis-Romano (1994) resamples; emits the value matrix directly.
- **`bootstrap_mean_ci(values, *, n_bootstrap, ci, …)`** —
  stationary-bootstrap CI for a statistic (default `mean`; pass
  `statistic=` for Sharpe / median / skew).

The Layer-B FWER procedures (Holm step-down / Bonferroni /
Romano-Wolf bootstrap step-down) live as private helpers under
`factrix._stats.multiple_testing`; they ship in #153 and are
consumed by the Layer-B slice-test verbs in #176 — the verb's
default-selection logic picks Holm for time-disjoint slices and
Romano-Wolf for date-shared slices.

## Estimator protocol

```python
@runtime_checkable
class Estimator(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def applicable_to(self, scope: FactorScope, signal: Signal) -> bool: ...
    def emits_for(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> StatCode: ...
```

Selection-only by design: cell-internal `compute()` lives on a
future `ComputableEstimator(Estimator)` sub-protocol so the
common-case Estimator instance stays trivial to write.
