---
title: factrix.stats
---

Inference-method instances + standalone statistical helpers. The
public surface is what screening functions (`bhy` / `bhy_hierarchical`) and
the slice-test functions (`slice_pairwise_test` / `slice_joint_test`)
accept on their `estimator=` kwarg, plus a small set of FDR /
bootstrap utilities for callers who
want to drive inference outside the dispatch chain.

The numerical implementations live in the private `factrix._stats`
package; nothing under `_stats` is part of the public API.

## Estimator catalogue

`Estimator` is the base [`Protocol`](#estimator-protocol) — each
instance names *which inference path* downstream code reads from
`FactorProfile.stats`. `HACEstimator(Estimator)` is the sub-protocol
adding cell-internal `compute(series, *, forward_periods) ->
InferenceResult` for HAC-on-mean estimators; pass instances to
`AnalysisConfig.estimator=` for evaluate-time inference dispatch.
Default-constructed instances live in
`factrix.stats._ESTIMATOR_REGISTRY` and surface through
`list_estimators(scope, signal)` (top-level factrix export).

| Class | Protocol | Algorithm family | Emits | Applicable to | Use when |
|---|---|---|---|---|---|
| `NeweyWest` | `HACEstimator` | NW Bartlett HAC | `(T_NW, P_NW)` | every cell | Default — drives `primary_p` on every PANEL / TIMESERIES procedure. |
| `HansenHodrick` | `HACEstimator` | HH rectangular HAC | `(T_HH, P_HH)` | `(INDIVIDUAL, CONTINUOUS)` only | Overlapping forward returns on IC PANEL / FM PANEL — the MA(h-1) overlap structure has a closed-form rectangular-kernel SE. Pass via `AnalysisConfig.individual_continuous(estimator=HansenHodrick())` to drive `primary_p` from the HH path instead of NW. |
| `WaldNWCluster` | Cluster-Wald χ² (NW HAC + 1-way cluster on slice) | `(WALD_NWCL, P_WALD_NWCL)` | `(INDIVIDUAL, CONTINUOUS)` | Slice test on a stacked per-date metric panel (#176 functions). |
| `WaldTwoWayCluster` | Cluster-Wald χ² (Cameron-Gelbach-Miller two-way cluster on (date, asset)) | `(WALD_TWOWAY, P_WALD_TWOWAY)` | `(INDIVIDUAL, CONTINUOUS)` | Reserved interface — raw asset-date panel path. No function consumes it until `factor_decomposition` lands later. |
| `BlockBootstrap` | Politis-Romano stationary or Künsch fixed block bootstrap; Politis-White auto block length | `(P_BOOT,)` | `(INDIVIDUAL, CONTINUOUS)` | Paired-diff slice test when distributional assumptions of the cluster-Wald path are uncomfortable (heavy tails, persistent shocks). |

!!! warning "`WaldTwoWayCluster` is a reserved interface"
    The class ships in #153 so the `(WALD_TWOWAY, P_WALD_TWOWAY)` StatCode
    pair has a stable home, but no function populates `profile.stats`
    with `P_WALD_TWOWAY` until `factor_decomposition` lands. Calling
    `bhy(estimator=WaldTwoWayCluster())` against a profile produced
    by `evaluate()` raises a missing-stat error pointing at the
    precondition.

### Picking an Estimator

| Question | Estimator |
|---|---|
| Default single-series significance on `evaluate()` output | `NeweyWest` |
| Overlapping forward returns (`forward_periods > 1`) on IC PANEL / FM PANEL | `HansenHodrick` |
| Slice contrast on per-date IC / FM (regime, sector, decile) | `WaldNWCluster` |
| Slice paired-diff on heavy-tailed / persistent series, distributional assumptions uncomfortable | `BlockBootstrap` |
| Raw asset-date panel inference (factor × slice interaction) | `WaldTwoWayCluster` (reserved) |

Pass an instance to a screening function to override the default
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
are distinct Estimators from a function's perspective; the function writes
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
| `(WALD_NWCL, P_WALD_NWCL)` | Cluster-Wald χ² + p under NW HAC + 1-way slice cluster — emitted by the slice-test functions. |
| `(WALD_TWOWAY, P_WALD_TWOWAY)` | Cluster-Wald χ² + p under two-way cluster on (date, asset) — reserved. |
| `(P_BOOT,)` | Block-bootstrap empirical p — singleton, no parametric test statistic to publish. |
| `(J_GMM, P_GMM)` | Hansen (1982) GMM J-statistic + right-tail p (`1 - χ²_df.cdf(J)`) on a moment-condition system. Populated by `factrix.stats.GMM` (#191); see [Estimator alternatives](estimator-alternatives.md#gmm-moment-condition-tests) for usage. |

Diagnostic StatCodes (`FACTOR_ADF_*`, `RESID_LJUNG_BOX_*`,
`EVENT_HHI_VALUE`) follow a different naming axis; see
[`StatCode`](../reference/warning-codes.md#statcode) for the full
enum.

## FDR / bootstrap utilities

Standalone helpers that don't go through the Estimator dispatch
chain:

- **`bhy_adjust(p_values, fdr=0.05, *, n_tests=None)`** —
  Benjamini-Yekutieli step-up rejection mask. Returns
  `np.ndarray[bool]` aligned to input order.
- **`bhy_adjusted_p(p_values, *, n_tests=None)`** — per-hypothesis
  BHY-adjusted p-values (clipped at 1).
- **`stationary_bootstrap_resamples(values, n_bootstrap, …)`** —
  Politis-Romano (1994) resamples; emits the value matrix directly.
- **`bootstrap_mean_ci(values, *, n_bootstrap, ci, …)`** —
  stationary-bootstrap CI for a statistic (default `mean`; pass
  `statistic=` for Sharpe / median / skew).

The FWER procedures (Holm step-down / Bonferroni / Romano-Wolf
bootstrap step-down) live as private helpers under
`factrix._stats.multiple_testing`; they ship in #153 and are
consumed by the slice-test functions in #176 — the function's
default-selection logic picks Holm for time-disjoint slices and
Romano-Wolf for date-shared slices.

## Estimator protocol

Three-layer protocol: base `Estimator` for family-function selection
(`bhy(profiles, estimator=...)`); `HACEstimator(Estimator)` for
evaluate-time HAC-on-mean dispatch (`AnalysisConfig.estimator=`);
`MomentEstimator(Estimator)` for over-identifying-restriction tests
on a multivariate moment system (`AnalysisConfig.moment_estimator=`).

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


@runtime_checkable
class HACEstimator(Estimator, Protocol):
    @property
    def min_periods(self) -> int: ...
    def compute(
        self,
        series: np.ndarray,
        *,
        forward_periods: int,
    ) -> InferenceResult: ...


@runtime_checkable
class MomentEstimator(Estimator, Protocol):
    @property
    def min_periods(self) -> int: ...
    def compute(
        self,
        moments: np.ndarray,         # (T, K) moment matrix
        *,
        forward_periods: int,        # overlap horizon — floors the LRCov bandwidth
    ) -> GMMResult: ...
```

`NeweyWest` and `HansenHodrick` implement `HACEstimator`; `GMM`
implements `MomentEstimator`; the slice-test instances
(`WaldNWCluster` / `WaldTwoWayCluster` / `BlockBootstrap`) implement
only the selection base since their compute paths are multivariate
(cross-asset / cross-slice) rather than mean-on-series or
moment-system. A slope-axis HAC sub-protocol (TS β / TS Dummy) is
tracked separately rather than overloading `HACEstimator.compute`.

### `InferenceResult`

`HACEstimator.compute` returns a frozen dataclass carrying the
inference layer of a `FactorProfile` (procedure stitches descriptive
stats like `MEAN` on top):

```python
@dataclass(frozen=True, slots=True)
class InferenceResult:
    stat: float                          # t-statistic
    p: float                             # two-sided p-value
    stat_name: StatCode                  # T_NW / T_HH / ...
    p_name: StatCode                     # P_NW / P_HH / ...
    metadata: Mapping[str, Any]          # {"nw_lags": k} / {"kernel": ..., "variance_clamped": bool}
    warnings: frozenset[WarningCode]
```

### `GMMResult`

`MomentEstimator.compute` returns a frozen dataclass for the
over-identifying-restriction test on a moment-condition system:

```python
@dataclass(frozen=True, slots=True)
class GMMResult:
    j_stat: float                        # Hansen J statistic
    df: int                              # n_moments - n_params
    overid_p: float                      # 1 - χ²_df.cdf(j_stat)
    n_moments: int
    n_params: int                        # 0 in current release (pure overid)
    metadata: Mapping[str, Any]          # {"weight_matrix_iter": 2, "weight_singular": False, ...}
    warnings: frozenset[WarningCode]
```

Unlike `InferenceResult`, no `stat_name` / `p_name` field — the type
itself implies the `(StatCode.J_GMM, StatCode.P_GMM)` pair, and cell
procedures key `FactorProfile.stats` accordingly.

### `get_estimator(name) -> Estimator`

Registry lookup helper used by `AnalysisConfig.from_dict` to
rehydrate `cfg.estimator` / `cfg.moment_estimator` from their
serialized name strings. Raises [`UnknownEstimatorError`][factrix.UnknownEstimatorError] if `name`
is not registered; the error message lists every available estimator.
