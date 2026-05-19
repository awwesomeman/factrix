---
title: Estimator alternatives
---

How to swap the heteroskedasticity-and-autocorrelation-consistent (HAC) inference path that drives `primary_p`, and why
factrix's design keeps that swap study-scoped rather than per-call.

## The choice

For HAC-on-mean cells (information coefficient (IC) PANEL, FM PANEL, CAAR PANEL) you can choose
which HAC standard-error path computes `primary_p`:

| Estimator | When to pick |
|---|---|
| `NeweyWest()` *(default)* | Bartlett kernel + NW1994 auto-bandwidth + Hansen-Hodrick overlap floor. Universally applicable; produces a positive-semidefinite variance estimate by construction. |
| `HansenHodrick()` | Rectangular kernel matched to the MA(h-1) overlap structure forward returns induce. Closed-form variance under the textbook overlap assumption; no PSD guarantee on short / mildly anti-correlated samples (factrix clamps to 0 and emits `WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE`). Applicable to `(INDIVIDUAL, CONTINUOUS)` cells only. |

```python
# example pending v0.14.0 docs rewrite
```

## Why study-scoped, not per-call

Harvey, Liu, Zhu (2016, "... and the Cross-Section of Expected
Returns," RFS) is the canonical citation for cross-sectional
specification-search bias: when researchers can swap inference methods
freely after seeing results, p-values lose their guarantee. The
straightforward defence is "always use one estimator forever," but
that's stronger than what HLZ actually argue — they argue against
**post-hoc** estimator picking, not against estimator plurality
itself.

factrix's design splits the difference:

1. **The estimator is set once per study.** The factory call sites are
   the only place to wire it; there is no `evaluate(panel, cfg,
   estimator=...)` per-call kwarg by design.
2. **Provenance lands in `profile.context["estimator"]`.** Audit-time
   review can see which estimator drove each profile; running the
   same factor under both Newey-West (NW) and Hansen-Hodrick (HH) produces two profiles whose
   `context` makes the difference visible (and downstream tools that
   detect "same identity, different context" can flag the A/B path).
3. **Benjamini-Hochberg-Yekutieli (BHY) family-function false discovery rate (FDR) doesn't fan out on estimator.** Same
   hypothesis under a different estimator is not a new hypothesis —
   `bhy([nw_profile, hh_profile])` with the same `factor_id` raises
   on duplicate identity rather than silently widening the family.

The defence is "the choice is recorded and visible" rather than "the
choice is locked." Pre-registered single-factor studies can ship one
estimator; methodology papers comparing estimators ship multiple cfgs
under different `factor_id`s.

## What changes when you swap

The shape of `profile.stats` changes — only the chosen estimator's
`(stat_name, p_name)` pair populates the inference layer:

```python
# example pending v0.14.0 docs rewrite
```

This is a v0.13 BREAKING change from v0.12, which auto-side-emitted
both pairs on every IC / FM PANEL evaluate. Downstream code should
read `profile.primary_p` / `profile.primary_stat_name` (canonical,
estimator-agnostic) rather than hardcoded `stats[StatCode.T_NW]` /
`stats[StatCode.P_HH]` lookups; see the CHANGELOG entry for the full
migration note.

## Inapplicable estimators raise early

The applicability gate runs at config construction, not at
`evaluate`. Mis-matched cells fail loud:

```python
# example pending v0.14.0 docs rewrite
```

To inspect what's available for a given cell:

```python
fx.list_estimators(fx.FactorScope.INDIVIDUAL, fx.FactorSignal.CONTINUOUS)
# ['BlockBootstrap', 'HansenHodrick', 'NeweyWest', 'WaldNWCluster', 'WaldTwoWayCluster']
```

`list_estimators` returns every registered `Estimator` (selection-axis
and HAC-axis); the construction gate further narrows to `HACEstimator`
instances applicable to the cell.

## When not in scope

- **Per-factor estimator swap.** Permanently not opened — single-cfg
  decision frequency is the spec-search lock.
- **Slope-axis HAC (TS β, TS Dummy).** Single-asset ordinary least squares (OLS) regressions
  with NW HAC SE on the slope run a different math shape (`(y, x)
  → β, SE(β)`) than the series-mean `compute(series, *,
  forward_periods)` contract. A slope-axis sub-protocol is tracked
  for a future release.
- **Multi-horizon generalized method of moments (GMM) cell auto-dispatch.** The `GMM`
  `MomentEstimator` itself ships now (see below), but factrix does
  not yet auto-build the multi-horizon moment matrix from a raw
  forward-return panel. Users construct moments themselves and call
  `GMM().compute(...)` directly. The integrated cell — including
  horizon-grid specification on `cfg`, per-horizon IC construction,
  and side-emit / alternative-path dispatch semantics — is tracked
  as a follow-up to [#191](https://github.com/awwesomeman/factrix/issues/191).

## GMM moment-condition tests

The `GMM` `MomentEstimator` ships as a standalone primitive for
Hansen (1982) over-identifying-restriction tests. Pure
over-identification (`n_params = 0`) is the only mode in this
release — the null is `E[g] = 0` for all `K` moments, with the
J-statistic distributed `χ²(K)` under H₀.

```python
import numpy as np
from factrix.stats import GMM

# Build a (T, K) moment matrix yourself — e.g. per-date IC at K
# forward horizons, factor-sorted decile spreads at K horizons,
# cross-asset shared-β residuals at K asset groups, etc.
moments = build_my_moment_system(panel, ...)   # shape (T, K)

result = GMM().compute(moments, forward_periods=max_horizon)
result.j_stat        # Hansen J statistic (chi-square under H₀)
result.df            # K - n_params (n_params = 0 in this release)
result.overid_p      # right-tail χ²_df p-value
result.warnings      # SINGULAR_WEIGHT_MATRIX if Ŝ rank-deficient
```

`forward_periods` floors the long-run-covariance bandwidth at
`forward_periods - 1`, sharing the convention with `NeweyWest` /
`HansenHodrick`. Solver tuning (`max_iter`) lives on the instance:

```python
GMM(max_iter=2)   # default — Hansen's two-step efficient GMM
```

### Why no auto-dispatched cell yet

The choice of moment-condition system (multi-horizon panel?
multi-bucket spread? cross-sectional shared-β?) is a research-design
question with no canonical factrix default. The standalone primitive
ships now so users with a specific moment system can run J-tests
immediately; the integrated multi-horizon cell lands as a focused
follow-up so its design (horizon-grid spec, alternative-path vs
side-emit, EMITS_STATS extension, K-scaled `min_periods`) gets a
clean review pass.

### Wiring in the config

`cfg.moment_estimator: MomentEstimator | None` exists for the
integrated path; setting it without a corresponding cell procedure
is a no-op at evaluate-time but round-trips through
`to_dict` / `from_dict`:

```python
# example pending v0.14.0 docs rewrite
```

Pre-#191 serialized configs without the `moment_estimator` key are
read back with `moment_estimator=None`, preserving backward
compatibility.
