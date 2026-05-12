# Estimator alternatives

How to swap the HAC inference path that drives `primary_p`, and why
factrix's design keeps that swap study-scoped rather than per-call.

## The choice

For HAC-on-mean cells (IC PANEL, FM PANEL, CAAR PANEL) you can choose
which HAC standard-error path computes `primary_p`:

| Estimator | When to pick |
|---|---|
| `NeweyWest()` *(default)* | Bartlett kernel + NW1994 auto-bandwidth + Hansen-Hodrick overlap floor. Universally applicable; produces a positive-semidefinite variance estimate by construction. |
| `HansenHodrick()` | Rectangular kernel matched to the MA(h-1) overlap structure forward returns induce. Closed-form variance under the textbook overlap assumption; no PSD guarantee on short / mildly anti-correlated samples (factrix clamps to 0 and emits `WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE`). Applicable to `(INDIVIDUAL, CONTINUOUS)` cells only. |

```python
import factrix as fx
from factrix.stats import HansenHodrick

cfg = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.IC,
    forward_periods=5,
    estimator=HansenHodrick(),       # default NeweyWest()
)
profile = fx.evaluate(panel, cfg)

profile.primary_stat_name             # StatCode.T_HH
profile.primary_p                     # HH p-value, drives bhy / bonferroni / etc.
profile.context["estimator"]          # "HansenHodrick" — audit trail
```

The estimator is part of the cfg, so it serializes with `to_dict()`
and rehydrates via `AnalysisConfig.from_dict(...)`:

```python
cfg.to_dict()
# {"scope": "individual", "signal": "continuous", "metric": "ic",
#  "forward_periods": 5, "estimator": "HansenHodrick"}

# Round-trip is exact; missing-key legacy dicts fall back to NeweyWest()
restored = fx.AnalysisConfig.from_dict(cfg.to_dict())
restored == cfg                       # True
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

1. **`AnalysisConfig.estimator` is set once per study.** The factory
   call sites are the only place to wire it; there is no
   `evaluate(panel, cfg, estimator=...)` per-call kwarg by design.
2. **Provenance lands in `profile.context["estimator"]`.** Audit-time
   review can see which estimator drove each profile; running the
   same factor under both NW and HH produces two profiles whose
   `context` makes the difference visible (and downstream tools that
   detect "same identity, different context" can flag the A/B path).
3. **BHY family-verb FDR doesn't fan out on estimator.** Same
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
# default cfg — NW path
profile = fx.evaluate(panel, fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC))
sorted(profile.stats)         # [MEAN, P_NW, T_NW]

# HH cfg
profile = fx.evaluate(
    panel,
    fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC, estimator=HansenHodrick()),
)
sorted(profile.stats)         # [MEAN, P_HH, T_HH]
```

This is a v0.13 BREAKING change from v0.12, which auto-side-emitted
both pairs on every IC / FM PANEL evaluate. Downstream code should
read `profile.primary_p` / `profile.primary_stat_name` (canonical,
estimator-agnostic) rather than hardcoded `stats[StatCode.T_NW]` /
`stats[StatCode.P_HH]` lookups; see the CHANGELOG entry for the full
migration note.

## Inapplicable estimators raise early

The applicability gate runs at `AnalysisConfig` construction, not at
`evaluate`. Mis-matched cells fail loud:

```python
fx.AnalysisConfig.common_continuous(estimator=HansenHodrick())
# IncompatibleAxisError: estimator='HansenHodrick' not applicable to
# (scope=common, signal=continuous). Applicable HAC estimators: NeweyWest
```

To inspect what's available for a given cell:

```python
fx.list_estimators(fx.FactorScope.INDIVIDUAL, fx.Signal.CONTINUOUS)
# ['BlockBootstrap', 'HansenHodrick', 'NeweyWest', 'WaldNWCluster', 'WaldTwoWayCluster']
```

`list_estimators` returns every registered `Estimator` (selection-axis
and HAC-axis); the construction gate further narrows to `HACEstimator`
instances applicable to the cell.

## When not in scope

- **Per-factor estimator swap.** Permanently not opened — single-cfg
  decision frequency is the spec-search lock.
- **Slope-axis HAC (TS β, TS Dummy).** Single-asset OLS regressions
  with NW HAC SE on the slope run a different math shape (`(y, x)
  → β, SE(β)`) than the series-mean `compute(series, *,
  forward_periods)` contract. A slope-axis sub-protocol is tracked
  for a future release.
- **Moment-condition estimators (GMM J-test).** GMM consumes
  multi-dimensional moment conditions; the parallel
  `MomentEstimator(Estimator)` sub-protocol lands with
  [#191](https://github.com/awwesomeman/factrix/issues/191).
