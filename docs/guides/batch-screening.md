# Batch Screening with BHY

BHY controls FDR **within a statistical family**: evaluate multiple candidate factors under the same procedure, then apply step-up correction on the resulting p-values. **Do not mix families** — p-values from IC / FM / TS-β carry different null distributions and cannot be pooled.

## Basic usage

```python
import factrix as fl

candidates = ["mom_5d", "mom_20d", "mom_60d"]
cfg = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)

profiles = [fl.evaluate(panel, cfg, factor_col=name) for name in candidates]
survivors = fl.multi_factor.bhy(profiles, threshold=0.05)
```

Each `evaluate` call repays the per-date cross-section overhead
(sort / group-by / rank) on its own — that cost is intrinsic to
producing one `FactorProfile` per signal in factrix today. `bhy()`
operates on the resulting list for FDR control; it does not reduce
the per-signal evaluation cost.

[`bhy()`][factrix.multi_factor.bhy] automatically partitions by `(procedure, forward_periods)` — you do not pass a group key. Same-procedure, same-horizon profiles form one family. Different horizons always split: each horizon carries its own null distribution and effective sample size; pooling dilutes the step-up threshold and silently inflates FDR.

If any family degenerates to size=1 (typical misuse: one factor evaluated across multiple scenarios), `bhy()` emits a `RuntimeWarning` — at size=1 BHY equals the raw threshold and provides no FDR correction.

## Horizon-shopping correction

`bhy()` controls FDR within a horizon. If you sweep multiple horizons per factor and pick the minimum p, the horizon selection itself is hidden multiple testing (K = number of horizons). You must collapse the horizon dimension first with a FWER procedure, then feed the result into BHY.

**Recommended FWER procedures:**

| Procedure | When to use |
|---|---|
| Bonferroni (`p × K`) | K small (≤ 5), horizons approximately independent |
| Holm (step-down) | K larger, or p-values vary widely in strength |

Do **not** use BHY as the inner procedure: (1) picking one representative p is a FWER problem, not FDR; (2) BHY ∘ BHY has no composition theorem; (3) for small K, BHY's `c(m)` factor makes it more conservative than Bonferroni.

> Do not flatten K factors × H horizons into K×H profiles and call `bhy()` directly. BHY partitions by horizon into H families of K — correct for "pick factors within each horizon" — but wrong for "pick best horizon per factor."
