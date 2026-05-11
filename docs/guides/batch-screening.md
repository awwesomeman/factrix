# Batch Screening with BHY

BHY controls FDR **within a statistical family**: evaluate multiple candidate factors under the same procedure, then apply step-up correction on the resulting p-values.

!!! warning "Do not mix families"
    p-values from IC / FM / TS-β carry different null distributions and cannot be pooled. `bhy()` partitions automatically — see below — but if you assemble the input list yourself across procedures, the FDR guarantee breaks.

## Basic usage

```python
import factrix as fx

candidates = ["mom_5d", "mom_20d", "mom_60d"]
cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC, forward_periods=5)

profiles = [fx.evaluate(panel, cfg, factor_col=name) for name in candidates]
survivors = fx.multi_factor.bhy(profiles, q=0.05)
survivor_names = [p.factor_id for p in survivors.profiles]
```

`evaluate()` stamps `factor_col` into `profile.factor_id`, so the survivor → name mapping reads off the survivor profiles directly — no external `name → profile` dict, no `is`-comparison idiom. `Survivors.profiles` lists the survivors in their original input order; `Survivors.adj_q` carries the bucket-local BHY-adjusted p in matching order.

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

!!! warning "Do not use BHY as the inner procedure"
    (1) Picking one representative p is a FWER problem, not FDR. (2) BHY ∘ BHY has no composition theorem. (3) For small K, BHY's `c(m)` factor makes it more conservative than Bonferroni anyway.

!!! warning "Do not flatten K × H profiles into one `bhy()` call"
    Flattening K factors × H horizons into K×H profiles and feeding them to `bhy()` directly is wrong. BHY partitions by horizon into H families of K — correct for "pick factors within each horizon" — but wrong for "pick best horizon per factor."
