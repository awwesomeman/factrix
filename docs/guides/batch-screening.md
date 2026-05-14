---
title: Batch screening with Benjamini-Hochberg-Yekutieli
---

!!! abstract "Answers"
    What is Benjamini-Yekutieli (BHY), when to use it, and how `multi_factor.bhy` partitions the candidate set across statistical families.
    For the API signature, see [`multi_factor`](../api/multi-factor.md).
    For the underlying theorem and assumptions, see [Statistical methods](../reference/statistical-methods.md).

BHY controls false discovery rate (FDR) **within a statistical family**: evaluate multiple candidate factors under the same procedure, then apply step-up correction on the resulting p-values.

!!! warning "Do not mix families"
    p-values from information coefficient (IC) / FM / TS-β carry different null distributions and cannot be pooled. `bhy()` partitions automatically — see below — but if you assemble the input list yourself across procedures, the FDR guarantee breaks.

## Basic usage

```python
import factrix as fx

candidates = ["mom_5d", "mom_20d", "mom_60d"]
cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC, forward_periods=5)

profiles = [fx.evaluate(panel, cfg, factor_col=name) for name in candidates]
survivors = fx.multi_factor.bhy(profiles, q=0.05)
survivor_names = [p.factor_id for p in survivors.profiles]
```

`evaluate()` stamps `factor_col` into `profile.factor_id`, so the survivor → name mapping reads off the survivor profiles directly — no external `name → profile` dict, no `is`-comparison idiom. `Survivors.profiles` lists the survivors in their original input order; `Survivors.adj_p` carries the bucket-local BHY-adjusted p in matching order.

Each `evaluate` call repays the per-date cross-section overhead
(sort / group-by / rank) on its own — that cost is intrinsic to
producing one `FactorProfile` per signal in factrix today. `bhy()`
operates on the resulting list for FDR control; it does not reduce
the per-signal evaluation cost.

[`bhy()`][factrix.multi_factor.bhy] automatically partitions by `(procedure, forward_periods)` — you do not pass a group key. Same-procedure, same-horizon profiles form one family. Different horizons always split: each horizon carries its own null distribution and effective sample size; pooling dilutes the step-up threshold and silently inflates FDR.

If any family degenerates to size=1 (typical misuse: one factor evaluated across multiple scenarios), `bhy()` emits a `RuntimeWarning` — at size=1 BHY equals the raw threshold and provides no FDR correction.

## Horizon-shopping correction

`bhy()` controls FDR within a horizon. If you sweep multiple horizons per factor and pick the minimum p, the horizon selection itself is hidden multiple testing (K = number of horizons). You must collapse the horizon dimension first with a family-wise error rate (FWER) procedure, then feed the result into BHY.

### Background

The multiple-testing discipline for factor research established by [Harvey-Liu-Zhu 2016][harvey-liu-zhu-2016] motivates correcting for selection once factor candidates and horizons are swept — a 5% nominal threshold no longer controls type-I error. factrix's specific composition (FWER across horizons, then FDR within) is a project-level application; HLZ themselves prescribe stricter thresholds, not this two-axis stack. The reason factrix picks FWER for the inner step is the dependence structure [Boudoukh-Richardson-Whitelaw 2008][boudoukh-richardson-whitelaw-2008] documents: under the null and a persistent regressor, ordinary least squares (OLS) slope estimators across horizons are highly correlated — approaching unity between adjacent horizons at dividend-yield-like persistence — so the K horizons behave more like one repeatedly-tested null than K independent draws. Independence- and positive regression dependence on a subset (PRDS)-friendly FDR procedures (Benjamini-Hochberg (BH) / BHY) assume neither identity and lose their level guarantees in this regime.

[Bailey & López de Prado (2014)][bailey-lopez-de-prado-2014] formalises the parallel multiple-trials problem on the Sharpe axis (Deflated Sharpe Ratio) for backtest selection — same correction path, different statistic; not implemented in factrix.

**Recommended FWER procedures:**

| Procedure | When to use |
|---|---|
| Bonferroni (`p × K`) | K small (≤ 5), horizons approximately independent |
| Holm (step-down) | K larger, or p-values vary widely in strength |

!!! warning "Do not use BHY as the inner procedure"
    (1) Picking one representative p is a FWER problem, not FDR. (2) BHY ∘ BHY has no composition theorem. (3) For small K, BHY's `c(m)` factor makes it more conservative than Bonferroni anyway.

!!! warning "Do not flatten K × H profiles into one `bhy()` call"
    Flattening K factors × H horizons into K×H profiles and feeding them to `bhy()` directly is wrong. BHY partitions by horizon into H families of K — correct for "pick factors within each horizon" — but wrong for "pick best horizon per factor."
