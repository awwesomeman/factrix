---
title: Multi-factor screening
---

Apply Benjamini-Yekutieli (BHY) false discovery rate (FDR) control to a batch of
candidate factors. Demonstrates the explicit-family contract and the
duplicate-identity defense — the behaviour that is impossible
to learn from docstrings alone.

Runnable notebook: [`examples/multi_factor_screening.ipynb`](https://github.com/awwesomeman/factrix/blob/main/examples/multi_factor_screening.ipynb).

## Factor type

This recipe uses `multi_factor.bhy(...)` over a list of `FactorProfile`
objects. Each profile is produced by `evaluate(panel, cfg,
factor_col=<name>)` for any registered cell — screening is
factor-type-agnostic.

The input list **is** the family. Each profile must carry a unique
`identity = (factor_id, forward_periods)` — the anti-shopping defense.
The recommended path: name the factor column distinctly per candidate
panel and pass it via `factor_col=`; `evaluate` auto-stamps `factor_id`
from that name. `dataclasses.replace(profile, factor_id=...)` is an
escape hatch for cases where you cannot rename the column.

Pass `expand_over=[<context key>]` to declare per-bucket independent
step-ups (Benjamini & Bogomolov 2014 selective inference). Mixing
`forward_periods` without `expand_over` emits `RuntimeWarning` —
different horizons carry different null distributions and pooling
silently inflates FDR.

Literature: [Benjamini & Yekutieli (2001)](../reference/bibliography.md).

## Use this when

- You have ≥2 candidate factors and want type-I error control under
  multiple testing.
- Candidates are evaluable under the same `AnalysisConfig` (same
  scope, signal, metric, forward horizon). Mixed cells / horizons in
  one call are caller's responsibility — see § 4 below.
- You prefer FDR control (BHY) over family-wise error (Bonferroni) —
  appropriate when discoveries can tolerate a known false-positive
  rate and power matters.

## What it tests

For an input family of size `N`, BHY step-up keeps profiles whose
ranked p-values satisfy `p_(k) ≤ q · k / (N · c(N))` where
`c(N) = Σ 1/i` (Benjamini-Yekutieli adjustment for arbitrary
dependence). Pass your nominal `q` directly — the `c(N)` correction
is baked in. Cross-family aggregation is *not* performed; that is the
user's responsibility and is deliberately out of scope.

## Output to read

1. `len(survivors)` (= `len(survivors.profiles)`) vs `len(profiles)`
   — coarse hit rate.
2. `survivors.profiles[i].primary_p` and `.factor_id` — which factor
   cleared the family-adjusted bar.
3. `survivors.adj_p[i]` — the BHY-adjusted p-value driving the
   survivor decision (`survivor[i] iff adj_p[i] <= q`). Computed
   bucket-locally when `expand_over` is set.
4. `survivors.n_tests` — per-bucket `m` fed into each step-up; audit
   trail for two-stage screening.
5. Any emitted `RuntimeWarning` — flags mixed `forward_periods`
   without `expand_over` (FDR-inflation foot-gun) or singleton
   `expand_over` buckets (BHY on n=1 is a raw cutoff).

## 1. Setup

```python
import factrix as fx
import numpy as np
import polars as pl
from factrix.preprocess import compute_forward_return
```

## 2. Build a single-family batch

Five candidate factors, all under the same
`individual_continuous(IC, forward_periods=5)` cell — a *valid* BHY
input where the step-up actually controls FDR.

We start from one ground-truth factor and add increasing independent and identically distributed (IID) noise
to produce variants with varying signal strengths. Each variant is
materialised under its own column name (`variant_0` … `variant_4`)
so `evaluate(..., factor_col=name)` auto-stamps a distinct
`factor_id` per profile — no post-hoc identity surgery needed.

```python
raw = fx.datasets.make_cs_panel(
    n_assets=100,
    n_dates=500,
    ic_target=0.08,
    seed=2024,
)
panel = compute_forward_return(raw, forward_periods=5)
cfg = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.IC,
    forward_periods=5,
)


def variant_panel(
    base: pl.DataFrame, *, name: str, scale: float, seed: int
) -> pl.DataFrame:
    """Add IID noise on top of the ground-truth factor and store under a fresh column name."""
    rng = np.random.default_rng(seed)
    noisy = base["factor"].to_numpy() + scale * rng.standard_normal(base.height)
    return base.drop("factor").with_columns(pl.Series(name, noisy))


candidates = {
    f"variant_{i}": variant_panel(
        panel, name=f"variant_{i}", scale=0.5 + 0.3 * i, seed=100 + i
    )
    for i in range(5)
}
profiles = [fx.evaluate(p, cfg, factor_col=name) for name, p in candidates.items()]
# Raw primary_p only — FDR-controlled decisions wait for BHY in §3.
# Per-factor `primary_p < 0.05` thresholding on N candidates is the
# spec-search anti-pattern factrix explicitly avoids.
for prof in profiles:
    print(f"  {prof.factor_id:12s} primary_p={prof.primary_p:.4g}")
```

## 3. Apply BHY

The input list **is** the family. `bhy` runs one Benjamini-Yekutieli
step-up over all profiles and returns a `Survivors` container —
`.profiles` in input order, `.adj_p` aligned, `.q` / `.expand_over` /
`.n_tests` for audit. Survivor membership is definitionally
`adj_p <= q`. The `c(N)` correction is applied internally; pass your
nominal `q`.

```python
survivors = fx.multi_factor.bhy(profiles, q=0.05)
print(f"BHY survivors: {len(survivors)} / {len(profiles)}")
for prof, adj in zip(survivors.profiles, survivors.adj_p, strict=True):
    print(f"  {prof.factor_id:12s} primary_p={prof.primary_p:.4g}  adj_p={adj:.4g}")
```

Illustrative output:

```text
BHY survivors: 5 / 5
  variant_0    primary_p=2.136e-39  adj_p=2.439e-38
  variant_1    primary_p=4.052e-26  adj_p=2.313e-25
  variant_2    primary_p=6.682e-22  adj_p=2.543e-21
  variant_3    primary_p=3.987e-17  adj_p=1.138e-16
  variant_4    primary_p=7.407e-14  adj_p=1.691e-13
```

In Jupyter, `survivors` additionally renders as a three-column
`identity | primary_p | adj_p` table via `Survivors._repr_html_`.

## 4. Duplicate-identity defense

`evaluate()` defaults `factor_col="factor"`, which means every
profile built via the lazy path lands at `identity = ("factor",
forward_periods)`. Pass two such profiles to `bhy()` and the
family-resolution layer raises [`UserInputError`][factrix.UserInputError]
rather than silently treating distinct candidates as one hypothesis.

The canonical fix is what § 2 already does: distinct column name per
panel + `evaluate(..., factor_col=name)`. When you cannot rename the
column (e.g. because the panel comes from an upstream loader you do
not own), `dataclasses.replace(profile, factor_id=name)` is the
escape hatch.

Below we deliberately take the lazy path to surface the error.

```python
import dataclasses

cfg_fm = fx.AnalysisConfig.individual_continuous(
    metric=fx.Metric.FM,
    forward_periods=5,
)

# Lazy path: both calls default factor_col="factor" → identity collide.
unstamped = [
    fx.evaluate(panel, cfg),     # IC, factor_id="factor"
    fx.evaluate(panel, cfg_fm),  # FM, factor_id="factor" — same identity
]
try:
    fx.multi_factor.bhy(unstamped, q=0.05)
except fx.UserInputError as exc:
    print("UserInputError raised as expected:")
    print(str(exc))

# Canonical fix: pass factor_col= on each evaluate call. Requires distinct
# column names; here we rename the same panel's "factor" column upfront.
stamped = [
    fx.evaluate(panel.rename({"factor": "ic_var"}), cfg, factor_col="ic_var"),
    fx.evaluate(panel.rename({"factor": "fm_var"}), cfg_fm, factor_col="fm_var"),
]
print(f"\ncanonical fix → identities: {[p.factor_id for p in stamped]}")

# Escape hatch: when you cannot rename the column, post-hoc stamp via
# dataclasses.replace. Identical end state, slightly more imperative.
escape = [
    dataclasses.replace(fx.evaluate(panel, cfg), factor_id="ic_var"),
    dataclasses.replace(fx.evaluate(panel, cfg_fm), factor_id="fm_var"),
]
print(f"escape hatch  → identities: {[p.factor_id for p in escape]}")
```

Illustrative output:

```text
UserInputError raised as expected:
bhy(): invalid profiles=('factor', 5)
  Expected: unique partition key across input; duplicate first seen at index 0, again at 1. Stamp distinct factor_id per profile via `evaluate(..., factor_col=<name>)` (canonical) or `dataclasses.replace(profile, factor_id=<name>)` (escape hatch when the column cannot be renamed); or pass `expand_over=[<context key>]` to declare per-bucket families
  Docs: https://awwesomeman.github.io/factrix/api/bhy#partition-key

canonical fix → identities: ['ic_var', 'fm_var']
escape hatch  → identities: ['ic_var', 'fm_var']
```

## 5. Where to go next

For the broader `n_assets` × factory behaviour matrix see [Guides §
Panel vs timeseries](../guides/panel-timeseries.md); for the BHY
contract specifically see [Guides § Batch
screening](../guides/batch-screening.md).
