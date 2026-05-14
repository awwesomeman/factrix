---
title: Glossary
---

factrix names its dispatch axes after the structure of the data
(`Scope × Signal`) and the regression layout (`Mode`). The literature
this borrows from uses overlapping but **not identical** vocabulary —
this glossary maps the factrix terms to their nearest industry
equivalents and flags collisions where the same word means different
things in factrix vs Alphalens / Barra / standard panel-econometrics.

## Scope axis — `INDIVIDUAL` vs `COMMON`

### `INDIVIDUAL`

Per-asset factor values: each `(date, asset_id)` pair has its own
factor reading. The cross-section is the unit of inference (per-date
IC, FM λ, CAAR per event date). This is the regime most equity factor
research operates in.

**Industry equivalents**:

- **Alphalens**: "characteristic" / "factor". Alphalens panels are
  always per-asset signals, so the term is implicit.
- **Standard cross-sectional asset pricing**: "stock characteristic"
  (size, book-to-market, momentum). Same regime; factrix routes these
  through `Individual × Continuous`.
- **MSCI / Axioma**: "factor exposure" of asset `i` to factor `k`.

### `COMMON`

A single broadcast series shared across **all** assets on a given
date — the same value for every `asset_id` at that `date`. Macro
factors (VIX, USD index, term spread, monetary-policy shocks) and
broadcast event dummies (FOMC announcement days) live here.

**Industry equivalents and collisions**:

- **Macro / global factor literature**: "common factor" = a single
  series that drives multiple assets. **Matches** factrix `COMMON`.
- **Barra "common factor"**: a shared risk factor *estimated* from
  cross-sectional regressions on a panel of asset characteristics
  (Barra style / size / value / momentum factors). Estimation-driven,
  per-asset latent loadings. **Collides with** factrix `COMMON`,
  which is a *given* broadcast series with no estimation step on the
  factor itself.
- **Alphalens**: no direct equivalent. Closest is "group-neutral" /
  benchmark series, but Alphalens does not natively dispatch to a
  `COMMON` cell.

When porting Barra terminology in: a Barra "common factor" with raw
cross-sectional loadings is a factrix `INDIVIDUAL × CONTINUOUS`
problem (the loadings *are* the per-asset characteristic). A Barra
"common factor" treated as a single series shared across assets is a
factrix `COMMON × CONTINUOUS` problem.

## Signal axis — `CONTINUOUS` vs `SPARSE`

### `CONTINUOUS`

Factor takes a real-valued reading on every observation in the panel.
Most equity characteristics; macro factors; ML predictor scores.

### `SPARSE`

Factor is non-zero only on event dates and zero elsewhere; the panel
encodes a discrete event arrival process. Canonical encoding is
`{−1, 0, +1}` (signed event direction); factrix also accepts
magnitude-weighted `{0, R}` for continuous-magnitude events
(see `compute_caar`'s input-form table for the resulting estimator
distinction).

**Industry equivalents**:

- **MacKinlay (1997) event-study vocabulary**: factrix's `SPARSE`
  cell is the event-study cell; the factor column is the
  announcement indicator. factrix's `factor != 0` filter is the
  event-window selector.
- **Alphalens**: no direct equivalent — Alphalens assumes continuous
  characteristics. Event-study workflows live outside Alphalens.

## Mode axis — `PANEL` vs `TIMESERIES`

Mode is **derived from data** at evaluate-time, not configured:

- `N ≥ 2` → `PANEL`.
- `N == 1` → `TIMESERIES`.

### `PANEL`

Multi-asset panel; cross-section participates in inference (per-date
aggregation, cross-asset SE).

**Industry equivalents**: panel data; longitudinal data; pooled
cross-section / time-series. The econometrics literature usually says
"panel" with no further qualification.

### `TIMESERIES`

Single-asset series. Time axis is the only sample axis; cross-section
is degenerate.

**Industry equivalents**: time-series regression; single-asset OLS.
Note that factrix `TIMESERIES` mode does **not** mean Alphalens'
"time-series of cross-sectional means" (that is per-date aggregation
of a `PANEL`); the two collide on the word *time-series* and disagree
on what is aggregated. See
[Timeseries-mode conventions](ts-mode-conventions.md) for the
specific contracts that apply when factrix routes here.

## Other terms

### the metric `evaluate()` runs

Every `(scope, signal, metric, mode)` dispatch cell maps to exactly one
metric callable that [`evaluate()`](../api/evaluate.md) runs internally
— `ic`, `fama_macbeth`, `caar`, or `ts_beta` — and that metric's
p-value becomes [`FactorProfile.primary_p`](../api/factor-profile.md).
Defined by the
`factrix._registry._DISPATCH_REGISTRY`. Other metrics applicable to
the same cell are called separately via
[`list_metrics`](../api/list-metrics.md) and the per-metric callable;
they do not enter the `FactorProfile`.

The cell-keyed reverse-index lives at
[Metric applicability § Cell to evaluate-metric](metric-applicability.md#cell-to-evaluate-metric).

**Legacy synonyms** (pre-#144 docs and source comments): `procedure-canonical`,
`cell-canonical`, `primary metric`, `canonical metric`. New content uses the
behavioral phrasing above; legacy occurrences are phased out via the ratchet
test in `tests/test_docs_terminology_ratchet.py` so the count can only decrease.

### `factor`

The signal column. Same role as Alphalens "factor", Barra "exposure",
or generic "alpha score" / "predictor". The procedure-level
`INPUT_SCHEMA` consumes a column literally named `factor`; pass
`evaluate(..., factor_col="alpha")` to use a differently-named
column without renaming the panel first
([Panel schema § `factor_col=`](../api/panel-schema.md#factor_col--non-default-signal-column-name)).

### `forward_return`

The forward return column consumed by every dispatch cell. factrix's
`compute_forward_return(df, forward_periods=N)` produces a
**per-period** return with **`t+1` entry**:

```
forward_return[t] = (price[t+1+N] / price[t+1] − 1) / N
```

Two non-textbook choices to internalize:

- **`t+1` entry, not `t`**: signal at `t` is computed from data up to
  and including `price[t]`, so trading at `price[t]` would assume
  same-bar execution. Entry at `t+1` enforces a causal boundary and
  cleanly separates the return window from the BMP estimation window.
- **Divided by `N`**: result is expressed per period rather than as
  the cumulative `N`-period return, so factor evaluations at
  different `forward_periods` are directly comparable. If you need
  the cumulative `N`-period number, multiply by `N`.

Only simple returns are implemented; no log-return option. factrix
takes `forward_return` as input rather than computing it inside
`evaluate()` — attach it with
[`compute_forward_return`](../api/preprocess.md) before dispatch so
the horizon `N` is explicit and aligned with
`AnalysisConfig.forward_periods`.

Distinct from "spot return" (contemporaneous one-period return) and
"realised return" (ex-post return for risk attribution).

### `signal_horizon` (datasets only)

A property of the *synthetic* signal embedded in `make_cs_panel` /
`make_event_panel` — the horizon at which the generator's nominal IC
is realised. Aligning `AnalysisConfig.forward_periods` with
`signal_horizon` is the regime where the synthetic IC / drift is
calibrated; mismatched horizons induce IC decay (synthetic) and bias
(in TIMESERIES mode — see
[Timeseries-mode conventions § Non-overlap](ts-mode-conventions.md#non-overlap-convention--forward_periods-vs-signal_horizon)).

### `non-overlapping` returns

Sampling every `h`-th date so consecutive observations are
independent under the `h`-period forecasting null. factrix's
`Individual × Continuous` cell defaults to this for `ic` and `caar`;
NW HAC is the alternative on the full overlapping series. See
[Statistical methods § HAC SE](statistical-methods.md#1-hac-se-under-overlapping-returns).

### `estimation_window`

Per-asset pre-event sample used to fit the abnormal-return baseline
for `bmp_test` and `corrado_rank_test`. See
[Metric applicability § estimation_window](metric-applicability.md#estimation_window).

## Multiple testing

### `FDR` — False Discovery Rate

The expected proportion of false positives among rejected nulls, in
contrast to FWER (Family-Wise Error Rate, the probability of *any*
false positive). For a screening rule that rejects `R` of `m`
hypotheses and produces `V` false positives, FDR ≡ E[V / max(R, 1)].
factrix's primary screening primitive is BHY (Benjamini-Yekutieli
2001), which controls FDR at a user-chosen `q` under arbitrary
dependence. See [Statistical methods](statistical-methods.md) and
[`bhy`](../api/multi-factor.md).

### `BHY` — Benjamini-Yekutieli (2001)

Step-up FDR-controlling procedure that allows arbitrary dependence
between p-values (paying a `log(m)` factor versus the
independent-or-PRDS Benjamini-Hochberg 1995 procedure). factrix's
default multiplicity-correction method because financial p-values are
typically not independent.

### `family`

The set of hypotheses jointly entered into a single FDR / FWER
controlling procedure. Once a hypothesis joins a family, the
procedure's FDR claim only holds *within* that family. Family choice
is a contract decision: re-running BHY on a filtered subset of
survivors does not preserve FDR ≤ q. See
[Cross-function reference § `expand_over`](../api/decision-tree.md#expand_over-is-not-one-concept)
for the sample-restriction vs hypothesis-dimension split.

### `Survivors`

Result type from the screening functions (`bhy`, `partial_conjunction`,
`bhy_hierarchical`). Carries the post-correction adjusted q-value
(`adj_q`) per identity plus a boolean `survivor[i]` ↔ `adj_q[i] ≤ q`
duality so downstream functions (`compare(survivors)`) can render
leaderboards without re-applying the threshold.
