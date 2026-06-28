---
title: Timeseries-mode conventions
---

!!! tip "Canonical reference"
    For the `DataStructure.PANEL` vs `DataStructure.TIMESERIES` dispatch concept and sample-guard contract, see [Panel vs timeseries](../guides/panel-timeseries.md). For the statistical disciplines (heteroskedasticity-and-autocorrelation-consistent (HAC) SE, augmented Dickey-Fuller (ADF) / Stambaugh, non-overlap default) that the rules below build on, see [Statistical methods](statistical-methods.md). This page documents the per-asset (stage-1) time-series conventions of the `Common × Continuous` metrics.

`Common × Continuous` metrics (`ts_beta`, `ts_quantile`, `ts_asymmetry`
and their variants) are **PANEL** metrics: they need `n_assets >= 2` and
raise `IncompatibleAxisError` at `n_assets == 1` (there is no single-asset
mode). This page documents the conventions that govern their **stage-1
per-asset time-series regressions** — run inside `compute_ts_betas` —
which are not visible from the per-metric API page. Each metric page
links here so the rationale is reachable without source-diving.

## Plain SE in stage-1 per-asset ordinary least squares (OLS)

`ts_beta` is a two-stage estimator: stage 1 is a per-asset OLS of
`forward_return ~ factor`, stage 2 is the cross-asset distribution of
the resulting β. **Stage 1 deliberately retains plain OLS SE rather
than Newey-West (NW) / HAC** even when `forward_periods > 1`
introduces overlap.

The rationale lives in
[Statistical methods § stage-1 plain SE](statistical-methods.md#stage1-plain-se):
the dominant bias under a persistent predictor is Stambaugh
coefficient bias, which HAC does not address. Stage 2 cross-asset
inference handles whatever residual time-axis structure leaks through
the β distribution.

Operational tip: if overlap-induced SE inflation is the binding concern on a
**cross-sectionally varying** factor, use the `Individual × Continuous` IC
pipeline (`ic(inference=fx.inference.NEWEY_WEST)`), where HAC adjustment is the
canonical inferential primitive. A broadcast `Common × Continuous` factor cannot
be rescued by IC: it has no per-date cross-sectional rank dispersion, so it
belongs in the `ts_beta` family and should be interpreted with the stage-1
plain-SE caveat here.

## No persistence diagnostic on this family

The `ts_beta` family emits **no** unit-root / ADF persistence
diagnostic — the only ADF diagnostic in factrix is on `ic_trend`
(`metadata["adf_p"]` / `unit_root_suspected`, see
[Statistical methods § Persistence diagnostics](statistical-methods.md#4-persistence-diagnostics-under-near-unit-root-predictors)).
The persistence *caveat* below still matters for interpreting the
stage-1 slopes, but it is not surfaced automatically on these metrics.

## `forward_periods` vs `signal_horizon`: bias under mismatch
[](){ #non-overlap-convention }

The mainstream `Individual × Continuous` metrics (`ic`, `caar`)
use non-overlapping resampling as the inferential default
([Statistical methods § non-overlap default](statistical-methods.md#non-overlap-default)).
The `ts_beta` family inverts this: the per-asset stage-1 regression
runs on the **full** overlapping series — a single asset's series
lacks a cross-section axis to "burn" `h` periods of samples, so
non-overlapping resampling at stage 1 would leave inadequate `T` for
the per-asset OLS at typical horizons.

When the dataset's `signal_horizon` differs from the
`forward_periods` passed to `evaluate()`
([`datasets.md`](../api/datasets.md) frames the *decay* side of this),
the realised stage-1 signal is also **biased**, not only
decayed. Two distinct sources compound:

- **Overlap structure**: `h ≠ signal_horizon` produces an MA(h−1)
  residual whose autocovariance is no longer the simple Bartlett
  approximation HAC assumes. The Hansen-Hodrick floor in NW absorbs
  this for HAC-using metrics; the plain-SE stage-1 in `ts_beta` does
  not.
- **Stambaugh-style coefficient bias**: when the predictor is
  persistent (a near-unit-root regressor),
  `forward_periods ≠ signal_horizon` shifts the OLS coefficient
  itself, not only its SE.

Treat `forward_periods == signal_horizon` as the regime where
TIMESERIES-mode inference is calibrated; other horizons are
exploratory and the reported *p*-values should be discounted
accordingly. This is a **bias** caveat distinct from the IC-decay
framing on the synthetic datasets page.
