# TIMESERIES-mode conventions

`Common × Continuous` evaluations on a single time series (`ts_beta`,
`ts_quantile`, `ts_asymmetry` and their variants) inherit four shared
conventions that are not visible from the per-metric API page. Each
metric page links here so the rationale is reachable without
source-diving.

## Plain SE in stage-1 per-asset OLS

`ts_beta` is a two-stage estimator: stage 1 is a per-asset OLS of
`forward_return ~ factor`, stage 2 is the cross-asset distribution of
the resulting β. **Stage 1 deliberately retains plain OLS SE rather
than NW / HAC** in TIMESERIES mode even when `forward_periods > 1`
introduces overlap.

Reasoning summarised in
[Statistical methods § HAC SE](statistical-methods.md#1-hac-se-under-overlapping-returns)
(last paragraph): the dominant bias under a persistent predictor is
[Stambaugh 1999][stambaugh-1999] coefficient bias, which HAC does not
address. Stage 2 cross-asset inference handles whatever residual
time-axis structure leaks through the β distribution.

If overlap-induced SE inflation is the binding concern, prefer
`ic_newey_west` on the same series (Individual × Continuous cell)
where the HAC adjustment is the canonical inferential primitive.

[stambaugh-1999]: bibliography.md#stambaugh-1999

## `FACTOR_ADF_P` persistence diagnostic

Every TIMESERIES-mode procedure emits `FACTOR_ADF_P` on the input
factor series (see `_procedures.py`). A failed unit-root rejection
(`FACTOR_ADF_P > 0.05`) does not short-circuit the metric — the slope
is still returned — but the diagnostic surfaces on the profile and
slope significance should be read with that caveat.

Full derivation, threshold rationale, and interpolation accuracy live
in
[Statistical methods § Persistence diagnostics](statistical-methods.md#4-persistence-diagnostics-under-near-unit-root-predictors).
The
[`FactorProfile` StatCode → method table](../api/factor-profile.md#statcode--statistical-method)
maps `FACTOR_ADF_P` to that section directly. The
`unit_root_suspected=True` flag is `ic_trend` metadata only — the
TIMESERIES-mode cells expose the raw `FACTOR_ADF_P` *p*-value and
leave the threshold call to the caller.

The diagnostic is on the *input* factor, not the regression residual.

## Non-overlap convention — `forward_periods` vs `signal_horizon`

The cell-canonical `Individual × Continuous` metrics (`ic`, `caar`)
use **non-overlapping resampling** as the inferential default
([Statistical methods § HAC SE](statistical-methods.md#1-hac-se-under-overlapping-returns)).
TIMESERIES mode inverts this: the per-asset stage-1 regression runs
on the **full** overlapping series — TIMESERIES lacks the
cross-section axis to "burn" `h` periods of samples, so
non-overlapping resampling at stage 1 would leave inadequate `T` for
the per-asset OLS at typical horizons.

When the dataset's `signal_horizon` differs from
`AnalysisConfig.forward_periods`
([`datasets.md`](../api/datasets.md) frames the *decay* side of this),
the realised TIMESERIES-mode signal is also **biased**, not only
decayed. Two distinct sources compound:

- **Overlap structure**: `h ≠ signal_horizon` produces an MA(h−1)
  residual whose autocovariance is no longer the simple Bartlett
  approximation HAC assumes. The Hansen-Hodrick floor in NW absorbs
  this for HAC-using metrics; the plain-SE stage-1 in `ts_beta` does
  not.
- **Stambaugh-style coefficient bias**: when the predictor is
  persistent (the typical regime flagged by `FACTOR_ADF_P`),
  `forward_periods ≠ signal_horizon` shifts the OLS coefficient
  itself, not only its SE.

Treat `forward_periods == signal_horizon` as the regime where
TIMESERIES-mode inference is calibrated; other horizons are
exploratory and the reported *p*-values should be discounted
accordingly. This is a **bias** caveat distinct from the IC-decay
framing on the synthetic datasets page.

## Single-series null

TIMESERIES dispatch routes `Common × Continuous` to a single-asset β
whose null is `H₀: β = 0` for the one series — **not** the PANEL null
`H₀: E[β] = 0` over the cross-section. The two tests answer different
questions and their *p*-values are not comparable.
`describe_analysis_modes(format="text")` annotates the routing
distinction inline ("`single-series test (null differs from PANEL)`")
when listing the TIMESERIES side of a `Common × Continuous` cell, so
callers comparing PANEL and TIMESERIES outputs do not assume parity.
