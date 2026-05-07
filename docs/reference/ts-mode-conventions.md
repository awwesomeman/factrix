# TS-regression conventions

The `Common × Continuous` cell (and its `ts_beta` / `ts_quantile` /
`ts_asymmetry` family of metrics) is built on a per-asset
**time-series regression**: `forward_return ~ factor` fitted by OLS on
each asset independently. PANEL aggregates the per-asset slopes into
a cross-asset *t* on `E[β]`; TIMESERIES (`N == 1`) is the degenerate
case of the same regression — one series, no aggregation step.

This page collects four conventions that govern the family. *"TS"
here means time-series regression, the estimator pattern, not
`Mode.TIMESERIES` dispatch* — most conventions apply to both PANEL
and TIMESERIES dispatch and the differences are flagged per section.
Each metric page links here so the rationale is reachable without
source-diving.

## Plain SE in PANEL stage-1 per-asset OLS

PANEL `Common × Continuous` is a two-stage estimator
([`_compute_common_panel`](../development/architecture.md)): stage 1
fits per-asset OLS via `compute_ts_betas`; stage 2 takes the
cross-asset *t* on the resulting β distribution. **Stage 1
deliberately retains plain OLS SE rather than NW / HAC** even when
`forward_periods > 1` introduces overlap.

Reasoning summarised in
[Statistical methods § HAC SE](statistical-methods.md#1-hac-se-under-overlapping-returns)
(last paragraph): the dominant bias under a persistent predictor is
[Stambaugh 1999][stambaugh-1999] coefficient bias, which HAC does not
address. Stage 2 cross-asset inference handles whatever residual
time-axis structure leaks through the β distribution.

The choice is best understood against the three textbook
"corrected SE" alternatives, each of which fixes a different
residual-level problem but **none of which touches the
coefficient**:

| SE variant | Fixes (residual-level) | Coefficient bias under persistent predictor |
|---|---|---|
| HC0 / HC1 ([White 1980][white-1980]) | Residual heteroskedasticity (variance varies across observations) | Untouched |
| HAC / Newey-West ([Newey-West 1987][newey-west-1987]) | Heteroskedasticity + autocorrelation (incl. MA(`h−1`) overlap) | Untouched |
| Cluster-robust ([Petersen 2009][petersen-2009] / [Cameron-Gelbach-Miller 2011][cameron-gelbach-miller-2011]) | Within-group residual correlation (e.g. firm or date clusters) | Untouched |
| **plain OLS SE** (factrix stage-1 choice) | — (no robustness claim) | Untouched (and acknowledged) |

Adopting any of the robust variants in stage-1 would advertise a
robustness factrix does not deliver: the SE would *look* fixed, but
β̂ would still carry Stambaugh bias. Plain SE is therefore an
honesty signal — the reported uncertainty does not claim to fix what
the SE level cannot fix anyway. Cluster-robust SE on a per-asset
stage-1 is additionally moot (each regression already conditions on a
single asset; there is no within-group axis left to cluster on).

[white-1980]: bibliography.md#white-1980
[newey-west-1987]: bibliography.md#newey-west-1987
[petersen-2009]: bibliography.md#petersen-2009
[cameron-gelbach-miller-2011]: bibliography.md#cameron-gelbach-miller-2011

**TIMESERIES dispatch (`N == 1`) is different.** With no
cross-section to aggregate over, the single-series path
(`_TSBetaContTimeseriesProcedure`) **does** apply NW HAC directly via
`_ols_nw_slope_t` — the Hansen-Hodrick floor on the bandwidth absorbs
the MA(h−1) overlap structure on the one available series. The
plain-SE rationale above is therefore PANEL-stage-1 specific and does
not extend to the TIMESERIES single-series test.

If overlap-induced SE inflation is the binding concern in PANEL
mode, prefer `ic_newey_west` on the same series (Individual ×
Continuous cell) where HAC is the canonical inferential primitive.

[stambaugh-1999]: bibliography.md#stambaugh-1999

## `FACTOR_ADF_P` persistence diagnostic

Every `Common × Continuous` procedure (PANEL and TIMESERIES alike)
emits `FACTOR_ADF_P` on the input factor series. A failed unit-root
rejection (`FACTOR_ADF_P > 0.05`) does not short-circuit the metric
— the slope is still returned — but the diagnostic surfaces on the
profile and slope significance should be read with that caveat.

Full derivation, threshold rationale, and interpolation accuracy live
in
[Statistical methods § Persistence diagnostics](statistical-methods.md#4-persistence-diagnostics-under-near-unit-root-predictors).
The
[`FactorProfile` StatCode → method table](../api/factor-profile.md#statcode--statistical-method)
maps `FACTOR_ADF_P` to that section directly. The
`unit_root_suspected=True` flag is `ic_trend` metadata only — the
`Common × Continuous` cells expose the raw `FACTOR_ADF_P` *p*-value
and leave the threshold call to the caller.

The diagnostic is on the *input* factor, not the regression residual.

## Non-overlap convention — `forward_periods` vs `signal_horizon`

The cell-canonical `Individual × Continuous` metrics (`ic`, `caar`)
use **non-overlapping resampling** as the inferential default
([Statistical methods § HAC SE](statistical-methods.md#1-hac-se-under-overlapping-returns)).
The TS-regression family inverts this in **both dispatch modes**: the
per-asset (or single-asset) regression runs on the **full**
overlapping series. The asset axis cannot be used to "burn" `h`
periods of samples in either case — at PANEL each asset's stage-1 has
its own short series; at TIMESERIES there is only the one series — so
non-overlapping resampling would leave inadequate `T` for OLS at
typical horizons.

The two modes mitigate the resulting overlap differently: PANEL
absorbs it through stage-2 cross-asset aggregation (see plain-SE
section above), TIMESERIES absorbs it through the NW HAC bandwidth
on the single series.

When the dataset's `signal_horizon` differs from
`AnalysisConfig.forward_periods`
([`datasets.md`](../api/datasets.md) frames the *decay* side of this),
the realised signal is also **biased**, not only decayed. Two
distinct sources compound:

- **Overlap structure**: `h ≠ signal_horizon` produces an MA(h−1)
  residual whose autocovariance is no longer the simple Bartlett
  approximation HAC assumes. The Hansen-Hodrick floor in NW absorbs
  this for HAC-using metrics (TIMESERIES single-series; `ic_newey_west`);
  the plain-SE PANEL stage-1 does not.
- **Stambaugh-style coefficient bias**: when the predictor is
  persistent (the typical regime flagged by `FACTOR_ADF_P`),
  `forward_periods ≠ signal_horizon` shifts the OLS coefficient
  itself, not only its SE — present in both PANEL and TIMESERIES.

Treat `forward_periods == signal_horizon` as the regime where
TS-regression inference is calibrated; other horizons are
exploratory and the reported *p*-values should be discounted
accordingly. This is a **bias** caveat distinct from the IC-decay
framing on the synthetic datasets page.

## Single-series null (TIMESERIES dispatch only)

TIMESERIES dispatch routes `Common × Continuous` to a single-asset β
whose null is `H₀: β = 0` for the one series — **not** the PANEL null
`H₀: E[β] = 0` over the cross-section. The two tests answer different
questions and their *p*-values are not comparable.
`describe_analysis_modes(format="text")` annotates the routing
distinction inline ("`single-series test (null differs from PANEL)`")
when listing the TIMESERIES side of a `Common × Continuous` cell, so
callers comparing PANEL and TIMESERIES outputs do not assume parity.
