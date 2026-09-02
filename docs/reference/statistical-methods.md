---
title: Statistical methods
---

!!! note "Audience"
    This page assumes familiarity with applied econometrics. For a first
    result, start with [Quickstart](../getting-started/quickstart.md). For
    empirical size measurements and known finite-sample limits, use
    [Inference calibration and limitations](inference-calibration.md).

This page defines the cross-cutting statistical contracts shared by factrix
metrics. Per-metric formulas, parameters, and return schemas remain on the
[Metrics API pages](../api/metrics/index.md).

!!! note "Periods, not calendar time"
    Every horizon, lag, stride, and window counts periods on the panel's own
    distinct-date grid. No method infers frequency or annualises from a date.
    See [Period grid, not calendar](../development/architecture.md#period-grid-not-calendar).

## 1. HAC SE under overlapping returns
[](){ #nw-hac }

An `h`-period forward return makes adjacent observations share up to `h - 1`
periods. factrix exposes three series-mean approaches. None removes economic
persistence that remains after the overlap horizon.

| Method | Dependence treatment | Reference and main trade-off |
|---|---|---|
| `NON_OVERLAPPING` | Keep observations at least `overlap_periods` apart | Student-t reference on the strided series. Mechanical overlap is removed; the reference is exact only under iid normality. It spends roughly a factor of `h` in sample size. |
| `NEWEY_WEST` | Bartlett-kernel heteroskedasticity-and-autocorrelation-consistent (HAC) variance on the full series | Retains observations but uses an asymptotic HAC reference and a calibrated deterministic bandwidth. |
| `STATIONARY_BOOTSTRAP` | Stationary block resampling with an automatically selected mean block length | Avoids a parametric-normal reference for the series mean, but still requires stationarity or weak dependence, a suitable block length, and an adequate sample. It is a second read for long series with distributional doubt, not a rescue for a short series. |

`NON_OVERLAPPING` is the default for the mainstream series-mean metrics. The
caller can select the other admitted members explicitly; factrix never switches
the requested method in response to a warning.

These metrics also accept `alternative="two-sided"`, `"greater"`, or `"less"`.
Keep the default while discovering a factor. Use a one-sided tail only when its
direction was fixed independently and before inspecting the evaluated sample;
factrix never infers a tail from the observed sign. Holm and BHY consume the
resulting calibrated p-values as supplied and do not reinterpret their tails.
[](){ #non-overlap-default }

### Newey-West bandwidth families

The bandwidth depends on the rank of the tested restriction, not simply on
whether a regression has multiple coefficients.

For a scalar series mean or rank-one regression contrast, factrix uses

$$
L = \min\!\left(\max\!\left(1.3\sqrt{T},\;3(h-1)\right),\;\lceil T/3\rceil\right),
$$

scales the variance by $T/(T-L-1)$, and reads the statistic against

$$
\nu = \max\!\left(\min\!\left(1.5T/L-1,\;T/h-1\right),\;1\right).
$$

The $1.3\sqrt{T}$ term follows [LLSW (2018)][llsw-2018]; the overlap floor
extends the [Hansen-Hodrick (1980)][hansen-hodrick-1980] coverage requirement.
The wider `3(h - 1)` floor, finite-sample scale, and effective degrees of
freedom are factrix calibration choices rather than textbook Newey-West.

Multi-restriction Wald tests instead use
$L=\max(\text{auto_bartlett}(T), h-1)$ and an F reference. A wide kernel can
degrade the inversion of a multi-restriction HAC matrix, so the scalar recipe
does not transfer to this family.

### HAC path map
[](){ #hac-families }

| Path | Bandwidth and reference |
|---|---|
| `ic`, `quantile_spread`, `quantile_spread_vw`, `k_spread` with `NEWEY_WEST`; `fm_beta` stage 2 | Scalar series-mean recipe and $t_\nu$ |
| `spanning_alpha`, `common_quantile_spread`, `common_asymmetry`, single-restriction OLS contrasts | Scalar rank-one recipe and $t_\nu$ or $F_{1,\nu}$ |
| `predictive_beta`, `h > 1` | Scalar bandwidth and effective-df reference after the Amihud-Hurvich correction |
| `pooled_beta` | Driscoll-Kraay covariance over period-level score sums, using the scalar HAR bandwidth, variance scale, and $t_\nu$ reference; it is not a series-mean covariance |
| Cross-sectional slice Wald tests | Narrow multi-restriction bandwidth and $F$ reference |
| Date-disjoint `slice_period_*_test` | Per-slice NW variances for `method="analytic"`; independent stationary bootstrap draws for `method="bootstrap"` |

See [HAC family calibration](inference-calibration.md#hac-family-calibration)
for measured size bands and known oversized regimes.

### Which metrics expose `inference=`

`ic`, `quantile_spread`, `quantile_spread_vw`, and `k_spread` expose the
closed allowlist `NON_OVERLAPPING`, `NEWEY_WEST`, and
`STATIONARY_BOOTSTRAP`. Each member was measured on the relevant IC or spread
series before admission.

Other metrics keep estimator-specific parameters instead:

| Metric family | Surface |
|---|---|
| `fm_beta`, `predictive_beta`, `common_quantile_spread`, `common_asymmetry` | `newey_west_lags`; `None` selects the documented rule, while an integer is an uncalibrated research override |
| `pooled_beta` | `driscoll_kraay_lags`; `None` selects the calibrated rule, while an integer replaces the automatic base but cannot undercut the overlap floor or estimability cap |
| `spanning_alpha` and slice Wald tests | Bandwidth determined from `overlap_periods`; no caller lag knob |

## 2. Multiple-testing under dependence
[](){ #bhy }

The literature names two step-up procedures three ways:

- **BH** is the [Benjamini-Hochberg (1995)][benjamini-hochberg-1995]
  procedure. Its FDR guarantee assumes independence or positive regression
  dependence on a subset (PRDS).
- **BY** is the [Benjamini-Yekutieli (2001)][benjamini-yekutieli-2001]
  arbitrary-dependence procedure. It divides the BH threshold by
  $c(m)=\sum_{i=1}^{m}1/i$.
- **BHY** is common quant shorthand for the BY procedure. factrix follows
  that convention in the function name `bhy()`.

Factor-test dependence is usually difficult to prove to be PRDS. factrix
therefore uses the BY/BHY threshold when a family must be valid under arbitrary
dependence. Bonferroni also remains valid under arbitrary dependence, but it
controls the family-wise error rate (FWER), a stricter and usually more
conservative target than FDR; its conservatism does not come from assuming
independence.

The declared hypothesis family is part of the analysis. Placeholder results
are excluded rather than treated as p-values, and separate economic questions
should not be pooled merely to increase `m`. See
[BHY screening](../api/bhy.md) for family construction and hierarchical
procedures.

### Resampling knobs

Entry points that turn resamples into a p-value or interval use
`n_resamples` and `rng` consistently.

For `STATIONARY_BOOTSTRAP`, result metadata keeps the requested draw count in
`n_resamples` so `seed` can reproduce the run. `n_resamples_used` reports the
finite studentized roots that actually enter the empirical p-value; invalid
roots are excluded from both its denominator and `p_value_mc_se`. If none are
usable, the p-value is unavailable rather than `1.0`.

| Entry point | Default `n_resamples` | Enforces the 199 floor | Reports resolved seed |
|---|---:|---:|---:|
| `STATIONARY_BOOTSTRAP` on admitted metrics | 999 | Yes | In metric metadata |
| `monotonicity` | 999 | Yes | In metric metadata |
| `bootstrap_mean_ci` | 999 | Yes | No; returns an interval |
| `slice_period_pairwise_test` / `slice_period_joint_test` with `method="bootstrap"` | 999 | Yes | In the result column |
| `stationary_bootstrap_resamples` | 999 | No; returns draws, not inference | No |

A smoothed empirical p-value lies on the `1/(B + 1)` grid. Values such as
199, 399, and 999 align common significance levels with that grid; 999 gives
roughly 0.001 resolution. The Monte Carlo standard error
$\sqrt{p(1-p)/B}$ describes resampling noise, not uncertainty in the factor
estimate. Automatic block-length selection is a separate operation; it does
not determine the number of resamples.

`rng` accepts an integer, `None`, or `numpy.random.Generator`. An integer is
reproducible and reported unchanged. `None` draws and reports an integer seed.
A generator is advanced in place and reports no seed because the caller owns
the stream.

## 3. Robust scale and trend handling

For each period, `mad_winsorize` clips a factor to

$$
\text{centre} \pm n_{\text{MAD}}\,b_n\,1.4826\,\operatorname{MAD},
$$

where the centre can be the median (default) or mean. The MAD itself is always
computed about the median. `1.4826` gives asymptotic Gaussian consistency;
the Croux-Rousseeuw $b_n$ factor corrects finite-cross-section bias. This
keeps small cross-sections from receiving a materially narrower band solely
because raw MAD is biased downward.

Sn and Qn have higher Gaussian efficiency than MAD, but require more pairwise
order-statistic machinery. factrix keeps the simpler MAD contract and applies
the explicit finite-sample correction instead.

`ic_trend` pairs two rank-based quantities:

- Theil-Sen supplies the trend magnitude and descriptive confidence interval.
- Hamed-Rao Mann-Kendall supplies `stat` and `p_value` after non-overlapping
  sampling.

The p-value is not backed out of the Theil-Sen interval. A constant series
returns the zero slope while withholding `stat` and `p_value` with a
degenerate-variance warning.

## 4. Persistence diagnostics under near-unit-root predictors

Persistent predictive regressors can carry finite-sample coefficient bias
([Stambaugh 1999][stambaugh-1999]). HAC changes the variance estimate; it does
not remove that coefficient bias. `predictive_beta` therefore applies the
Amihud-Hurvich correction unconditionally and reports the raw OLS slope in
metadata. The persistence warning describes the remaining inference regime;
it is not the switch that decides whether the correction runs.

At `overlap_periods > 1`, the corrected predictive-slope test remains oversized
on the measured null grid. `overlapping_predictive_inference` makes that limit
explicit on every such result. Use the estimate as an effect size, raise the
evidentiary hurdle for its p-value, and compare a one-period or genuinely
non-overlapping design. factrix does not silently substitute IVX,
Bonferroni-Q, or another model-dependent estimator.

For `ic_trend`, non-overlapping sampling happens before Theil-Sen,
Mann-Kendall, the augmented Dickey-Fuller (ADF) screen, and residual
autocorrelation diagnostics. The ADF test therefore reads the strided tested
series, not the raw MA(`h - 1`) overlap. Its lag order is selected by AIC up to
the Schwert ceiling. The default `adf_threshold=0.10` is a conventional
practitioner cutoff, not a direct prescription from Stock-Watson (1988).

A series that remains autocorrelated after the stride can still make HAC,
bootstrap, or Mann-Kendall p-values optimistic. factrix raises
`serial_correlation_detected`; changing inference member does not make that
regime calibrated. See [Persistence beyond the overlap
horizon](inference-calibration.md#persistence-beyond-the-overlap-horizon).

For stage-1 per-asset regressions inside common-factor metrics, factrix keeps
plain OLS standard errors. Adding HAC there would not correct Stambaugh bias and
would imply robustness the coefficient estimator does not provide.
[](){ #stage1-plain-se }

## 5. Event-study inference

Event metrics differ in their inference unit. Compare `n_obs` only after
checking `n_obs_axis` and the metric metadata.

| Metric | Inference unit | Test and dependence treatment | `n_obs` |
|---|---|---|---:|
| `caar` | Event periods | Mean signed abnormal return on a non-overlapping event-period series; Student-t reference | Sampled event periods |
| `bmp_z` | Valid event rows | Standardised abnormal returns; per-asset event overlap removal plus optional within-period design-effect deflation | Valid events |
| `corrado_rank` | Event periods | Signed within-asset ranks collapsed by event period; z reference on the event-period series | Event periods |
| `event_hit_rate` / `event_ic` | Event rows with period clustering adjustment | Metric-specific test with the same-period design effect where applicable | Metric-specific event count; inspect `n_event_periods` separately |

### CAAR event-period t
[](){ #caar-cross-event-t }

`caar` forms a per-event-period cross-sectional mean of signed abnormal
returns, samples event periods at the declared overlap stride, and tests the
sampled mean. It is not a raw-event-count t-test. Event-induced variance can
invalidate the simple reference, which motivates reading `bmp_z` alongside it.

### BMP-style standardised abnormal returns
[](){ #bmp-standardised-ar }

`bmp_z` standardises each valid event by its estimation-window volatility.
The default is a documented BMP-style simplification; set
`include_prediction_error_variance=True` for the constant textbook
prediction-error factor. Same-period dependence is estimated with an ANOVA
ICC(1) and a design-effect deflator. The result reports both valid events and
distinct event periods because the latter controls the residual small-sample
limit under clustering.

The published Kolari-Pynnönen statistic also includes a `(1 - r)` numerator
for variance estimated within one event date. factrix pools standardised
abnormal returns across event periods, so that between-period variance is
already present; applying the numerator again would deflate the statistic
twice.

### Corrado rank
[](){ #corrado-rank }

`corrado_rank` replaces abnormal returns with signed within-asset ranks and
collapses them by event period before inference. It is the distribution-robust
read when standardised abnormal returns are sensitive to skew or extreme
returns.

### Event skewness is descriptive
[](){ #event-skewness-no-calibrated-test }

`event_skewness` reports Fisher skewness without `stat` or `p_value`. A pooled
D'Agostino test was removed because non-normal signed returns and same-period
shocks break it in different directions; a single clustering deflator did not
calibrate both. Use `caar`, `bmp_z`, `corrado_rank`, `event_hit_rate`, or
`event_ic` for a declared directional hypothesis.

## 6. Calibration and known limitations
[](){ #6-known-simplifications-deliberately-retained }

The full size tables, known oversized regimes, and test provenance now live on
[Inference calibration and limitations](inference-calibration.md). The split
keeps method contracts scannable without hiding the evidence behind them.

Useful direct links:

- [HAC family calibration](inference-calibration.md#hac-family-calibration)

[](){ #single-restriction-wald }

- [Single-restriction Wald contrasts](inference-calibration.md#single-restriction-wald)

[](){ #joint-period-test-on-short-slices-known-over-rejection }

- [Joint period tests on short slices](inference-calibration.md#joint-period-test-on-short-slices-known-over-rejection)

[](){ #persistence-beyond-the-overlap-horizon-no-hac-or-bootstrap-path-is-calibrated }

- [Persistence beyond the overlap horizon](inference-calibration.md#persistence-beyond-the-overlap-horizon)

For null versus NaN handling, use the canonical
[Data schema](../api/data-schema.md) and
[Preparing data](../guides/preparing-data.md#6-missing-data) guidance.
