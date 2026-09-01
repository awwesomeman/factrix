---
title: Inference calibration and limitations
---

This page records empirical null-size measurements, deliberate departures from
textbook estimators, and regimes where factrix p-values are not calibrated. It
is evidence for the contracts in [Statistical methods](statistical-methods.md),
not a promise that one simulation design covers every user dataset.

Sizes below are rejection rates at a nominal 5% unless stated otherwise.
Characterisation tests rerun selected cells at lower replication counts to
detect material drift; the larger research sweeps remain documented here.

## Coverage contract

Every registered metric that can publish a non-null `p_value` must appear on
this page with either a measured size result or an explicit gap. A source-based
test enforces names, not a claim that every path is calibrated.

| Family | Covered p-value metrics |
|---|---|
| Series means and spreads | `ic`, `quantile_spread`, `quantile_spread_vw`, `k_spread`, `fm_beta` |
| Predictive and panel regressions | `predictive_beta`, `spanning_alpha`, `common_asymmetry`, `common_quantile_spread`, `pooled_beta`, `common_beta` |
| Event inference | `caar`, `bmp_z`, `corrado_rank`, `event_hit_rate`, `event_ic` |
| Direction, order, and trend | `directional_hit_rate`, `monotonicity`, `positive_rate`, `ic_trend` |

## HAC family calibration

The same overlapping input reaches different references depending on the rank
of the restriction. The ranges are intentionally kept beside the path rather
than repeated throughout the method guide.

| Path | Measured null size | Interpretation |
|---|---:|---|
| Scalar series mean with `NEWEY_WEST` | 3.9–7.3% over `T ∈ {60,120,240,500}`, `h ∈ {1,5,21}`; 5.4–8.1% on AR(0.6) at `h=1` | Calibrated to mildly liberal on the measured grid; conservative when `h` is declared on a non-overlapping series |
| `NON_OVERLAPPING` mean, including `ic`, `caar`, and spread metrics | 4.5–5.4% on measured overlap nulls | Removes mechanical overlap; does not remove persistence beyond the stride |
| `predictive_beta`, `h=1` | 4.3–5.5% at regressor persistence `ρ=0`; 6.2–8.3% in the strongest measured Stambaugh cells | Amihud-Hurvich path is characterised at one-period horizons |
| `predictive_beta`, `h>1` | **7.5–14.5%** | Known oversized overlapping-regression HAC regime |
| Rank-one `common_*` contrasts | 3.3–8.0% on non-persistent common-factor nulls; 7.3–16.3% on persistent input | Scalar reference fixes the ordinary overlap case; persistence remains a warning regime |
| `spanning_alpha` rank-one contrast | 5.7–11.7% on the measured overlapping-sum spread null | Uses the scalar reference, but the upper measured cells remain mildly oversized |
| Date-aligned multi-restriction slice Wald tests | 8–9% for `K=5` on 50–90 periods per slice | Known short-slice over-rejection; converges near 5.5% around 150 periods |
| `pooled_beta` Driscoll-Kraay | **2.6–8.7%** on its persistent-regressor, cross-sectionally correlated, overlapping-return panel grid | The scalar HAR reference reduced the former path's 6.3–22.5% range; the low-overlap-dependence cells can be conservative |

The scalar recipe combines three factrix-specific choices: a `3(h - 1)`
overlap floor, variance scale `T/(T-L-1)`, and effective degrees of freedom
bounded by `T/h - 1`. These move together. Applying only the wider lag floor
while keeping an ordinary reference made rank-one regressions worse in the
measured cells.

Producing tests include `tests/stats/test_hac_overlap_size.py`,
`test_overlap_floor_size.py`, `test_scalar_wald_overlap_size.py`,
`test_driscoll_kraay_size.py`, and `tests/test_stambaugh_bias.py`.
Run `python tests/stats/test_driscoll_kraay_size.py` from the repository root
to regenerate the complete 1,000-replication Driscoll-Kraay grid.

### Single-restriction Wald contrasts
[](){ #single-restriction-wald }

`common_asymmetry`, `common_quantile_spread`, and `spanning_alpha` each test
one linear restriction. Treating them like a multi-restriction Wald matrix
left the two common-factor metrics 10–34% oversized at `h > 1`. Moving the
bandwidth, finite-sample scale, and reference together to the scalar recipe
put every non-persistent common-factor cell at or below 8.0%.

The correction is not a persistence cure. At `T=60, h=5`, a common factor
with `φ=0.9` still produced 13.0% and 16.3% rejection rates for the two
metrics. `serial_correlation_detected` identifies that regime. Increasing a
raw lag override does not repair the reference.

### Shanken correction on `fm_beta`

On a true-null panel, the measured uncorrected versus Shanken rejection rates
were 7.3/7.3%, 3.3/3.3%, 4.3/4.0%, and 6.0/6.0% over
`(T,h)={(120,1),(120,5),(240,1),(240,5)}`. Under the null the estimated
premium approaches zero and the multiplicative correction approaches one, so
near-equality is expected. This table characterises size; it does not measure
power or the omitted additive term in the simplified single-factor variance.

## Persistence beyond the overlap horizon

[](){ #persistence-beyond-the-overlap-horizon }

The persistence screen reads lag-1 autocorrelation on the series after the
`overlap_periods` stride. Reading the raw series would flag the MA(`h - 1`)
structure that the stride or kernel was designed to handle.

On a directly constructed persistent per-period signal, the measured ranges
were:

| Lag-1 persistence | Plain t | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` |
|---:|---:|---:|---:|
| 0 | 7–9% | 7–9% | 7–9% |
| 0.6 | 32–34% | 13–17% | 12–19% |
| 0.85 | 55–61% | 32–34% | 20–32% |

No member is calibrated in the strongest rows. Switching from analytic to
bootstrap inference changes the number but does not resolve the model failure.
Use a raised hurdle, redesign the sample, or fit a model that explicitly
handles the persistence. A screen estimated close to the 10-period floor is
itself noisy; always read `n_obs`.

Andrews-Monahan AR(1) prewhitening is measured behind a private flag. It
substantially reduces moderate-persistence excess on univariate mean paths but
does not rescue near-unit-root input and is not implemented for every vector
or regression kernel. Keeping it private avoids implying a library-wide
correction that has not been calibrated.

## Spread and rank-based paths

| Metric or path | Measured result | Main limit |
|---|---|---|
| Spread-series `STATIONARY_BOOTSTRAP` | 4.8–9.0% on the iid-factor grid; 3.7–10.0% on the persistent-factor overlap grid | Long-horizon cells can be mildly liberal while NW becomes conservative |
| `monotonicity` stationary-bootstrap MR test | 2.0–7.3% over `T ∈ {120,240}`, `h ∈ {1,5}`, factor persistence `φ ∈ {0,0.9}` | Calibrated to conservative on the measured grid |
| `positive_rate` exact binomial | 2.3–4.7% in most cells; one IC-pipeline cell measured 8.0% at `n=240` | Exact-binomial discreteness is conservative at short `n`; the 8% cell is within two Monte Carlo SE of the iid cell, not evidence that every cell is at or below nominal |
| `directional_hit_rate` Pesaran-Timmermann with period deflation | 3.0–7.7% on the measured panel grid | Same-period dependence must remain visible in the event-period metadata |
| `ic_trend` strided Hamed-Rao Mann-Kendall | 1.7–5.3% over `T ∈ {60,120,240}`, `h ∈ {1,5}` | Strong residual persistence and unit-root input remain warning regimes |
| `common_beta` calendar-time cross-asset t | 2.0–7.3% on the measured common-factor panel grid | A hand-built beta table cannot recover the calendar-time variance and falls back to the iid cross-asset reference |

Producing tests include `test_spread_bootstrap_size.py`,
`test_monotonicity_size.py`, `test_positive_rate_size.py`,
`test_directional_hit_rate_size.py`, `test_ic_trend_size.py`, and
`test_common_beta_size.py` under `tests/stats/`.

## Event-study paths

`caar` and `corrado_rank` infer on event periods; `bmp_z` infers on valid
events and uses distinct event periods to assess clustering and sample
reliability. Their `n_obs` values are therefore not interchangeable.

| Path | Characterisation | Limit |
|---|---|---|
| `caar` non-overlapping event-period t | Falls in the 4.5–5.4% non-overlap band on the measured null | Event-induced variance motivates the standardised read |
| `bmp_z` with design-effect deflation | About 7% on the sign-aligned shared-shock null; around 10% at 8 effective event periods, around 7% at 15, clearing near 30 | Residual size follows distinct event periods, not raw events; heavy skew at long horizons can bias standardised abnormal returns |
| `corrado_rank` | 6% on the same sign-aligned null | Rank robustness does not create power from few event periods |
| `event_hit_rate` | 7% on the sign-aligned null | Uses the same-period dependence adjustment; still thin with few event periods |
| `event_ic` | Covered by the shared clustering characterisation and dedicated clustered-inference tests | A pooled event correlation remains sensitive to the effective event-period count |

### `event_skewness` has no calibrated test
[](){ #event-skewness-no-calibrated-test }
[](){ #event-skewness-has-no-calibrated-test }

The withdrawn D'Agostino path failed on both non-normal signed returns and
sign-aligned same-period shocks. On two panel nulls with only about 1.5 events
per event period, it rejected 19.0% and 23.3%; on a sign-aligned clustered
null it rejected 30.3%. Applying the event design-effect deflator barely moved
the first failure and drove the last to 0.0%, so no single adjustment
calibrated both. `event_skewness` is descriptive and publishes no p-value.

## Joint period test on short slices: known over-rejection
[](){ #joint-period-test-on-short-slices-known-over-rejection }

For date-disjoint slices, `slice_period_joint_test` with `K ≥ 3` and fewer
than roughly 150 periods per slice is not a reliably sized omnibus test. At
`K=5`, measured size is 8–9% for the analytic path and about 12% for the
bootstrap path at the short end. The bootstrap inherits noisy per-slice
variance estimates; it is not automatically safer because the sample is
short.

For pairwise contrasts, both methods were near nominal at `K=3, T=60`
(5.8% bootstrap, 5.4% analytic Holm). At `K=5`, analytic Holm measured
5.7–6.7% versus 7.4–8.5% for the bootstrap/Romano-Wolf path. Prefer analytic
pairwise contrasts or longer slices in that regime. The default remains a
compatibility choice, not a claim that bootstrap dominates finite samples.

## Uneven evaluation grids

The main size tables use a constant stride. Scalar series-mean paths were also
checked on one uneven `(20, 20, 40)`-spaced grid: `NON_OVERLAPPING` measured
5.8% and `NEWEY_WEST` 5.5%. Regression HAC, multi-restriction Wald, ADF,
rolling, event-window, and adjacent-period paths have not been comprehensively
recalibrated on uneven grids. `uneven_evaluation_grid` is therefore a design
warning, not a claim that every method is invalid or valid there.
