# Statistical methods

Cross-cutting statistical disciplines that govern multiple metrics in
factrix. This page sits **above** the per-metric API pages: it
describes the four discipline lines that recur across cells, explains
the variant of each that factrix implements, and points at the
[bibliography](bibliography.md) anchor for the source treatment.

For per-metric formulae and signatures see the
[Metrics API pages](../api/metrics/index.md). For the design choices
behind which disciplines factrix does *not* implement, see
[Development § Design notes](../development/design-notes.md).

The four sections are the only first-class disciplines in factrix:

1. **HAC SE under overlapping returns** — Newey-West with a
   deterministic bandwidth rule.
2. **Multiple-testing under dependence** — Benjamini-Yekutieli FDR,
   not Bonferroni.
3. **Robust scale and outlier handling** — MAD-based winsorisation
   with the consistency factor; Theil-Sen for slope.
4. **Persistence diagnostics under near-unit-root predictors** —
   ADF flag, no auto-correction.

---

## 1. HAC SE under overlapping returns

When forward returns span `h > 1` periods, consecutive observations
inherit MA(`h − 1`) structure. Two standard responses, with different
trade-offs: NW HAC corrects SE on the full series at the cost of a
kernel choice and asymptotic-Gaussian inference, while non-overlapping
sampling preserves an exact-distribution `t` at the cost of a factor
of `h` in effective sample size. factrix exposes both — non-overlap
as the default for cell-canonical metrics, NW HAC as an explicit
sibling.
When NW HAC is selected, factrix uses the
[Newey-West 1987][newey-west-1987] Bartlett kernel with a
deterministic bandwidth, applied by `ic_newey_west`, `fama_macbeth`,
`pooled_ols`, `ts_quantile_spread`, and `ts_asymmetry`.

The bandwidth rule is

$$
L = \max\!\left(\lfloor T^{1/3} \rfloor,\; h - 1\right)
$$

with $h$ = `forward_periods`. The first term is the [Andrews 1991][andrews-1991] optimal Bartlett
growth rate; the second is the
[Hansen-Hodrick 1980][hansen-hodrick-1980] overlap floor that ensures
the kernel covers the MA(`h − 1`) structure of overlapping returns.
factrix takes the maximum so the bandwidth is always at least large
enough to absorb the overlap, with the Andrews term taking over once
`T` is large.

Two choices factrix deliberately did not adopt:

- The data-adaptive plug-in of [Newey-West 1994][newey-west-1994] is
  not used. Its sampling variability defeats the point of having a
  reproducible reported SE; the deterministic Andrews rule is
  adequate at typical research `T` and is auditable.
- The prewhitening refinement of
  [Andrews-Monahan 1992][andrews-monahan-1992] is also not used.
  Same reason: deterministic outputs over marginal efficiency.

The [White 1980][white-1980] HC0 sandwich estimator is the
heteroskedasticity-only ancestor of NW and is mentioned in metric
docstrings as background, not implemented separately.

For the FM-cell, the NW HAC sits at stage 2
([Fama-MacBeth 1973][fama-macbeth-1973]). When the Stage-1 regressor
is itself an estimated quantity (rolling β, PCA score, ML predictor),
[`fama_macbeth(is_estimated_factor=True)`](../api/metrics/fama_macbeth.md)
applies the [Kan-Zhang 1999][kan-zhang-1999] single-factor
simplification of the [Shanken 1992][shanken-1992] errors-in-variables
correction, scaling SE by $\sqrt{1 + \hat\lambda^2 / \sigma^2_f}$. The
full Shanken variance has an additional $+\sigma^2_f / T$ term that
factrix omits: at finite $T$ the omission **understates** the EIV
inflation and so **overstates** the resulting $t$. The simplification
is honest only when $T$ is large enough that the dropped term is
negligible.

When `factor_return_var` is not supplied, factrix falls back to
$\mathrm{var}(\hat\beta_t)$ as a proxy for $\sigma^2_f$. Because
$\hat\beta_t$ already absorbs
estimation noise from the upstream factor score, this proxy
**inflates the denominator** of the EIV factor and so **further
deflates** the correction. Treat the
`betas_timeseries_proxy` result as a lower bound on the true
inflation — i.e. an upper bound on the reported `t`.

Default versus paired t-test is a separate choice: the cell-canonical
metrics (`ic`, `caar`) use **non-overlapping resampling** as the
default rather than NW HAC. NW is exposed as an explicit sibling
(`ic_newey_west`) for callers who prefer the HAC route. Resampling has
the advantage of exact rather than asymptotic-Gaussian inference at
the cost of a factor of `h` in effective sample size; users with long
panels often prefer NW.

---

## 2. Multiple-testing under dependence

Factor pools are dependent by construction: 200 momentum variants on
the same return panel correlate, and a Bonferroni step that assumes
independence over-corrects. factrix's `multi_factor.bhy` wrapper
implements [Benjamini-Yekutieli 2001][benjamini-yekutieli-2001] FDR
control with the dependence correction $c(m) = \sum_{i=1}^{m} 1/i$ —
valid under arbitrary positive or negative dependence at the cost of
a $1/\ln m$ shrinkage relative to plain BH.

[Benjamini-Hochberg 1995][benjamini-hochberg-1995] BH is *not* the
default because the typical factor-pool dependence violates its PRDS
assumption; factrix offers BHY as the safe choice and surfaces the
adjusted `q`-values rather than a binary pass/fail at a fixed `α`.

Three positions on multiple testing that the literature has converged
on and factrix takes:

- The [Harvey-Liu-Zhu 2016][harvey-liu-zhu-2016] case for raising
  t-thresholds is taken seriously. `verdict()` defaults to $t \geq 2.0$
  but exposes the BHY-adjusted `q` so users can apply a stricter
  threshold for new factor proposals.
- The [Harvey 2017][harvey-2017] case against ad-hoc p-hacking is the
  reason factrix runs registered procedures with fixed pipelines —
  the lag rule, the sample guards, the resampling stride are not
  user-tunable per call.
- The [White 2000][white-2000] reality-check and
  [Hansen 2005][hansen-2005] SPA family are *not* implemented. The
  cost of bootstrap-based data-snooping correction is high relative
  to the BHY recipe under realistic `m`, and the empirical
  disagreement is concentrated in the marginal cases where neither is
  decisive (see [Design notes § BHY rather than
  Bayesian](../development/design-notes.md#5-bhy-rather-than-bayesian-multiple-testing)).

Greedy forward selection (`greedy_forward_selection`) inflates t-stats
by selection and is documented as **not for inference**. The PoSI
literature ([Berk-Brown-Buja-Zhang-Zhao 2013][berk-brown-buja-zhang-zhao-2013],
[Leeb-Pötscher 2005][leeb-potscher-2005]) gives the rigorous
correction; factrix does not implement it because the function is
intended as a diagnostic, not a hypothesis test.

---

## 3. Robust scale and outlier handling

factrix preprocesses cross-sectional factor exposures with
**MAD-based winsorisation**: per date, clip values to
$\text{median} \pm k \cdot \mathrm{MAD} \cdot 1.4826$. The $1.4826$ factor restores Gaussian
consistency of the median absolute deviation as a scale estimator
([Huber 1964][huber-1964], textbook treatment in
[Huber 1981][huber-1981]). This avoids letting the same outlier that
breaks a sample mean break the scale estimator that gates its
treatment.

Theil-Sen slope is used for the trend metric (`ic_trend`) for the
same reason. The estimator computes the median pairwise slope
([Sen 1968][sen-1968]) and inherits a 29.3% breakdown point; the
SE recovered from the rank-based confidence interval is approximate,
not asymptotically exact, which is the trade-off factrix accepts in
return for not letting a single COVID-era IC spike dominate the slope.

Two robust-scale choices factrix did not adopt:

- The Sn / Qn estimators of [Rousseeuw-Croux 1993][rousseeuw-croux-1993]
  have higher Gaussian efficiency than MAD. The factrix winsorisation
  pipeline keeps MAD because the small efficiency gain does not
  justify the extra complexity at the boundaries (Sn / Qn need
  sorted-pair lookups; MAD is a single median).
- The stationary block bootstrap of
  [Politis-Romano 1994][politis-romano-1994] is cited as the proper
  way to recover SE under serial dependence when an analytical kernel
  is not available. factrix's parametric NW HAC is preferred because
  it is deterministic; the bootstrap is left to external packages
  (`arch`).

The influence-function framework underlying the breakdown-point
language is [Hampel 1974][hampel-1974].

---

## 4. Persistence diagnostics under near-unit-root predictors

Predictive regressions with persistent regressors carry the
[Stambaugh 1999][stambaugh-1999] bias: when the regressor's
innovation correlates with the dependent return innovation, OLS $\hat\beta$
carries a finite-sample bias of order $O(1/T)$ that does not vanish
at conventional research sample sizes ($\hat\beta$ is consistent
asymptotically, but the bias is large enough to flip inference at
$T \approx 10\text{–}30$ years of monthly data). The textbook corrections
([Campbell-Yogo 2006][campbell-yogo-2006] Bonferroni Q,
[Phillips-Magdalinos 2009][phillips-magdalinos-2009] /
[Kostakis-Magdalinos-Stamatogiannis 2015][kostakis-magdalinos-stamatogiannis-2015]
IVX) are out of factrix's lean-dependency scope.

factrix's response is **flag, do not fix**: `ic_trend(adf_threshold=
0.10)` runs an Augmented Dickey-Fuller test on the input series
([Dickey-Fuller 1979][dickey-fuller-1979], [Said-Dickey 1984][said-dickey-1984]
ARMA extension). When the ADF p-value exceeds `0.10` —
[Stock-Watson 1988][stock-watson-1988] practitioner cutoff — the
metadata records `unit_root_suspected=True` and the slope significance
is annotated with that caveat. The slope value itself is still
returned; the caller decides whether to trust it.

The ADF p-value is interpolated from
[MacKinnon 1996][mackinnon-1996] response-surface critical values for
the constant-only specification (`_adf_pvalue_interp`). The
interpolation accuracy is ±0.03 — ample for the qualitative "is this
a unit root" decision the threshold drives.

Overlapping multi-period returns inherit MA(`h − 1`) autocorrelation
([Richardson-Stock 1989][richardson-stock-1989]), which the NW lag
floor in section 1 absorbs. The persistence diagnostic in this
section is on the *input* series itself, not the residual structure
captured by HAC.

For per-asset β regressions in `compute_ts_betas`, factrix
deliberately retains plain OLS SE rather than HAC: the Stambaugh bias
arises from the predictor's persistence, not from SE estimation, and
HAC fixes only the SE while leaving the coefficient bias untouched.
Adding HAC there would advertise robustness factrix does not deliver.
