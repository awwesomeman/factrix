---
title: Statistical methods
---

!!! note "Audience"
    This page assumes working familiarity with applied econometrics
    (heteroskedasticity-and-autocorrelation-consistent (HAC) SE, false discovery rate (FDR) control, near-unit-root inference). If you are
    looking for a first-pass orientation to factrix output, start at
    [Quickstart](../getting-started/quickstart.md) and
    [Concepts](../getting-started/concepts.md); return here when you
    need the discipline-level rationale behind a specific metric.

Cross-cutting statistical disciplines that govern multiple metrics in
factrix. This page sits **above** the per-metric API pages: it
describes the five discipline lines that recur across cells, explains
the variant of each that factrix implements, and points at the
[bibliography](bibliography.md) anchor for the source treatment.

For per-metric formulae and signatures see the
[Metrics API pages](../api/metrics/index.md). For the design choices
behind which disciplines factrix does *not* implement, see
[Development § Design notes](../development/design-notes.md).

The five sections are the only first-class disciplines in factrix:

1. **HAC SE under overlapping returns** — Newey-West with a
   deterministic bandwidth rule.
2. **Multiple-testing under dependence** — Benjamini-Yekutieli FDR,
   not Bonferroni.
3. **Robust scale and outlier handling** — MAD-based winsorisation
   with the consistency factor; Theil-Sen for slope.
4. **Persistence diagnostics under near-unit-root predictors** —
   augmented Dickey-Fuller (ADF) flag, no auto-correction.
5. **Event-study cross-sectional inference** — CAAR cross-event $t$,
   BMP-style standardised AR, Corrado rank.

!!! note "Every estimator here is frequency-agnostic"
    No estimator in factrix reads the calendar (the library-wide principle:
    [Period grid, not calendar](../development/architecture.md#period-grid-not-calendar)). `date` is only an ordering
    key, and every window, lag, horizon and stride (`forward_periods`,
    `estimation_window`, `window`, Bartlett lags, block lengths) is a count
    of **periods on the panel's own distinct-date grid**, i.e. of whatever
    interval one period represents. There is no annualisation factor, no
    trading-day constant and no date arithmetic anywhere in the library, and
    the grid may be unevenly spaced. Where a docstring says "one period" it
    means one distinct date; "within-period" means "among the rows sharing
    one timestamp", whatever that timestamp's granularity. The
    caller owns frequency consistency between the factor, the return and
    the price column — see
    [Preparing data](../guides/preparing-data.md).

---

## 1. HAC SE under overlapping returns
[](){ #nw-hac }

When forward returns span `h > 1` periods, consecutive observations
inherit MA(`h − 1`) structure. Two standard responses, with different
trade-offs: Newey-West (NW) HAC corrects SE on the full series at the cost of a
kernel choice and asymptotic-Gaussian inference, while non-overlapping
sampling preserves an exact-distribution `t` at the cost of a factor
of `h` in effective sample size. factrix exposes both — non-overlap
as the default for the mainstream metrics, NW HAC as an explicit
sibling.
When NW HAC is selected, factrix uses the
[Newey-West 1987][newey-west-1987] Bartlett kernel with a
deterministic bandwidth. The full path-to-family map is the table in
[section 6](#hac-families).

There are **two** bandwidth rules, because the scalar mean $t$-test and
the $K$-restriction Wald tests degrade in opposite directions under a
wide kernel (see [section 6](#hac-families) for the full path table).

- **Scalar series-mean HAR $t$-test** (`ic` / `quantile_spread` /
  `quantile_spread_vw` / `k_spread` under `NeweyWest`, `fm_beta` stage 2) —
  `_resolve_har_lags`:
  $$
  L = \min\!\left(\max\!\left(1.3\sqrt{T},\; 3(h - 1)\right),\; \lceil T/3 \rceil\right)
  $$
  read against effective degrees of freedom
  $\nu = \max\!\left(\min\!\left(1.5T/L - 1,\; T/h - 1\right),\; 1\right)$,
  with the SE carrying a $T/(T - L - 1)$ finite-sample scale.
- **Multivariate / $K$-restriction HAC paths** (`pooled_beta`,
  `spanning_alpha`, `common_quantile_spread`, `common_asymmetry`, the
  slice Wald tests, `_ols.py`) — `_resolve_nw_lags`:
  $$
  L = \max\!\left(\text{auto\_bartlett}(T),\; h - 1\right)
  $$
  where $\text{auto\_bartlett}(T) = \max\!\left(1,\; \lfloor 4 \cdot (T/100)^{2/9} \rfloor\right)$ per Newey-West (1994),
  read against $t_{T-k}$ / $F_{r,\,T-k}$.

with $h$ = `overlap_periods` — the overlap of adjacent observations on the
evaluation grid, stamped by `compute_forward_return` (equal to
`forward_periods` on the full grid, derived from `dates=` on a coarser one).
The $1.3\sqrt{T}$ base is
[LLSW (2018)][llsw-2018]'s HAR recommendation; `auto_bartlett` is the
[Newey-West 1994][newey-west-1994] automatic Bartlett bandwidth; the
$h - 1$ term is the [Hansen-Hodrick 1980][hansen-hodrick-1980] overlap
floor that ensures the kernel covers the MA(`h − 1`) structure of
overlapping returns. factrix takes the maximum so the bandwidth is
always at least large enough to absorb the overlap.

Both bandwidth rules and the effective-df cap $T/h - 1$ read
`overlap_periods`, not the return horizon. On the full grid the two are
equal; on a coarser evaluation grid built with
`compute_forward_return(..., dates=)` the overlap is the derived
`1 + max_i #\{j : 0 < \mathrm{idx}(j) - \mathrm{idx}(i) < h\}`
([Evaluating on a coarser grid](../api/preprocess.md#evaluating-on-a-coarser-grid)).
**Disclosure:** the size tables in this section were measured on regular
grids (a constant stride, overlap equal to the horizon). On an unevenly
spaced evaluation grid the scalar series-mean paths were re-checked at the
derived overlap — `NonOverlapping` 5.8% and `NeweyWest` 5.5% at a nominal 5%
on the `(20, 20, 40)`-spaced grid in
`tests/stats/test_uneven_grid_overlap_size.py` — but the Amihud-Hurvich
path of `predictive_beta` and the $K$-restriction Wald paths have not been
re-measured there, and the $T/h - 1$ cap is less conservative on a
sub-sampled panel by construction.

#### Three departures from the textbook HAR form

Each is a factrix choice, not a published one, kept because it is
load-bearing for measured size on overlapping horizons. The counterpart
statements live in `factrix._stats.hac`'s docstring Notes.

| Piece | Standard | What factrix does | Why | Measured consequence |
|---|---|---|---|---|
| Overlap floor | $h - 1$ (Hansen-Hodrick consistency floor) | $3(h-1)$ | At $L = h-1$ the Bartlett weight sends the lag-$(h{-}1)$ autocovariance to $1/h$, so the kernel discards most of the overlap covariance it exists to capture; at $3(h-1)$ the mean weight across the MA band is ≈0.83 rather than ≈0.5. | $T{=}240$, $h{=}21$: size 10.6% → 5.9%. |
| SE scale | none (textbook is $\sqrt{\text{LRV}/T}$); Stata's `newey` scales by $T/(T-k)$ with $k$ = **regressor count**, never the bandwidth | $T/(T - L - 1)$ | Self-derived white-noise demeaning-bias correction: in-sample demeaning biases each $\hat\gamma_j$ by ≈$-\gamma_0/T$, giving $\mathbb{E}[\widehat{\text{LRV}}] \approx \gamma_0(1 - (L+1)/T)$, which this undoes exactly. | $h{=}5$: 8.2–9.6% → 6.2–6.7%. Partly double-counts with the fixed-$b$ reference (whose KV limit already embeds demeaning), so $h{=}1$ iid cells come out slightly *under*-sized (4.3–4.7% at $T \le 120$). |
| Effective df | LLSW's `harreg.ado` uses $\lceil 1.5T/S \rceil$ | $\min(1.5T/L - 1,\; T/h - 1)$ | The $-1$ is a sub-one-df small-sample conservatism. The $T/h - 1$ cap is self-derived: an $h$-period overlapping series carries at most $T/h$ independent observations however the kernel is tuned. | Cap: $T{=}60$, $h{=}21$ size 12.2% → 4.3%. Cost: passing `forward_periods` on a series with *less* dependence than $h$ implies makes the test markedly conservative — measured 0.2% ($T{=}60$), 2.1% ($T{=}120$), 3.5% ($T{=}240$) on iid input at $h{=}21$. |

Producing module: the three cells above are re-measured at reduced
replication by `tests/stats/test_hac_overlap_size.py`; the full sweep
behind the exact percentages is not committed.

Two choices factrix deliberately did not adopt:

- The data-adaptive plug-in of [Newey-West 1994][newey-west-1994] is
  not used. Its sampling variability defeats the point of having a
  reproducible reported SE; the deterministic Newey-West rule is
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
[`fm_beta(is_estimated_factor=True)`](../api/metrics/fm_beta.md)
applies the [Shanken (1992)][shanken-1992] single-factor case of the
errors-in-variables correction, scaling SE by
$\sqrt{1 + \hat\lambda^2 / \sigma^2_f}$ (Shanken's general multi-factor
multiplicative term $1 + \lambda'\Sigma_f^{-1}\lambda$ collapses to
$1 + \hat\lambda^2 / \sigma^2_f$ when there is one factor). The full
Shanken variance has an additional $+\sigma^2_f / T$ term that factrix
omits: at finite $T$ the omission **understates** the EIV inflation
and so **overstates** the resulting $t$. The simplification is honest
only when $T$ is large enough that the dropped term is negligible.

When `factor_return_var` is not supplied, factrix falls back to
$\mathrm{var}(\hat\beta_t)$ as a proxy for $\sigma^2_f$. Because
$\hat\beta_t$ already absorbs
estimation noise from the upstream factor score, this proxy
**inflates the denominator** of the EIV factor and so **further
deflates** the correction. Treat the
`betas_timeseries_proxy` result as a lower bound on the true
inflation — i.e. an upper bound on the reported `t`.

Default versus paired t-test is a separate choice: the mainstream
metrics (`ic`, `caar`) use **non-overlapping resampling** as the
default rather than NW HAC.
[](){ #non-overlap-default }
NW is exposed as an explicit sibling
(`ic(inference=fx.inference.NEWEY_WEST)`) for callers who prefer the HAC route. Resampling has
the advantage of exact rather than asymptotic-Gaussian inference at
the cost of a factor of `h` in effective sample size; users with long
panels often prefer NW.

When the non-overlapping effective sample (`n / overlap_periods`) is
thin, `ic` surfaces `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` on the
returned result — the post-stride series is too short for a reliable
`t`. Switching to `ic(inference=fx.inference.NEWEY_WEST)` keeps every
observation and recovers test power in that regime. The guidance is
one-directional: a thin non-overlapping sample is a reason to move to
NW, but an ample sample is **not** a reason to move back — both methods
are valid there and `ic` never changes the method for you.

### NW vs HH-1980 vs Hodrick-1992 — when to use which

The first three procedures target overlap-induced SE distortion; the
stationary bootstrap targets the same distortion plus the normal-
approximation assumption all three analytic methods still make:

| Procedure | Mechanism | Strengths | Weaknesses | Where factrix uses it |
|---|---|---|---|---|
| **Newey-West (1987)** | Bartlett-kernel HAC on the full overlapping series, bandwidth `L = max(bandwidth_base, h−1)`. | Simple, deterministic, asymptotically valid for arbitrary autocorrelation up to `L`. | Asymptotic Gaussian — finite-sample size distortion when `h/T` is non-trivial; bandwidth rule is conservative. | `ic` / `quantile_spread` / `quantile_spread_vw` / `k_spread` (with `NeweyWest` inference), `fm_beta` stage 2, `pooled_beta`, `common_quantile_spread`, `common_asymmetry`. |
| **Hansen-Hodrick (HH) (1980)** | A generalized method of moments (GMM)-style HAC estimator with a rectangular kernel truncated at `h−1`; the canonical reference for overlap-aware long-horizon SEs. | Targets the MA(`h−1`) residual structure overlap induces. | Rectangular kernel can yield a non-PSD covariance matrix in finite samples; still asymptotic. | Exposed as `factrix.inference.HANSEN_HODRICK` (rectangular-kernel SE → t-statistic → two-sided p-value), but deliberately in **no** metric's `applicable_inference` allowlist — passing it to a metric raises `IncompatibleInferenceError`. Also borrows the `h−1` lag idea as a floor on the NW bandwidth above. |
| **Hodrick (1992) "1B"** | Reverse-regression: regress one-period return on the predictor sum `X_t = Σ x_{t-j}` over the last `h` periods. | Size-correct in finite samples even at large `h/T`; no bandwidth choice. | Coefficient interpretation differs — `β` is the response to a cumulative-predictor stimulus (MA on the RHS) rather than a long-horizon forecast slope (MA on the LHS in the standard form); not a drop-in replacement for the canonical `β`. | **Not implemented**. Cited as the right tool when overlap is severe; the `Individual × Continuous` cell side-steps the issue with non-overlapping resampling instead. |
| **Stationary bootstrap (Politis-Romano 1994)** | Block-resamples the series (geometric block length, Politis-White 2004 automatic selection), centred under `H0`, studentizes both the observed and the resampled root by a batch-means SE at the same block length, and reports the empirical two-sided p from the resampled `t` ratios. | No normality or asymptotic-variance assumption at all — valid when the IC/return distribution is heavy-tailed or skewed enough that NW's / HH's Gaussian p-value is itself suspect. | Heavier to compute (resampling, not closed-form); the reported `stat` is the observed mean rather than the root the p is computed from. On a zero-dispersion sample there is no block SE to divide by, and the kernel falls back to the raw-mean root — reported as `metadata["studentized"] = False` and `degenerate_variance`. | Exposed as `factrix.inference.STATIONARY_BOOTSTRAP` on `ic` only — `quantile_spread` / `k_spread` dispatch on a hard `isinstance(NeweyWest)` check that would need to go polymorphic first. |

Practical rule of thumb:

- `h ≤ 1`: NW with the Andrews rule is fine; the HH-1980 floor is
  inactive.
- `1 < h, h/T ≤ 0.05`: NW + HH-1980 floor is the factrix default; the
  finite-sample bias is small enough.
- `h/T > 0.05`: prefer non-overlapping resampling (`ic`, `caar`
  defaults) over NW. If a slope estimate is needed and resampling
  burns too much sample, Hodrick-1992 1B is the literature-preferred
  alternative; pre-compute externally.
- Distribution itself in doubt (heavy tails, skew) regardless of `h/T`:
  prefer `STATIONARY_BOOTSTRAP` over any of the analytic methods above —
  it is the only one of the four that does not assume asymptotic
  normality.

---

## 2. Multiple-testing under dependence
[](){ #bhy }

!!! tip "Canonical reference"
    For when to use Benjamini-Hochberg-Yekutieli (BHY), family partitioning, and worked screening recipes, see [BHY screening](../api/bhy.md). This section is the underlying theorem and assumptions.

!!! note "BH / BY / BHY — three names, two procedures"
    The literature uses three labels that map to two distinct procedures:

    - **BH** — [Benjamini & Hochberg (1995)][benjamini-hochberg-1995]. Original FDR step-up; requires independence or positive regression dependence on a subset (PRDS) among the test statistics.
    - **BY** — [Benjamini & Yekutieli (2001)][benjamini-yekutieli-2001]. Generalises BH to arbitrary dependence by dividing the threshold by $c(m) = \sum_{i=1}^{m} 1/i$. **This is the procedure factrix's `multi_factor.bhy()` implements mathematically.**
    - **BHY** — quant / factor-research shorthand (e.g. [Harvey-Liu-Zhu 2016][harvey-liu-zhu-2016]) for the BY-2001 procedure, naming all three authors of the BH/BY lineage. Pure statistics / biostatistics literature (R `mutoss`, `sgof`, etc.) uses **BY** for the same procedure.

    factrix follows the quant convention: the function is `bhy()`, the abbreviation in prose is `BHY`, the full form on first use is `Benjamini-Hochberg-Yekutieli`. Paper-citation links (`[Benjamini & Yekutieli (2001)][benjamini-yekutieli-2001]`) still point at the actual two-author paper because that is the work being cited.

Factor pools are dependent by construction: 200 momentum variants on
the same return panel correlate, and a Bonferroni step that assumes
independence over-corrects. factrix's `multi_factor.bhy` wrapper
implements [Benjamini-Yekutieli 2001][benjamini-yekutieli-2001] FDR
control with the dependence correction $c(m) = \sum_{i=1}^{m} 1/i$ —
valid under arbitrary positive or negative dependence at the cost of
a $1/\ln m$ shrinkage relative to plain Benjamini-Hochberg (BH).

[Benjamini-Hochberg 1995][benjamini-hochberg-1995] BH is *not* the
default because the typical factor-pool dependence violates its positive regression dependence on a subset (PRDS)
assumption; factrix offers BHY as the safe choice and surfaces the
adjusted `q`-values rather than a binary pass/fail at a fixed `α`.

Three positions on multiple testing that the literature has converged
on and factrix takes:

- The [Harvey-Liu-Zhu 2016][harvey-liu-zhu-2016] case for raising
  t-thresholds is taken seriously. Single-factor pre-registered comparisons default to $t \geq 2.0$
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
**MAD-based winsorisation**: per period, clip values to
$\text{median} \pm k \cdot \mathrm{MAD} \cdot 1.4826$. The $1.4826$ factor restores Gaussian
consistency of the median absolute deviation as a scale estimator
([Hampel 1974][hampel-1974] is the canonical reference popularising
MAD as a scale estimator; textbook treatment in
[Huber 1981][huber-1981]). This avoids letting the same outlier that
breaks a sample mean break the scale estimator that gates its
treatment.

Theil-Sen slope is used for the trend metric (`ic_trend`) for the
same reason. The estimator computes the median pairwise slope
([Sen 1968][sen-1968]) and inherits a 29.3% breakdown point; the
SE recovered from the rank-based confidence interval is approximate,
not asymptotically exact, which is the trade-off factrix accepts in
return for not letting a single COVID-era information coefficient (IC) spike dominate the slope.

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

The influence-function framework that places robust-scale and
breakdown-point reasoning on a common conceptual map is
[Hampel 1974][hampel-1974].

---

## 4. Persistence diagnostics under near-unit-root predictors

Predictive regressions with persistent regressors carry the
[Stambaugh 1999][stambaugh-1999] bias: when the regressor's
innovation correlates with the dependent return innovation, ordinary least squares (OLS) $\hat\beta$
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

The lag order of the augmentation is selected by AIC over
$0 \ldots \lfloor 12 (T/100)^{1/4} \rfloor$ ([Schwert 1989][schwert-1989]
ceiling) on a common estimation sample, then refitted on the full sample —
the `statsmodels.tsa.stattools.adfuller(autolag="AIC")` procedure. The
series this is applied to carry MA($h-1$) autocorrelation from overlapping
forward returns, so the un-augmented $\text{lags}=0$ regression (the
earlier default) is mis-sized. Pass `lags=` explicitly to `_adf` to fix
the order.

The ADF p-value is interpolated from
[MacKinnon 1996][mackinnon-1996] response-surface critical values for
the constant-only specification (`_adf_pvalue_interp`); the upper tail
uses Fuller's $\tau_\mu$ points ($-0.44$ / $-0.07$ / $0.23$ / $0.60$ at
90 / 95 / 97.5 / 99%). The interpolation accuracy is ~±0.02 — ample for
the qualitative "is this a unit root" decision the threshold drives.

Overlapping multi-period returns inherit MA(`h − 1`) autocorrelation
([Richardson-Stock 1989][richardson-stock-1989]), which the NW lag
floor in section 1 absorbs. The persistence diagnostic in this
section is on the *input* series itself, not the residual structure
captured by HAC.

For per-asset β regressions in `compute_common_betas`, factrix
deliberately retains plain OLS SE rather than HAC.
[](){ #stage1-plain-se }
The Stambaugh bias arises from the predictor's persistence, not from
SE estimation, and HAC fixes only the SE while leaving the coefficient
bias untouched. Adding HAC there would advertise robustness factrix
does not deliver.

---

## 5. Event-study cross-sectional inference

The `individual_sparse` cell aggregates per-event abnormal returns
across assets. Three estimators are exposed, each making a different
assumption about the cross-event distribution:

### CAAR cross-event t
[](){ #caar-cross-event-t }

Default for `caar`. Per event period, take the cross-sectional mean of
$\text{return} \times \text{factor}$ across event rows; the resulting
CAAR series is greedily subsampled by `date_ordinal` so consecutive
kept event periods are at least `overlap_periods` periods apart,
then tested with a standard $t$ on the sampled mean. `n_obs` is the
non-overlap event-period count, not the raw event count or full calendar
length. Specification follows [Brown-Warner 1985][brown-warner-1985].
The test is correctly sized under no event-induced variance; under variance
inflation around the event period it is mis-specified — the documented
motivation for switching to the BMP-style estimator below.

### BMP standardised AR
[](){ #bmp-standardised-ar }

`bmp_z`. Standardises each event's abnormal return by its
estimation-window standard deviation before taking the cross-event
mean, restoring size under event-induced variance
([Boehmer-Musumeci-Poulsen 1991][boehmer-musumeci-poulsen-1991]).
factrix's implementation is a **BMP-style simplification**: by
default it uses mean-adjusted abnormal returns and omits the original
BMP prediction-error correction, so results do not match a textbook
BMP implementation byte-for-byte. The strict denominator is
available via `bmp_z(..., include_prediction_error_variance=True)`
when the textbook form is required.

### Corrado nonparametric rank
[](){ #corrado-rank }

`corrado_rank`. Replaces returns with their uniform rank within
the (estimation ∪ event) window, averages each event period's ranks into
one observation, then runs the $z$ on that event-period series
([Corrado 1989][corrado-1989]). Robust to extreme returns,
non-normality, cross-asset heteroscedasticity, and **same-period event
clustering** — the last because collapsing each date first puts the
within-period correlation into the denominator, which is what makes this
the honest fallback when `clustering_hhi` says `caar`'s $t$ cannot be
trusted. `n_obs` therefore counts event *periods*, and a factor firing on
fewer than `MIN_EVENTS_HARD` periods short-circuits rather than estimating
a time-series SD from a handful of points — the same floor `caar` applies
to its own event-period series.

factrix adds the direction adjustment of
[Corrado-Zivney 1992][corrado-zivney-1992] for two-sided signed signals —
the rank itself is signed by $\text{sign}(\text{factor})$ before
aggregation, not the underlying return.

The denominator follows the *intent* of Corrado's eq. (5) (time-series SD
of a cross-sectional mean) rather than its literal form: Corrado's design
is event-time aligned so every date's cross-section is the full universe,
whereas a calendar-time sparse panel has thin event periods against full
non-event ones. Taking the SD over all periods there would understate the
relevant dispersion several-fold, so it is taken over the event-period
series, whose scale matches the numerator by construction.

## 6. Known simplifications (deliberately retained)

This section records the estimator simplifications kept deliberately, and the
calibration measurements that map each test to the HAC family and reference
distribution it actually uses. Each is intentional and has been reviewed; the
record exists so the choices are not re-litigated.

### Within-period clustering: ANOVA ICC(1) and the design-effect deflator

`_estimate_within_date_icc` (feeds the clustering deflation in `bmp_z` and
`directional_hit_rate`) is the one-way random-effects ANOVA estimator
$\hat r = (\text{MSB} - \text{MSW}) / (\text{MSB} + (n_0 - 1)\,\text{MSW})$
([Shrout-Fleiss 1979][shrout-fleiss-1979] ICC(1), Donner-Koval $n_0$ for
unbalanced periods), clipped to $[0, 1]$. An earlier version used the raw
between/total ratio $\operatorname{Var}(\bar x_d) / (\operatorname{Var}(\bar x_d) +
\hat\sigma^2_w)$; because $\mathbb{E}[\operatorname{Var}(\bar x_d)] = \sigma^2_b +
\sigma^2_w / n$, that ratio converges to $1/(n+1)$ under **independence** and
the deflator fired at full strength on unclustered data (empirical size
$\approx 1\%$ at nominal 5%).

The deflator itself is the Kish design effect $1/\sqrt{1 + (n_0 - 1)\hat r}$,
i.e. [Kolari-Pynnönen 2010][kolari-pynnonen-2010] **without** the $(1 - \bar r)$
numerator. K-P's numerator corrects a cross-sectional variance estimated on a
*single* event day (which under clustering estimates only $\sigma^2(1 - \bar
r)$). factrix pools SARs / hit indicators across many event periods, so the pooled
variance already contains the between-date component; applying $(1 - \bar r)$
on top would double-count. Choosing the design-effect form is the
engine-specific decision; the textbook K-P form is correct for the single-day
BMP setting it was derived in.

### Which HAC family a test is in
[](){ #hac-families }

After the HAR calibration work, the same overlapping $h$-period input can
reach two different bandwidth rules and three different reference
distributions depending on which test consumes it. This table is the map;
sizes are at a nominal 5% on a true null.

| Path | Bandwidth | Reference distribution | Measured size |
|---|---|---|---|
| `ic` / `quantile_spread` / `quantile_spread_vw` / `k_spread` (`NEWEY_WEST`), `fm_beta` stage 2 — scalar series mean | `_resolve_har_lags`: $\min(\max(1.3\sqrt T, 3(h{-}1)), \lceil T/3\rceil)$, SE scaled by $T/(T{-}L{-}1)$ | $t_\nu$, $\nu = \min(1.5T/L - 1,\ T/h - 1)$ | 3.9–7.3% over $T \in \{60,120,240,500\} \times h \in \{1,5,21\}$; 5.4–8.1% on AR(0.6) input at $h=1$; 0.2–3.5% (conservative) when `forward_periods` is passed on a series that is not actually overlapping |
| `ic` / `caar` / `quantile_spread` under `NON_OVERLAPPING` — strided mean | none (stride $h$, no kernel) | $t_{n_{\text{strided}}-1}$ | 4.5–5.4% in every overlapping cell measured; 32% on AR(0.6) input at $h=1$, which striding cannot touch |
| `predictive_beta` — single-restriction slope, $h = 1$ | none: Amihud-Hurvich's homoskedastic $s^2(X'X)^{-1}$ | $t_{m-3}$ | 4.3–5.5% at $\rho = 0$; 6.2–8.3% in the strongest Stambaugh cells |
| `predictive_beta` — single-restriction slope, $h > 1$ | `_resolve_har_lags` | $t_\nu$ via `_har_dof` | **7.5–14.5%** — known-oversized, see below |
| `spanning_alpha`, `pooled_beta`, `common_quantile_spread`, `common_asymmetry`, `_ols.py`, the slice tests — $K$-restriction Wald / multivariate | `_resolve_nw_lags`: $\max(\text{auto\_bartlett}(T), h-1)$ | $t_{T-k}$ / $F_{r,\,T-k}$ | 5.4–7.1% on iid residuals ($n = 60 \to 240$), 11.5–18.5% on AR(0.6); **8–9%** for the $K = 5$ slice joint test on 50–90-period slices |

Producing modules: `tests/stats/test_hac_overlap_size.py` for the scalar
series-mean rows, `tests/test_stambaugh_bias.py` for the `predictive_beta`
rows, and `tests/test_slice_period_joint_test.py` for the slice-test band
in the last row.

Two of these rows are **known-oversized regimes rather than calibrated
ones**, and are disclosed rather than corrected:

- The $K > 1$ Wald and slice tests keep the narrow bandwidth deliberately.
  A wide bandwidth on a $K \times K$ HAC matrix read against $\chi^2$ /
  $F$ critical values is exactly the case fixed-$b$ theory says needs
  Kiefer-Vogelsang / LLSW $F$-type critical values — moving them to the HAR
  rule measured *worse* (the $K = 5$ slice test goes from 8–9% to 21% at 50
  periods per slice). The narrow rule is the lesser evil until fixed-$b$
  Wald critical values are implemented.
- `predictive_beta` at $h > 1$ is the single-restriction case, so it does
  use the HAR rule; that took it from 10–19% to 7.5–14.5%, not to 5%. The
  excess is present at $\rho = 0$ for every $\phi$ and plain OLS-NW carries
  it too, so it is the overlapping-regression HAC problem rather than
  anything about the Stambaugh correction.

The residual anti-conservatism on **persistent** input is a property of the
Bartlett kernel rather than of any one metric, so every row above inherits
it. Measured on the `spanning_alpha` path (true null, one base factor, 4000
draws):

| residuals | $n=60$ | $n=120$ | $n=240$ |
|---|---|---|---|
| iid | 0.071 | 0.065 | 0.054 |
| AR(0.6) | 0.185 | 0.135 | 0.115 |

The AR column converges slowly enough to matter at realistic sample
lengths. A finite-sample fix (fixed-$b$ critical values for the Wald paths,
or Andrews-Monahan prewhitening) would change every HAC $p$-value in the
library and is a project rather than a patch. Read HAC $p$-values near the
threshold as optimistic when the input is persistent — the `persistence`
diagnostic in section 4 flags exactly that case.

### Joint period test on short slices: known over-rejection

`slice_period_joint_test` with `K ≥ 3` slices shorter than ~150 periods
over-rejects on a true null — measured 8–9% at a nominal 5% for `K = 5`
with 50–90-period slices, converging to ~5.5% by `T = 150`; `K = 2` is
calibrated throughout. The cause is the per-slice Bartlett HAC variance
estimate, whose effective degrees of freedom at `T = 50` are ≈ 21 rather
than `T − 1`: with the *true* variances the same Wald statistic rejects
3.8%, so neither the aggregation nor the `F` reference is at fault. The
bootstrap path inherits the same noise (12% at `K = 5`, `T = 50`).
Andrews-Monahan prewhitening, the Newey-West (1994) plug-in bandwidth, a
Hotelling-style `F` on the HAC effective df, and a Satterthwaite ν on
that df were each measured; none calibrates the iid short-slice case
(prewhitening is the right tool for *autocorrelated* input — see the HAC
size sweep tracker). The function warns in this regime and the
characterisation test in `tests/test_slice_period_joint_test.py` pins the
measured band; pairwise contrasts on the
same slices (5–6%) are the better-calibrated read.
### Persistent per-period series: no HAC or bootstrap path is calibrated

Every mean test on a per-period series (`ic` under any inference member,
the spread metrics, `fm_beta`) is measured on a true null against the
series' own persistence (`build_autocorrelated_ic_panel`, nominal 5%):

| lag-1 φ | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` | plain *t* |
|---|---|---|---|
| 0 | 7–9% | 7–9% | 7–9% |
| 0.6 | 13–17% | 12–19% | 32–34% |
| 0.85 | 32–34% | 20–32% | 55–61% |

Above φ ≈ 0.3 the Bartlett kernel's small-sample understatement of the
long-run variance is no longer a nuisance but the dominant term, and the
block bootstrap inherits it. This is the well-documented short-sample
behaviour of Newey-West, not a factrix defect, and the field's response
is not a different kernel: report the *t* against a raised hurdle
([Harvey, Liu & Zhu (2016)][harvey-liu-zhu-2016]: *t* > 3) or lengthen
the sample. factrix screens the tested series' lag-1 autocorrelation
and raises `WarningCode.SERIAL_CORRELATION_DETECTED` above
`PERSISTENT_SERIES_AUTOCORR` (0.3) so the regime is never silent.

Measured but deliberately **not** adopted: the Newey-West (1994) plug-in
bandwidth (worse than the fixed rule on iid input), longer Bartlett
bandwidths (no change on real overlapping series), and the Hansen-Hodrick
flat kernel (matches NW on real overlapping series; no PSD guarantee).
**Default is plain Newey-West; Andrews-Monahan prewhitening is measured
and available, not on.** Neither estimator is right everywhere. Plain
Newey-West at the automatic bandwidth understates the long-run variance
of a persistent series at the sample sizes factor research works with
(AR(0.6): 50% of the truth at `n = 50`, 61% at `n = 150`), so a drifting
IC or spread series produces a *t* that looks one to three times as
significant as the evidence warrants. [Andrews-Monahan
(1992)][andrews-monahan-1992] AR(1) prewhitening — strip the drift
component before the kernel sees it, then recolour — removes most of that
but not all of it, and none of it near a unit root. It is implemented
behind a private flag on `_newey_west_se` (the univariate mean-SE kernel
behind `NEWEY_WEST` and `fm_beta`) so both estimators are measured and
pinned side by side.

The default stays plain Newey-West because **matching published
factor-research numbers is a core use of this library**, and the
convention in that literature is plain Newey-West with a lag rule; the
tool ecosystem is split (R's `sandwich::NeweyWest` defaults prewhitening
on, statsmodels and Stata's `newey` do not implement it). This is a
convention choice, not a correctness claim — and it follows a line every
deliberate deviation in factrix has respected so far: BHY, the sample
floors and the persistence screen are all *additive* (they warn or
refuse); none silently moves a number a user is comparing to a paper.
Prewhitening by default would have been the first to cross it, on
exactly the input users most often compare. What the default gets wrong
on persistent input is therefore surfaced by `SERIAL_CORRELATION_DETECTED`
rather than corrected. Against the plain Bartlett estimate (nominal 5%):

| input | n | plain Bartlett | prewhitened |
|---|---|---|---|
| iid | 50 / 240 | 8.3% / 5.9% | 9.3% / 6.0% |
| AR(0.6), pure | 60 / 240 | 15.6% / 11.5% | 8.4% / 5.0% |
| AR(0.85), pure | 240 | 27.5–28.9% | 8.9–10.6% |
| IC fixture, φ = 0 (its own baseline) | 240 | 9.2% | 9.2% |
| IC fixture, φ = 0.6 | 240 | 14.4% | 8.0% |
| IC fixture, φ = 0.85 | 240 | 32.8% | 15.6% |
| real overlapping IC, h=5 | 240 / 480 | 5.2% / 8.8% | 5.2% / 8.4% |

The prewhitened-versus-plain bands above are pinned as characterisation
bands in `tests/stats/test_prewhitening.py`.

Two independent implementations agree on every row above. Read the IC
fixture against its *own* φ = 0 baseline of 9.2% (a property of the
fixture — 40-name cross-sections and a non-normal IC — not of
persistence): prewhitening removes the persistence-driven excess entirely
at φ = 0.6 (8.0% is below baseline) and ~73% of it at φ = 0.85
(0.236 → 0.064 over baseline). On pure AR(1) input it returns Newey-West
to its iid baseline outright. On iid and real overlapping input the two
estimates are indistinguishable, so the change is confined to the regime
it targets.

**What prewhitening does not do.** Near a unit root the AR(1) fit is
biased down (φ̂ ≈ 0.96 at true φ = 0.99), so the ±0.97 clip never bites
and the recolouring stays bounded (SE ratio 1.7–3.0×) — there is no
over-correction risk, but there is also no rescue: pure AR(1), n = 240,
prewhitened rejects 17% at φ = 0.95, 50% at 0.99, and 99.8% at a true
unit root, the same as plain Bartlett. No kernel change — this one or
any future one — relaxes `SERIAL_CORRELATION_DETECTED`: a user who knows
the kernel was improved will assume a persistent series is handled.
Should the default ever change, the multivariate `_nw_hac_vector_mean`
and the regression kernels `_ols_nw_slope_t` / `_ols_nw_multivariate`
would need their own measurement first — a vector series needs a VAR(1)
fit and regression scores a different derivation.

### Which path to read: a routing guide from the size measurements

The measurements above and in §1 say where each inference path is
calibrated and where it is not. This table turns them into the choice a
user faces after a warning fires — *keep the path, switch member, or
change the sample*. None of these rows changes a default: every deliberate
deviation in factrix is additive (it warns or refuses), so the routing is
the reader's, not the library's. Sizes are true-null rejection rates at a
nominal 5%; sample sizes count periods on the evaluation grid after the
`overlap_periods` stride.

| Input regime | What you see | What the measurements say | Read / do |
|---|---|---|---|
| Overlapping `forward_periods` panel, per-period series not persistent (the everyday case) | No warning | `NEWEY_WEST` 5–9% on real overlapping IC (h = 5, n = 240 / 480: 5.2% / 8.8%); `NON_OVERLAPPING` calibrated. | Keep the default member. |
| Persistent per-period series (lag-1 φ ≥ 0.3 on the *tested* series) | `serial_correlation_detected` | No path is calibrated — NW 13–17%, stationary bootstrap 12–19%, plain *t* 32–34% at φ = 0.6 (table above). | Do **not** switch member; it moves the number without fixing it. Read the *t* against a raised hurdle (*t* > 3) or lengthen the sample. A coarser `overlap_periods` stride under `NON_OVERLAPPING` also helps mechanically — the strided sample sits at φ^h, and AR(0.6) strided at h = 21 measures 4.5%. |
| Short strided series (fewer than ~120 periods after the stride) | `unreliable_se_short_periods`, or nothing on a moderately short series | The *t* / NW branch runs 7–9%. A block-bootstrap p is *worse* here — the spread metrics' former bootstrap branch (`_block_bootstrap_diff_p`, kernel-isolated on iid input) measured 13.6% at n = 12, 9.8% at 30, 7.4% at 60, reaching 5.2% only by 120 — and the strided series is short exactly when the horizon is long. | Stay on the analytic *t* / NW. Do not reach for the bootstrap to rescue a short sample; shorten the stride or lengthen the history instead. |
| Long strided series (≥ ~120 periods) whose distribution is in doubt (heavy tails, skew) | No warning | On iid input `STATIONARY_BOOTSTRAP` sits at the same 7–9% baseline as NW (φ = 0 row above) — it buys no *size*. Its case is distributional: it is the only member that does not assume asymptotic normality of the mean (§1). | Read `STATIONARY_BOOTSTRAP` on `ic` alongside the analytic p when tails or skew are the doubt; the spread metrics keep `NON_OVERLAPPING` / `NEWEY_WEST` (their allowlist). A documented option, not a default. |
| Heavy-tailed *and* short | `unreliable_se_short_periods` | The *t* is size-robust to tails — 3–4% on t(3) input, i.e. conservative — while the small-n bootstrap is not (6–14%). | Keep the *t*. Tails are not a reason to bootstrap a short series. |
| Thin cross-section (few names per leg) | `few_assets` | Each leg mean rests on a handful of names: a noisier estimate, not a differently-distributed one. The automatic bootstrap switch this once triggered rejected 8–20% against the *t*'s 7–9% and keyed on the wrong axis; it was removed. | Keep the requested member; read `n_assets` and treat the spread as fragile. |
| Joint test on K ≥ 3 short slices | `slice_period_joint_test` warning | 8–9% for K = 5 on 50–90-period slices, converging by T ≈ 150; the bootstrap path inherits it (12%). K = 2 is calibrated throughout. | Read the pairwise contrasts on the same slices (5–6%) rather than the joint p, or lengthen the slices. |
| Few event periods after the stride | `few_events` | Power-thin, not size-inflated for `caar` / `corrado_rank`; `bmp_z` is ~10% at 8 effective periods, ~7% at 15, clearing by ~30. | Read borderline p-values cautiously; extend the event history rather than switching estimator — all three count the same event periods. |
| 3–19 portfolio periods in `top_concentration` | `borderline_portfolio_periods` | Extremely conservative at the bottom of the range: 0 of 250 null draws rejected at exactly 3 periods. | Treat `value` as descriptive until the series is well inside the range; the p carries essentially no information there. |

Two rules fall out of the table. A *size* problem driven by persistence or
a short sample is not fixed by changing the inference member — the
analytic and bootstrap paths inherit the same small-sample distortion —
so the honest response is a raised hurdle or more periods. A
*distribution* problem on a long series is the one case where the
bootstrap earns its cost — as a second read, not a better-sized one — and
it is offered there rather than routed to automatically because a default
that silently moves a number is the line every other deviation in this
library has stayed behind.

---

## 7. Missing-value convention (null vs NaN)

polars distinguishes `null` (missing) from float `NaN` (a value), and
`drop_nulls` / `mean` / `std` / `sum` / `rank` do **not** skip NaN — a NaN
propagates through a mean, ranks above every finite value, and compares
`False` to everything (so `x > 0` counts it as a miss and `x != 0` as an
event). pandas users are used to `skipna=True` hiding this; there is no such
switch in polars, so factrix fixes one convention across the library:

1. **Producers drop and record.** A per-period primitive (`compute_ic`,
   `compute_caar`, the quantile bucketing, the beta primitives) drops a
   non-finite input row or an undefined per-period statistic (e.g. the
   Spearman $\rho$ of a constant cross-section, which `pl.corr` returns as
   NaN) at the boundary, and reports the count through `_drop_stats` /
   `n_*_dropped` metadata so the shrinkage is visible.
2. **Consumers use `drop_nulls().drop_nans()`.** Every series consumer treats
   NaN exactly like null: it is a missing observation, never a value. The
   headline `value`, `stat`, `p_value` and `n_obs` are computed on the same
   surviving sample.
3. **Kernels refuse non-finite input.** `factrix._stats` primitives
   (`_newey_west_*`, `_hansen_hodrick_*`, `_block_bootstrap_diff_p`) raise
   `ValueError` on NaN / inf — scipy's `nan_policy="raise"` semantics — rather
   than emit a NaN statistic or, worse, a spuriously small $p$ (an all-NaN
   bootstrap centring makes every `|boot| >= |obs|` comparison `False` and the
   empirical $p$ collapses to $1/(B+1)$).

The alternative — pandas-style silent `skipna` inside every reduction — was
rejected because it hides sample shrinkage from the reader and would still
leave `rank` / comparison semantics wrong. Imputing NaN to 0 was rejected
because a zero is a *value* (it pulls means toward zero and hit rates down).

