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

There are **two** bandwidth rules, and the split is by **restriction
count**, not by whether the fit is univariate: a scalar statistic (a series
mean, or a rank-one regression contrast) and a $K \ge 2$ Wald statistic
degrade in opposite directions under a wide kernel (see
[section 6](#hac-families) for the full path table).

- **Scalar series-mean HAR $t$-test** (`ic` / `quantile_spread` /
  `quantile_spread_vw` / `k_spread` under `NeweyWest`, `fm_beta` stage 2) —
  `_resolve_har_lags`:
  $$
  L = \min\!\left(\max\!\left(1.3\sqrt{T},\; 3(h - 1)\right),\; \lceil T/3 \rceil\right)
  $$
  read against effective degrees of freedom
  $\nu = \max\!\left(\min\!\left(1.5T/L - 1,\; T/h - 1\right),\; 1\right)$,
  with the SE carrying a $T/(T - L - 1)$ finite-sample scale.
- **Single-restriction regression contrasts** (`spanning_alpha`,
  `common_quantile_spread`, `common_asymmetry`, `_ols.py`) —
  `_resolve_scalar_wald_hac`: the *same* recipe as the scalar series mean,
  applied to $R\beta$ with $R$ of rank one. A rank-one contrast is a scalar
  statistic, so it inherits that calibration; the bandwidth, the
  $T/(T-L-1)$ variance scale and the effective $\nu$ move together.
- **Multi-restriction / cluster-mean Wald paths** (the slice Wald tests) —
  `_resolve_nw_lags`:
  $$
  L = \max\!\left(\text{auto\_bartlett}(T),\; h - 1\right)
  $$
  where $\text{auto\_bartlett}(T) = \max\!\left(1,\; \lfloor 4 \cdot (T/100)^{2/9} \rfloor\right)$ per Newey-West (1994),
  read against $F_{r,\,T-1}$. A $K$-restriction Wald statistic inverts a
  $K \times K$ HAC matrix and degrades under the wide kernel that helps a
  scalar one, so this family keeps the narrow rule (section 6).

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
sub-sampled panel by construction. Such a grid is disclosed at source:
`compute_forward_return(..., dates=)` fires
[`uneven_evaluation_grid`](warning-codes.md) when adjacent kept rows are not
a constant number of periods apart. The paths that read a constant spacing
into the grid they are handed are

- the regression HAC tests — `predictive_beta`, `spanning_alpha`, and the
  $K$-restriction Wald / slice period joint tests;
- the ADF and autocorrelation persistence screens;
- event-study estimation windows and offsets;
- adjacent-period metrics — turnover, rank autocorrelation, rolling windows.

The remedy is the caller's: pass `dates=` at a constant stride on the panel's
period grid if those paths must be calibrated; factrix does not resample.
Switching `inference=` is not that remedy — on an uneven grid
`overlap_periods` is the *maximum* overlap, so `NonOverlapping` strides at
that maximum and discards more of the series than it would on a
constant-stride grid, while `NeweyWest`'s $1.3\sqrt{T}$ base bandwidth is
insensitive to the unevenness and keeps the full series; that is sample
efficiency, not calibration.

#### Three departures from the textbook HAR form

Each is a factrix choice, not a published one, kept because it is
load-bearing for measured size on overlapping horizons. The counterpart
statements live in `factrix._stats.hac`'s docstring Notes.

| Piece | Standard | What factrix does | Why | Measured consequence |
|---|---|---|---|---|
| Overlap floor | $h - 1$ (Hansen-Hodrick consistency floor) | $3(h-1)$ | At $L = h-1$ the Bartlett weight sends the lag-$(h{-}1)$ autocovariance to $1/h$, so the kernel discards most of the overlap covariance it exists to capture; at $3(h-1)$ the mean weight across the MA band is ≈0.83 rather than ≈0.5. | $T{=}240$, $h{=}21$: size 10.1% → 6.0% (40000 replications, seed 20260828; re-run at a reduced replication count by `tests/stats/test_overlap_floor_size.py`). Floor moved alone, the other two rows held at what factrix does today. |
| SE scale | none (textbook is $\sqrt{\text{LRV}/T}$); Stata's `newey` scales by $T/(T-k)$ with $k$ = **regressor count**, never the bandwidth | $T/(T - L - 1)$ | Self-derived white-noise demeaning-bias correction: in-sample demeaning biases each $\hat\gamma_j$ by ≈$-\gamma_0/T$, giving $\mathbb{E}[\widehat{\text{LRV}}] \approx \gamma_0(1 - (L+1)/T)$, which this undoes exactly. | $h{=}5$: 8.2–9.6% → 6.2–6.7%. Partly double-counts with the fixed-$b$ reference (whose KV limit already embeds demeaning), so $h{=}1$ iid cells come out slightly *under*-sized (4.3–4.7% at $T \le 120$). |
| Effective df | LLSW's `harreg.ado` uses $\lceil 1.5T/S \rceil$ | $\min(1.5T/L - 1,\; T/h - 1)$ | The $-1$ is a sub-one-df small-sample conservatism. The $T/h - 1$ cap is self-derived: an $h$-period overlapping series carries at most $T/h$ independent observations however the kernel is tuned. | Cap: $T{=}60$, $h{=}21$ size 12.2% → 4.3%. Cost: passing `forward_periods` on a series with *less* dependence than $h$ implies makes the test markedly conservative — measured 0.2% ($T{=}60$), 2.1% ($T{=}120$), 3.5% ($T{=}240$) on iid input at $h{=}21$. |

Producing module: `tests/stats/test_hac_overlap_size.py` re-measures the
three pieces only as one contrast — the current recipe against the
pre-fix recipe that swapped all three at once. The per-piece cells above
are not re-measured by any committed module except where a cell names
one.

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

`factor_return_var` is **required** under `is_estimated_factor=True`;
omitting it raises `UserInputError`. There is no default because the
only readily-available substitute, $\mathrm{var}(\hat\beta_t)$, makes the
multiplier $1 + \overline{\beta}^2 / \mathrm{var}(\hat\beta_t) \equiv
1 + t^2_{\mathrm{iid}} / T$ identically — an algebraic restatement of the
$t$-stat that carries no errors-in-variables information about the
regressor at all. When $\sigma^2_f$ collapses below machine epsilon the
multiplier is undefined; factrix skips the correction, returns the
uncorrected Newey-West $p$, and raises
`WarningCode.DEGENERATE_VARIANCE` rather than switching silently.

The corrected $t$ is read against the **same** effective degrees of
freedom $\nu$ as the uncorrected one (`metadata["hac_dof"]`), so
$c \ge 1$ translates into $p_{\text{Shanken}} \ge p_{\text{uncorrected}}$
on any given series. Reading it against $T - 1$ instead would widen the
reference distribution far enough to more than undo the $\sqrt{c}$ SE
inflation and make the "conservative" correction return the *smaller*
$p$.

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
| **Newey-West (1987)** | Bartlett-kernel HAC on the full overlapping series, bandwidth `L = max(bandwidth_base, h−1)`. | Simple, deterministic, asymptotically valid for arbitrary autocorrelation up to `L`. | Asymptotic Gaussian — finite-sample size distortion when `h/T` is non-trivial; bandwidth rule is conservative. | `ic` / `quantile_spread` / `quantile_spread_vw` / `k_spread` (with `NeweyWest` inference), `fm_beta` stage 2, `common_quantile_spread`, `common_asymmetry`, `spanning_alpha`, the slice Wald tests. |
| **Hansen-Hodrick (HH) (1980)** | A generalized method of moments (GMM)-style HAC estimator with a rectangular kernel truncated at `h−1`; the canonical reference for overlap-aware long-horizon SEs. | Targets the MA(`h−1`) residual structure overlap induces. | Rectangular kernel can yield a non-PSD covariance matrix in finite samples; still asymptotic. | **Research-only.** Implemented as `factrix.inference.series_mean.HANSEN_HODRICK` (rectangular-kernel SE → t-statistic → two-sided p-value) and usable standalone via `HANSEN_HODRICK.compute(...)`, but deliberately in **no** metric's `applicable_inference` allowlist — passing it to a metric raises `IncompatibleInferenceError` — and for that reason not re-exported from `factrix.inference`. Also borrows the `h−1` lag idea as a floor on the NW bandwidth above. |
| **Hodrick (1992) "1B"** | Reverse-regression: regress one-period return on the predictor sum `X_t = Σ x_{t-j}` over the last `h` periods. | Size-correct in finite samples even at large `h/T`; no bandwidth choice. | Coefficient interpretation differs — `β` is the response to a cumulative-predictor stimulus (MA on the RHS) rather than a long-horizon forecast slope (MA on the LHS in the standard form); not a drop-in replacement for the canonical `β`. | **Not implemented**. Cited as the right tool when overlap is severe; the `Individual × Continuous` cell side-steps the issue with non-overlapping resampling instead. |
| **Stationary bootstrap (Politis-Romano 1994)** | Block-resamples the series (geometric block length, Politis-White 2004 automatic selection), centred under `H0`, studentizes both the observed and the resampled root by a batch-means SE at the same block length, and reports the empirical two-sided p from the resampled `t` ratios. | No normality or asymptotic-variance assumption at all — valid when the IC/return distribution is heavy-tailed or skewed enough that NW's / HH's Gaussian p-value is itself suspect. | Heavier to compute (resampling, not closed-form); the reported `stat` is the observed mean rather than the root the p is computed from. On a zero-dispersion sample there is no block SE to divide by, and the kernel falls back to the raw-mean root — reported as `metadata["studentized"] = False` and `degenerate_variance`. | Exposed as `factrix.inference.STATIONARY_BOOTSTRAP` on `ic`, `quantile_spread`, `quantile_spread_vw` and `k_spread` — every one of them dispatches the member polymorphically, and the spread series carries its own measured size table (§6). |

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

### Which metrics take `inference=`, and which take a raw lag knob

Two different surfaces set the HAC bandwidth, and which one a metric
offers is decided by whether that metric's series has a **measured size
table**, not by convenience:

| Metric | Bandwidth surface | Why |
|---|---|---|
| `ic`, `quantile_spread`, `quantile_spread_vw`, `k_spread` | `inference=` (allowlist: `NON_OVERLAPPING`, `NEWEY_WEST`, `STATIONARY_BOOTSTRAP`) | Headline test is "average an overlapping per-date series, test `mean != 0`". The IC series and the spread series each carry their own measured size table (this section and §6), so each member is admitted on the strength of it, and the bandwidth is chosen for the caller by the member. |
| `fm_beta`, `predictive_beta`, `common_quantile_spread`, `common_asymmetry` | raw `newey_west_lags` | Same Newey-West kernel, but no size table has been measured **on these metrics' own series**, so there is nothing to admit an `inference` member against. `None` (the default, and the only calibrated setting) resolves the bandwidth by the standard rule; an explicit integer is an unvetted override for research use. |
| `pooled_beta` | raw `driscoll_kraay_lags` | A different estimator entirely — Driscoll-Kraay is a two-dimensional (period × asset) kernel, not the series-mean HAC, so it is neither an `inference` member nor a `newey_west_lags`. |
| `spanning_alpha`, the slice Wald tests | neither | Bandwidth is fully determined by `overlap_periods` through `_resolve_nw_lags`; there is no caller knob to mis-set. |

The rule for moving a metric from the second row to the first is the same
one that governs the allowlists: measure size on **that metric's own
series** first, then admit. Until then the raw knob is deliberately raw —
it does not pretend to be a vetted choice.

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


### Resampling knobs

Every entry point that turns a resample count into a reported p or
interval takes the same two knobs under the same names, with the same
default and the same refusal floor.

| Entry point | `n_resamples` default | resolved `seed` reported as | Floor enforced | Reports `p_value_mc_se` |
|---|---|---|---|---|
| `ic(inference=StationaryBootstrap(...))` | 999 | `metadata["seed"]` | yes | yes (`metadata`) |
| `monotonicity(...)` | 999 | `metadata["seed"]` | yes | yes (`metadata`) |
| `bootstrap_mean_ci(...)` | 999 | not reported (returns an interval) | yes | no (returns an interval, not a p) |
| `slice_period_pairwise_test` / `slice_period_joint_test` (`method="bootstrap"`) | 999 | the `seed` output column | yes | no (see below) |
| `stationary_bootstrap_resamples(...)` | 999 | not reported (returns draws) | **no** | no (returns draws, not inference) |

The floor is `BOOTSTRAP_RESAMPLES_FLOOR = 199` and a lower count raises
`UserInputError`. A Davison-Hinkley smoothed p lives on the `1/(B+1)`
grid, so `B` should be chosen with `α·(B + 1)` an integer
([Davidson-MacKinnon 2000][davidson-mackinnon-2000]): 199 / 399 / 999 at
the conventional levels. 199 is the smallest such grid point — below it
the p resolves no finer than 0.005 — and the default 999 is
[Politis-White (2004)][politis-white-2004]'s recommendation for two-sided
5% work. `stationary_bootstrap_resamples` sits outside the floor
deliberately: it returns draws and claims no inference on them.

`p_value_mc_se` is `sqrt(p(1-p)/B)`, the binomial SE of the resampling
draw itself — how far the reported p would move on a re-run with a
different seed, not a statistical SE of the estimate. At `B = 999` and
p near 0.05 it is ~0.7pp, so 0.043 and 0.058 are one draw apart. It
shrinks only as `1/sqrt(B)`; no amount of data shrinks it. The period
slice tests do **not** report it: their `p_adj` is a Romano-Wolf
step-down adjustment, not a single binomial draw, so the formula does
not apply.

`rng` takes the same three types everywhere — including
`datasets.make_*`, which reports nothing:

- an **`int`** reproduces the run and is reported back unchanged;
- **`None`** (the default) draws a 32-bit int from system entropy and
  reports it, so an unseeded run is still reproducible after the fact;
- a **`numpy.random.Generator`** is used as-is and *advanced* by the
  call, so two calls on one generator draw differently while two
  generators built from the same seed agree. That is the numpy / scipy
  `rng=` semantics, and it is what a nested or large-scale simulation
  running off one stream needs. The reported seed is then `None`: the
  stream is the caller's, so only the caller can reproduce the draw.

Anything else raises `UserInputError`. The period slice tests carry the
report in a `seed` output column, which is null under
`method="analytic"` (that path draws nothing).

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
assumption about the cross-event distribution. A fourth pooled statistic —
the skewness of the signed event returns — is reported descriptively only;
see [event_skewness](#event-skewness-no-calibrated-test) in section 6.

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

### `event_skewness`: no calibrated test for the third moment
[](){ #event-skewness-no-calibrated-test }

`event_skewness` reports the Fisher skewness of the sampled signed
abnormal returns and **no p-value or test statistic**. It used to publish
D'Agostino's skew-test $z$ and its two-sided $p$ whenever
`n_events >= 20`; that test is withdrawn as uncalibrated. It has no
calibrated pooled form here for two independent reasons, and the
same-period clustering deflation `event_hit_rate` and `event_ic` apply
repairs neither.

The measured null size, at a nominal 5%, 300 replications per cell, base
seed 20260830 + replication. The deflated column routes the $z$ through
the design effect of the per-event standardised cubed deviation
$((x_i - \bar x)/s)^3$, whose mean is $g_1$ — the same helper the sibling
event tests use on their own per-event score. Three nulls:

- **Panel** — `make_event_panel(n_assets=50, post_event_drift_bps=0,
  event_rate=0.02, signal_horizon=5)` through
  `compute_forward_return(forward_periods=5)`, read by
  `event_skewness(overlap_periods=5)`. Events land independently, so an
  event period holds 1.55 events on average.
- **Sign-randomised clustered** — 50 events on each of 20 shared event
  periods, one shared shock per period added to every event's return, and
  each event's factor sign drawn independently per asset.
- **Sign-aligned clustered** — 40 assets, 400 periods, returns
  $r_{t,i} = \delta_t + \varepsilon_{t,i}$ with $\delta_t,
  \varepsilon_{t,i} \sim N(0, 0.01^2)$ compounded into a price series; all
  40 assets fire together on 20 randomly chosen periods, and every event
  on a period carries **one common factor sign** drawn for that period.
  This is the auditor's construction; it is the same shared-shock
  mechanism as the row above with the sign randomisation removed.

| Null | Events per event period | Excess kurtosis of signed CAR | SD of the null $z$ | Size, D'Agostino $z$ | Size, $z$ deflated for clustering |
|---|---|---|---|---|---|
| Panel, 252 periods | 1.55 | +0.63 | 1.31 | **19.0%** | **17.7%** |
| Panel, 504 periods | 1.52 | +0.95 | 1.50 | **23.3%** | **22.7%** |
| Panel, `n_assets=1`, 2000 periods | 1.00 | −0.29 | 0.92 | 3.0% | 3.0% |
| Sign-randomised clustered, 50 events per period | 50.0 | −0.08 | 1.13 | 4.7% | 2.0% |
| Sign-aligned clustered, 40 events per period | 40.0 | −0.09 | 2.04 | **30.3%** | **0.0%** |

**Failure one: non-normal signed CARs, with no clustering needed.**
D'Agostino's test assumes the sample is normal under the null and derives
$\operatorname{Var}(g_1) = 6/n$ from that; leptokurtic input inflates the
true variance of $m_3 / m_2^{3/2}$ above it, and the null $z$ comes out
over-dispersed by exactly the factor the size implies. The two panel rows
break the test at 19.0% and 23.3% while averaging 1.55 events per event
period — near enough no clustering at all — and the deflation moves them
barely, to 17.7% and 22.7%. On this null the within-period correlation of
the cubed-deviation score is near zero by construction: signing by
$\operatorname{sign}(\text{factor})$ enters a shared period shock with an
asset-random sign, so a common shock moves the cubed deviations
symmetrically and leaves no between-period variance for the ICC to find.
The sign-randomised clustered row is the control — maximal clustering,
near-zero excess kurtosis, correctly sized at 4.7%, and over-deflated to
2.0% for its trouble.

**Failure two: same-period shocks with sign-aligned events.** Remove the
sign randomisation and the shared shock no longer cancels. The sign-aligned
row rejects 30.3% at a nominal 5% with excess kurtosis of −0.09 — a pure
dependence failure, with none of the non-normality that drives the panel
rows. Here the deflator does grip (estimated ICC 0.30, mean Kish scale
0.286), but it does not restore size: it drives the rejection rate to
**0.0%**, trading a 6x over-rejection for a test with no power at all.

The two failures pull in opposite directions — a deflator strong enough to
touch the sign-aligned row annihilates it, and one weak enough to leave the
sign-randomised control alone does nothing for the panel rows, which are
not a dependence problem in the first place. factrix therefore has no
calibrated pooled test for the skewness of sampled event returns, and does
not manufacture one by tuning a deflator against whichever null it happens
to be measured on. Read `value` as the descriptive shape of the event
return distribution, and test the direction of the payoff with
`event_hit_rate`, `event_ic`, `caar` or `bmp_z` — all of which stay sized
on the mechanism that breaks hardest here. On the sign-aligned null, same
300 replications: `bmp_z` 7.0% (71.3% with `kolari_pynnonen_adjust=False`,
which is the deflator doing exactly the job it was built for, on a
statistic whose failure *is* a shared mean shift), `event_hit_rate` 7.0%,
`corrado_rank` 6.0%. `tests/stats/test_event_skewness_size.py` re-runs the
252-period and sign-aligned cells at a cut replication count.

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
| `spanning_alpha`, `common_quantile_spread`, `common_asymmetry`, `_ols.py` — single-restriction regression contrast | `_resolve_scalar_wald_hac`: the scalar HAR recipe (bandwidth, $T/(T{-}L{-}1)$ scale, effective $\nu$) | $t_\nu$ / $F_{1,\,\nu}$ | 3.3–8.0% for the two `common_*` metrics on the non-persistent common-factor null across $T \in \{60,120,240\} \times h \in \{1,5,21\}$; 5.7–11.7% for `spanning_alpha` on an overlapping-sum spread null; **7.3–16.3%** on a persistent ($\phi = 0.9$) common factor, flagged — see below |
| `pooled_beta` — pooled OLS slope under a Driscoll-Kraay (1998) sandwich | `driscoll_kraay_lags`: `auto_bartlett` on the period count, applied to the cross-sectionally summed scores | $t_{T_{\text{periods}}}$ | **not measured on its own series.** A two-dimensional (period × asset) kernel, not the series-mean HAC — it shares neither bandwidth rule above, and the rows in this table do not transfer to it. Its short-period regime is gated by `unreliable_se_short_periods` and by a hard floor below which the statistic is withheld |
| The slice Wald tests (`slice_joint_test`, `slice_pairwise_test`) — cluster-mean Wald | `_resolve_nw_lags`: $\max(\text{auto\_bartlett}(T), h-1)$ | $F_{r,\,T-1}$ | **8–9%** for the $K = 5$ joint test on 50–90-period slices, 5–6% for the pairwise contrasts |

Producing modules: `tests/stats/test_hac_overlap_size.py` for the scalar
series-mean rows, `tests/test_stambaugh_bias.py` for the `predictive_beta`
rows, and `tests/test_slice_period_joint_test.py` for the slice-test band
in the last row.

Two of these rows are **known-oversized regimes rather than calibrated
ones**, and are disclosed rather than corrected:

- The slice Wald tests keep the narrow bandwidth deliberately. A wide
  bandwidth on a $K \times K$ HAC matrix read against $\chi^2$ /
  $F$ critical values is exactly the case fixed-$b$ theory says needs
  Kiefer-Vogelsang / LLSW $F$-type critical values — moving them to the HAR
  rule measured *worse* (the $K = 5$ slice test goes from 8–9% to 21% at 50
  periods per slice; re-checked on a second null at 300 replications, seed
  `20260830 + rep`, where the wide floor takes it from 22.3% to 47.0% at 50
  periods per slice and $h = 5$). The narrow rule is the lesser evil until
  fixed-$b$ Wald critical values are implemented. The *single*-restriction
  contrasts are a different statistic and moved the other way — see
  "Single-restriction Wald contrasts" below.
- `predictive_beta` at $h > 1$ is the single-restriction case, so it does
  use the HAR rule; that took it from 10–19% to 7.5–14.5%, not to 5%. The
  excess is present at $\rho = 0$ for every $\phi$ and plain OLS-NW carries
  it too, so it is the overlapping-regression HAC problem rather than
  anything about the Stambaugh correction. Re-measured on the
  independent-regressor null used for the split above (iid $h$-period
  overlapping returns, an AR($\phi$) regressor drawn independently of them;
  300 replications, seed `20260830 + rep`): 9.7 / 6.3 / 5.3% at
  $\phi = 0$ and 15.0 / 15.7 / 12.0% at $\phi = 0.9$ for $h = 5$ and
  $T \in \{60, 120, 240\}$ — the same band, and unchanged by the split,
  because this path already resolved its headline bandwidth through
  `_resolve_har_lags`. Its uncorrected-OLS reference slope, which does read
  `_resolve_nw_lags`, measures 4.0–16.0% at $\phi = 0$ and 7.7–36.7% at
  $\phi = 0.9$ and gets *worse* under a wide floor with no matching
  reference (9.0 → 17.3% at $T = 60$, $h = 5$); it stays on the narrow rule,
  reported as the pre-correction reference it is.

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

### Single-restriction Wald contrasts: the restriction-count split

[](){ #single-restriction-wald }

`common_asymmetry`, `common_quantile_spread` and `spanning_alpha` all test
**one** linear restriction on an ordinary least squares (OLS) fit —
$\beta_{\text{long}} + \beta_{\text{short}} = 0$, $\beta_{\text{top}} =
\beta_{\text{bottom}}$, $\alpha = 0$. They used to share the narrow
`_resolve_nw_lags` rule with the $K$-restriction slice tests on the grounds
that both are *multivariate fits*. The grouping was wrong: what degrades
under a wide kernel is the $K \times K$ matrix inversion, and a rank-one
contrast has none. Measured on the common-factor null below, the narrow rule
left both `common_*` metrics 10–34% oversized at $h > 1$ and **not**
shrinking with $T$.

The null: one AR($\phi$) factor broadcast to 50 assets from a random stream
independent of the prices, `ic_target=0`, so every rejection is a false one.
300 replications per cell, seed `20260830 + rep`; Monte-Carlo standard error
about 1.3pp. Rejection rate at a nominal 5%, narrow rule → shipped rule:

| metric | $\phi$ | $T=60$, $h=1$ | $h=5$ | $h=21$ | $T=120$, $h=1$ | $h=5$ | $h=21$ | $T=240$, $h=1$ | $h=5$ | $h=21$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `common_asymmetry` | 0.0 | 8.0 → 5.7 | 15.3 → 8.0 | 34.0 → 5.7 | 3.3 → 3.3 | 10.7 → 3.3 | 23.7 → 7.7 | 4.0 → 5.0 | 10.0 → 5.7 | 16.0 → 7.7 |
| `common_asymmetry` | 0.9 | 10.3 → 9.0 | 19.3 → 13.0 | 41.0 → 7.3 | 5.0 → 4.3 | 12.7 → 6.7 | 28.3 → 8.7 | 4.0 → 4.3 | 9.7 → 6.0 | 17.3 → 8.3 |
| `common_quantile_spread` | 0.0 | 7.7 → 5.0 | 9.7 → 5.7 | 16.3 → 0.0 | 7.7 → 6.7 | 7.7 → 5.0 | 10.3 → 2.7 | 7.3 → 3.7 | 6.3 → 4.0 | 7.7 → 2.0 |
| `common_quantile_spread` | 0.9 | 13.0 → 14.3 | 21.0 → 16.3 | 42.3 → 9.3 | 9.7 → 7.3 | 18.7 → 14.7 | 30.7 → 11.0 | 7.0 → 5.7 | 12.0 → 7.7 | 13.0 → 8.0 |

Every $\phi = 0$ cell now sits at or below 8.0%, and the $h = 21$ column —
the worst regime under the old rule — is the one the change helps most.

`spanning_alpha`, on a null of two independent AR($\phi$) series summed over
$h$ periods (same replication count and seed stream), moves the same way:
9.0 → 6.0, 18.7 → 9.3 and 46.0 → 11.7% at $T = 60$ and $h \in \{1, 5, 21\}$
for $\phi = 0$, and 50.0 → 29.3, 50.3 → 30.3, 60.0 → 21.7% at $\phi = 0.9$.
That null is deliberately harsh — a spread series that *is* an overlapping
sum of a near-unit-root process — and the $\phi = 0.9$ column is the
persistent regime no path here is calibrated for, not a claim about the
metric on ordinary input. `spanning_alpha` runs the same two screens as the
two `common_*` metrics, on the **regressand**: the candidate spread is the
series whose long-run variance the alpha standard error estimates. Post-fix
size and the share of draws carrying a code, on the same null and grid:

| $\phi$ | | $T=60$, $h=1$ | $h=5$ | $h=21$ | $T=120$, $h=1$ | $h=5$ | $h=21$ | $T=240$, $h=1$ | $h=5$ | $h=21$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | size | 6.0 | 9.3 | 11.7 | 5.7 | 7.7 | 11.7 | 5.7 | 8.0 | 9.7 |
| 0.0 | flagged | 1.3 | 8.3 | 100 | 0.0 | 4.0 | 100 | 0.0 | 0.7 | 6.0 |
| 0.9 | size | 29.3 | 30.3 | 21.7 | 19.0 | 21.3 | 15.3 | 18.0 | 17.0 | 12.7 |
| 0.9 | flagged | 100 | 77.7 | 100 | 100 | 92.0 | 100 | 100 | 100 | 27.3 |

Every $\phi = 0.9$ cell is flagged on at least 27% of draws and most on
100%, against 0–8.3% of the calibrated $\phi = 0$ cells at $h \le 5$. One
cell is genuinely uncovered and disclosed rather than gated: $\phi = 0$,
$T = 240$, $h = 21$ measures 9.7% with 11 independent observations — above
the shortage floor, and with no persistence left after the stride. That is
the residual overlapping-regression HAC excess `predictive_beta` carries at
$h > 1$ for the same reason, and it fires no code of its own on either
path.

**What moving only the overlap floor would have done.** The narrow rule's
$h - 1$ floor is the [Hansen-Hodrick (1980)][hansen-hodrick-1980]
consistency floor, and widening it alone to $3(h-1)$ — the change the
scalar series-mean path made — measures *worse* here, not better:
`common_quantile_spread` at $\phi = 0$ goes 9.7 → 16.0% ($T=60$, $h=5$) and
16.3 → 35.7% ($T=60$, $h=21$); `common_asymmetry` at $\phi = 0.9$ goes
41.0 → 55.7% ($T=60$, $h=21$). A wide Bartlett kernel read against $T-k$
degrees of freedom is exactly the fixed-$b$ case: the bandwidth and the
reference distribution have to move together, which is why
`_resolve_scalar_wald_hac` returns the scale and the effective $\nu$
alongside the lag count rather than a lag count alone.

**The cost.** The fixed-$b$ reference is a genuinely higher hurdle, at
$h = 1$ as much as at $h > 1$: on a $T = 800$ split-slope alternative that
the narrow rule rejected at $p \approx 0.045$, the shipped rule reports
$p \approx 0.080$. That is the LLSW size-power trade the scalar series-mean
path already pays, now paid consistently by every scalar statistic in the
library.

**What it does not fix, and what is flagged instead.** Two regimes survive:

- *A factor persistent beyond the overlap horizon* ($\phi = 0.9$ column
  above, 13.0–16.3% at $T = 60$). Both metrics now run the same lag-1 screen
  on the strided per-period factor that the series-mean members run on their
  tested series, and emit `serial_correlation_detected`. On the null above
  it fires on 100% of the $\phi = 0.9$, $h = 1$ draws and 51–93% of the
  $h = 5$ draws, against 0–8% of the $\phi = 0$ draws.
- *Too few independent periods.* At $h = 21$ the strided sample falls below
  `MIN_SERIES_PERIODS_HARD` (10) for $T \le 240$, so the persistence screen
  withholds itself; the metrics emit `unreliable_se_short_periods` on the
  effective count $T / h$ instead, exactly as `predictive_beta` does. The
  two codes partition the regime rather than overlap.

All three metrics run both screens — the two `common_*` on the per-period
factor, `spanning_alpha` on the candidate spread it regresses.

Neither number was tuned to reach a target. `tests/stats/test_scalar_wald_overlap_size.py`
re-runs the contrast on a cheaper null at a cut replication count.

### Shanken EIV correction on `fm_beta`: measured size

Rejection frequency at a nominal 5% on a true null —
`make_cs_panel(ic_target=0.0)`, 50 assets, 300 replications per cell,
seed 20260830 + replication. $\sigma^2_f$ is the variance of the panel's
own realised long-short forward return (top-minus-bottom quintile,
equal-weighted, computed from the same panel), which is what a caller
holding the traded spread would supply.

| $T$ | $h$ | uncorrected | Shanken |
|---|---|---|---|
| 120 | 1 | 7.3% | 7.3% |
| 120 | 5 | 3.3% | 3.3% |
| 240 | 1 | 4.3% | 4.0% |
| 240 | 5 | 6.0% | 6.0% |

Under the null $\overline{\beta} \to 0$, so $c \to 1$ and the correction is
close to inert: the row simply inherits the scalar series-mean size of the
HAR path in the table above. That is the intended shape — the correction
costs no size on a true null and only shrinks $t$ where a premium is
actually estimated. Producing module:
`tests/stats/test_fm_beta_shanken_size.py`, which re-measures the same
grid at a reduced replication count.

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
### Persistence *beyond the overlap horizon*: no HAC or bootstrap path is calibrated

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
the sample.

#### What the screen reads, and why it is the strided series

factrix raises `WarningCode.SERIAL_CORRELATION_DETECTED` above
`PERSISTENT_SERIES_AUTOCORR` (0.3) so the regime above is never silent.
The number it screens is lag-1 autocorrelation **on the series strided at
`overlap_periods`** — first observation of each block, the same stride
`NON_OVERLAPPING` runs its *t*-test on — not on the full overlapping
series.

The distinction is the whole content of the screen. Overlapping *h*-period
forward returns carry an MA(*h*−1) structure by construction: lag-1
autocorrelation near 1 − 1/*h*, lag-*h* near zero. That is precisely what
the HAC bandwidth floor `3(h − 1)` and the bootstrap block-length floor
`overlap_periods` exist to absorb, and the size tables below show they do.
A lag-1 read on the unstrided series therefore fires on the everyday
overlapping case — where the paths *are* calibrated — and says nothing
about the regime where they are not. Measured on the persistent-factor
null below (300 replications, φ = 0.9), an unstrided lag-1 screen fires on
99–100% of draws at *h* ∈ {5, 21} while the measured size stays at
3.7–9.0%; the strided screen fires on 0–9% of the same draws.

Whether a factor is persistent is not itself the question. A per-asset
AR(φ) factor makes the *IC series* persistent only through the
forward-return overlap: measured lag-1 rises to 0.61–0.81 at *h* ∈ {5, 21}
while lag-*h* stays at −0.02 to −0.11, the MA(*h*−1) signature. What the
screen is for is a series that stays autocorrelated once the overlap is
strided away — a drifting signal level, a slow-moving regime — and there
the stride cannot help and neither can the kernel.

#### The `ic` pipeline: measured size and screen firing rate

True-null rejection rates at a nominal 5% for `ic` under each inference
member, on two nulls. Both are
`make_cs_panel(n_assets=50, ic_target=0.0, signal_horizon=h)` with
`n_dates = T + h`, `compute_forward_return(forward_periods=h)` and
`overlap_periods=h`; they differ only in `factor_persistence` — φ = 0 is
the iid-per-period factor, φ = 0.9 a per-asset AR(0.9) factor independent
of returns, which is what makes the overlap mechanism bite at all. 300
replications per cell, seed `20260830 + rep`, `n_resamples=499`.
Monte-Carlo standard error is ~1.3pp per cell. `T` counts periods on the
evaluation grid before the stride. "screen" is the fraction of draws
raising `serial_correlation_detected`; it is a property of the series, so
it is identical across the three members.

Null: **iid factor** (`factor_persistence=0.0`)

| T | h | `NON_OVERLAPPING` | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` | screen |
|---|---|---|---|---|---|
| 60 | 1 | 0.047 | 0.057 | 0.057 | 0.003 |
| 60 | 5 | 0.043 | 0.060 | 0.080 | 0.047 |
| 60 | 21 | — | 0.003 | 0.053 | 0.000 |
| 120 | 1 | 0.050 | 0.040 | 0.053 | 0.000 |
| 120 | 5 | 0.060 | 0.030 | 0.043 | 0.053 |
| 120 | 21 | — | 0.023 | 0.100 | 0.000 |
| 240 | 1 | 0.053 | 0.053 | 0.060 | 0.000 |
| 240 | 5 | 0.053 | 0.063 | 0.067 | 0.010 |
| 240 | 21 | 0.033 | 0.027 | 0.043 | 0.060 |

Null: **persistent factor** (`factor_persistence=0.9`)

| T | h | `NON_OVERLAPPING` | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` | screen |
|---|---|---|---|---|---|
| 60 | 1 | 0.020 | 0.037 | 0.030 | 0.000 |
| 60 | 5 | 0.040 | 0.060 | 0.083 | 0.067 |
| 60 | 21 | — | 0.010 | 0.040 | 0.000 |
| 120 | 1 | 0.063 | 0.073 | 0.063 | 0.000 |
| 120 | 5 | 0.073 | 0.063 | 0.067 | 0.030 |
| 120 | 21 | — | 0.067 | 0.077 | 0.000 |
| 240 | 1 | 0.060 | 0.053 | 0.060 | 0.000 |
| 240 | 5 | 0.080 | 0.087 | 0.090 | 0.020 |
| 240 | 21 | 0.057 | 0.060 | 0.070 | 0.080 |

`NON_OVERLAPPING` at `T ∈ {60, 120}, h = 21` has no rejection rate: `ic`
refuses the panel at its stride-scaled periods floor (`metric_unavailable`)
rather than testing ~3 effective periods.

Both nulls read the same way, which is the point: the persistent factor
moves no member outside the band the iid factor already sits in
(`NEWEY_WEST` 0.3–8.7%, `STATIONARY_BOOTSTRAP` 3.0–10.0%), and the screen
stays quiet on both. The `NEWEY_WEST` floor cells (0.3% at `T = 60,
h = 21`, 2.3% at `T = 120, h = 21`) are the documented long-horizon
conservatism of the HAR effective df `T/h − 1`, not a persistence effect —
they are as low on the iid null as on the persistent one.

The screen is **withheld** below `MIN_SERIES_PERIODS_HARD` (10) strided
observations — the library's floor for estimating a series statistic on
the periods axis, the same one `NON_OVERLAPPING`'s post-stride sample is
gated on. A lag-1 autocorrelation read off three to nine observations is
noise: the estimator's standard error there is 0.3–0.6 and a sample value
above `PERSISTENT_SERIES_AUTOCORR` is common under independence, so the
code would be reporting the shortage of periods rather than any
persistence, which is `UNRELIABLE_SE_SHORT_PERIODS`'s job. That is why
the `h = 21` screen column reads 0.000 at `T ∈ {60, 120}` on both nulls:
`T/h` is 3–6 observations there and nothing is estimated.

`T = 240, h = 21` clears the floor at 12 strided observations and still
fires on 6.0% (iid factor) / 8.0% (persistent factor) of draws. That is
the residual small-sample noise of a lag-1 estimate at *n* = 12, where the
estimator's own standard error is ~0.29 against a 0.3 threshold — equal on
both nulls, so it is not persistence. Read a
`serial_correlation_detected` on a strided series near the floor as weak
evidence and check `n_obs`.

#### A series that really is persistent beyond the horizon

Constructed directly rather than through the overlap: a per-period
factor→return signal level following a zero-mean AR(φ) across the grid, 50
assets, `overlap_periods=1`, so the IC series is AR(φ) with no overlap
component at all. 300 replications, seed `20260830 + rep`, nominal 5%.

| φ | T | plain *t* (`NON_OVERLAPPING`) | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` | screen |
|---|---|---|---|---|---|
| 0.6 | 60 | 0.340 | 0.103 | 0.153 | 0.947 |
| 0.6 | 120 | 0.267 | 0.067 | 0.067 | 1.000 |
| 0.6 | 240 | 0.347 | 0.073 | 0.073 | 1.000 |
| 0.9 | 60 | 0.653 | 0.273 | 0.213 | 0.997 |
| 0.9 | 120 | 0.623 | 0.193 | 0.143 | 1.000 |
| 0.9 | 240 | 0.630 | 0.143 | 0.097 | 1.000 |

This is the regime the code names, and the screen fires on 94.7–100% of
draws in it. Note that lengthening the sample does not fix the plain *t*
here — 34.7% at `T = 240` — because the persistence, not the sample size,
is what the OLS SE is missing.

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

### Stationary bootstrap on the spread series: measured size

`ic`'s size table is a table of the *IC* series. A long-short spread
series is a different object — a cross-sectional bucket difference, not a
rank correlation — so admitting `STATIONARY_BOOTSTRAP` to
`quantile_spread` / `quantile_spread_vw` / `k_spread` needed its own
measurement rather than an inherited one.

True-null rejection rates at a nominal 5% on `quantile_spread`, measured
on `make_cs_panel(n_assets=50, ic_target=0.0)` with
`compute_forward_return(forward_periods=h)` and `overlap_periods=h`. `T`
counts periods on the evaluation grid before the stride. Each table
states its null; both use base seed 20260830 and `n_resamples=499`.

Null: **iid factor** (`factor_persistence=0.0`, the default) — 500
replications per cell, Monte-Carlo standard error ~1.0pp.

| T | h | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` |
|---|---|---|---|
| 60 | 1 | 0.074 | 0.082 |
| 60 | 5 | 0.036 | 0.066 |
| 60 | 21 | — | — |
| 120 | 1 | 0.056 | 0.052 |
| 120 | 5 | 0.054 | 0.072 |
| 120 | 21 | 0.024 | 0.090 |
| 240 | 1 | 0.046 | 0.048 |
| 240 | 5 | 0.076 | 0.080 |
| 240 | 21 | 0.036 | 0.068 |

Null: **persistent factor** (`factor_persistence=0.9`, per-asset AR(0.9)
independent of returns) — 300 replications per cell, seed
`20260830 + rep`, Monte-Carlo standard error ~1.3pp. This is the null
under which the forward-return overlap actually shows up in the spread
series: with an iid factor, consecutive periods' spreads share return
windows but draw independent weights, so the overlap barely propagates.
"screen" is the fraction of draws raising `serial_correlation_detected`.

| T | h | `NEWEY_WEST` | `STATIONARY_BOOTSTRAP` | screen |
|---|---|---|---|---|
| 60 | 1 | 0.027 | 0.037 | 0.003 |
| 60 | 5 | 0.083 | 0.083 | 0.063 |
| 60 | 21 | — | — | — |
| 120 | 1 | 0.060 | 0.057 | 0.003 |
| 120 | 5 | 0.067 | 0.090 | 0.023 |
| 120 | 21 | 0.077 | 0.100 | 0.000 |
| 240 | 1 | 0.047 | 0.050 | 0.000 |
| 240 | 5 | 0.083 | 0.080 | 0.003 |
| 240 | 21 | 0.043 | 0.067 | 0.047 |

`T = 60, h = 21` is not measurable on either null: the metric refuses that
panel at its stride-scaled periods floor (`metric_unavailable`) rather
than testing ~2 effective periods, so there is no rejection rate to
report.

The persistent-factor null moves nothing outside the iid-factor bands
(`NEWEY_WEST` 2.7–8.3% against 2.4–7.6%, `STATIONARY_BOOTSTRAP`
3.7–10.0% against 4.8–9.0%; every difference is inside two Monte-Carlo
standard errors), and the screen stays quiet — the 4.7–7.3% at `h = 21` is
the small-sample false-positive floor of a 3–11-observation strided
sample, present on the iid null too.

The bootstrap is calibrated-to-slightly-liberal across every measurable
cell (4.8–9.0%) and never worse than the short-sample distortion already
documented for the *t*. `NEWEY_WEST` is tighter at `h = 1` and
conservative at long horizons (2.4–3.6% at `h = 21`), where the HAR
effective df `T/h - 1` spends most of the sample. Neither dominates; both
are reported, and the choice is the routing question the next section
answers. A reduced-replication re-run of two cells guards the numbers in
`tests/stats/test_spread_bootstrap_size.py`.

### `monotonicity`: measured size of the Patton-Timmermann MR bootstrap

True-null rejection rates at a nominal 5% for the MR statistic's
stationary-bootstrap p. Null: `make_cs_panel(n_assets=50, ic_target=0.0,
factor_persistence=φ)` → `compute_forward_return(forward_periods=h)`,
`overlap_periods=h`, `n_groups=5`, `n_resamples=499`. `T` counts periods
on the evaluation grid before the stride. 300 replications per cell, seed
`20260830 + rep`, Monte-Carlo standard error ~1.3pp.

| T | h | φ = 0 | φ = 0.9 |
|---|---|---|---|
| 120 | 1 | 0.037 | 0.033 |
| 120 | 5 | 0.020 | 0.027 |
| 240 | 1 | 0.073 | 0.060 |
| 240 | 5 | 0.050 | 0.033 |

Calibrated-to-conservative throughout (2.0–7.3%), and the persistent
factor moves nothing outside the iid band — every difference is inside two
Monte-Carlo standard errors. The conservatism at `h = 5` is the expected
cost of the block bootstrap's block-length floor at `overlap_periods`
absorbing the MA(*h*−1) overlap. A reduced-replication re-run guards the
numbers in `tests/stats/test_monotonicity_size.py`.

### `directional_hit_rate`: measured size of the Pesaran-Timmermann test

True-null rejection rates at a nominal 5% for the PT statistic $S_n$ with
the Kolari-Pynnönen within-period deflation applied (it fires on every
cell of this grid — the null panel is a cross-section, so the pooled
`(date, asset)` trials are not independent). Same null and seed scheme as
the table above; 300 replications per cell, Monte-Carlo standard error
~1.3pp.

| T | h | φ = 0 | φ = 0.9 |
|---|---|---|---|
| 120 | 1 | 0.043 | 0.053 |
| 120 | 5 | 0.030 | 0.047 |
| 240 | 1 | 0.040 | 0.067 |
| 240 | 5 | 0.060 | 0.077 |

Calibrated across the grid (3.0–7.7%). The persistent-factor column runs
slightly hotter than the iid one at `T = 240` (6.7% and 7.7% against 4.0%
and 6.0%), but the gap is within two Monte-Carlo standard errors and the
worst cell is still inside the 7–9% short-sample band every other
per-period path in this document sits in. The deflation is what keeps it
there: the raw pooled $S_n$ treats 50 same-date names as 50 independent
trials. A reduced-replication re-run guards the numbers in
`tests/stats/test_directional_hit_rate_size.py`.

### `positive_rate`: measured size of the exact binomial test, and its discreteness

True-null rejection rates at a nominal 5% for the two-sided exact
binomial test against `H0: p = 0.5`. Two nulls, both at
`overlap_periods=1` so `n` is the series length directly; 300
replications per cell, seed `20260830 + rep`, Monte-Carlo standard error
~1.3pp.

- **iid Gaussian series** — `value ~ N(0, 1)`, the textbook null the exact
  test is derived against.
- **IC-series pipeline** — the per-period Spearman IC of
  `make_cs_panel(n_assets=50, ic_target=0.0)` at `forward_periods=1`, i.e.
  the series a caller actually feeds it after `compute_ic`.

| n | iid Gaussian series | IC-series pipeline |
|---|---|---|
| 60 | 0.027 | 0.023 |
| 120 | 0.030 | 0.040 |
| 240 | 0.047 | 0.080 |

Both columns sit at or below nominal, which is the *expected* behaviour
and not a defect: the binomial distribution is discrete, so at most `n`
the attainable two-sided level nearest 5% from below is materially under
it — the exact test spends whatever is left over as conservatism. That is
the deliberate trade recorded in the function's docstring (the
normal-approximation `z` attains 5% by *over*-rejecting: `n=20, 15 hits`
gives `p=0.025` against the exact `0.041`). Read a `positive_rate` p as a
conservative bound, and do not calibrate a power study against a nominal
5% at short `n`. The IC-series column's 8.0% at `n = 240` is the one cell
above nominal and is inside two Monte-Carlo standard errors of the iid
column. A reduced-replication re-run guards the numbers in
`tests/stats/test_positive_rate_size.py`.

### Inference paths still without a size table

Every other `p_value` factrix publishes has a measured size somewhere in
this section. These two do not, and are listed here rather than left
silent so the gap is a documented one. A test
(`tests/stats/test_section6_covers_every_p_value_path.py`) enumerates the
metrics whose `MetricResult` can carry a non-`None` `p_value` and fails if
any is missing from this section, so a new inference path cannot be added
without either measuring it or naming it here.

**`common_asymmetry` and `common_quantile_spread` — measured, withheld at
`overlap_periods > 1`.** A COMMON-scope null found both Wald p-values
over-rejecting at `h > 1` in a way that does *not* shrink with `T`. The
cause is the bandwidth `_resolve_nw_lags` gives the Wald family: its
overlap floor is `overlap_periods - 1`, against the `3(h - 1)` the scalar
series-mean HAR path uses (`_resolve_har_lags`), so the MA(*h*−1)
structure an overlapping forward return carries by construction is
absorbed at the minimum admissible bandwidth. The numbers are withheld
until that floor is settled, because tabling them would record a defect as
a calibration. Until then read both metrics' `p_value` at
`overlap_periods > 1` as not calibrated — prefer `h = 1`, or read `value`
and the bucket / slope detail in `metadata` descriptively.

### `common_beta`: measured size of the calendar-time cross-asset $t$

True-null rejection rates at a nominal 5% for `common_beta`'s
cross-asset $t$ on $\mathbb{E}[\beta]$, on the COMMON-scope null the
metric's cell describes: one AR($\varphi$) factor series broadcast to
every asset, drawn from an RNG stream independent of the
`make_cs_panel(n_assets=50, ic_target=0.0)` prices, so every true
per-asset $\beta$ is zero while the assets keep the cross-sectional
return correlation the panel generates. 300 replications per cell, seed
`20260830 + rep` (factor stream `20260830 + 10000 + rep`), Monte-Carlo
standard error ~1.3pp. `T` counts periods on the evaluation grid and `h`
is `forward_periods`, passed through as `overlap_periods`.

| T | h | φ = 0 | φ = 0.9 |
|---|---|---|---|
| 60 | 1 | 0.067 | 0.057 |
| 60 | 5 | 0.073 | 0.033 |
| 120 | 1 | 0.060 | 0.067 |
| 120 | 5 | 0.067 | 0.027 |
| 240 | 1 | 0.053 | 0.047 |
| 240 | 5 | 0.053 | 0.020 |

Calibrated across the grid (2.0–7.3%), with the worst cells at the short
end where the Newey-West variance $V_{\mathrm{EW}}$ of the equal-weight
portfolio slope rests on fewest periods. The calendar-time SE is what
holds it there: the iid cross-asset $t$ the docstring keeps as
`metadata["stat_uncorrected"]` is the one this null breaks, because 50
assets loading on one shared regressor do not supply 50 independent
betas. The persistent-factor column is the *conservative* direction at
`h = 5` (3.3 / 2.7 / 2.0% against 7.3 / 6.7 / 5.3%) — a φ = 0.9 regressor
read through an overlapping forward return leaves more serial dependence
for the Newey-West kernel to pick up, so the SE widens. A
reduced-replication re-run guards the numbers in
`tests/stats/test_common_beta_size.py`. This is the panel-null companion
to the synthetic-regime measurements in the function's own docstring
(equicorrelated returns at fixed $T$, varying $N$ and $\rho$); neither
covers a hand-built beta table, which falls back to the iid $t$ and says
so in `metadata["calendar_time_se_applied"]`.

### `ic_trend`: measured size of the Mann-Kendall trend test

True-null rejection rates at a nominal 5% for `ic_trend`. The series is
the per-period Spearman IC of `make_cs_panel(n_assets=50,
ic_target=0.0)` through `compute_ic`, so the IC has no time trend to
find; `factor_persistence` supplies the second null. The test runs on the
non-overlapping subsample at stride `overlap_periods`, so the `h = 5`
rows test roughly `T / 5` observations. 300 replications per cell, seed
`20260830 + rep`, Monte-Carlo standard error ~1.3pp.

| T | h | φ = 0 | φ = 0.9 |
|---|---|---|---|
| 60 | 1 | 0.040 | 0.030 |
| 60 | 5 | 0.017 | 0.047 |
| 120 | 1 | 0.043 | 0.050 |
| 120 | 5 | 0.033 | 0.053 |
| 240 | 1 | 0.017 | 0.047 |
| 240 | 5 | 0.033 | 0.047 |

Calibrated-to-conservative throughout (1.7–5.3%), and unlike the
per-period mean paths in §1 the persistent factor moves nothing outside
the iid band: factor persistence carries into the *level* of the
per-period IC, not into a drift in it, and the Hamed-Rao variance
correction absorbs the serial dependence the IC series does carry. Two
things the table does not license. It is measured with the ADF gate at
its default `adf_threshold=0.10`, on a null that is trend-free but also
stationary — a unit-root input is the regime the gate exists for
(`PERSISTENT_REGRESSOR`), and no size here speaks to it. And it is a
*size* table only: the Theil-Sen slope's robustness is what recommends it
over OLS, not a power advantage, and nothing above measures power. A
reduced-replication re-run guards the numbers in
`tests/stats/test_ic_trend_size.py`.

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
| Overlapping `forward_periods` panel, per-period series not persistent once strided (the everyday case) | No warning | `NEWEY_WEST` 5–9% on real overlapping IC (h = 5, n = 240 / 480: 5.2% / 8.8%); `NON_OVERLAPPING` calibrated. On the persistent-factor null, where the overlap actually propagates into the per-period series, `ic` measures NW 1–8.7% and the bootstrap 3–9% across `T × h`, and the screen fires on 0–9% of draws (tables above). | Keep the default member. |
| Persistent per-period series (lag-1 φ ≥ 0.3 on the *strided* tested series) | `serial_correlation_detected` | No path is calibrated. On an AR(φ) per-period IC series at `h = 1`: plain *t* 26.7–34.7%, NW 6.7–10.3%, bootstrap 6.7–15.3% at φ = 0.6, and 62.3–65.3% / 14.3–27.3% / 9.7–21.3% at φ = 0.9 (table above); the screen fires on 94.7–100% of those draws. | Do **not** switch member; it moves the number without fixing it. Read the *t* against a raised hurdle (*t* > 3) or lengthen the sample — at φ = 0.6 the plain *t* is still 34.7% at `T = 240`, so more periods alone is not enough either. A coarser `overlap_periods` stride under `NON_OVERLAPPING` helps mechanically where the horizon allows it — the strided sample sits at φ^h, and AR(0.6) strided at h = 21 measures 4.5%. |
| Fewer than `MIN_SERIES_PERIODS_HARD` (10) periods after the stride | No `serial_correlation_detected` — the screen is withheld | A lag-1 autocorrelation estimated from 3–9 observations is noise, so the screen reports nothing rather than report the shortage of periods as persistence; measured firing rate at `h = 21, T ∈ {60, 120}` is 0.000 on both nulls. Just above the floor it is still weak: 12 strided observations at `T = 240, h = 21` fire on 6.0% / 8.0% of draws, equally on the iid and persistent nulls. | Read `unreliable_se_short_periods` and `n_obs` — the periods shortage, not the persistence, is what the sample is telling you. Lengthen the history or shorten the horizon. |
| Short strided series (fewer than ~120 periods after the stride) | `unreliable_se_short_periods`, or nothing on a moderately short series | The *t* / NW branch runs 7–9%. A block-bootstrap p is *worse* here — the spread metrics' former automatic bootstrap branch (`_block_bootstrap_diff_p`, kernel-isolated on iid input) measured 13.6% at n = 12, 9.8% at 30, 7.4% at 60, reaching 5.2% only by 120 — and the strided series is short exactly when the horizon is long. The spread-series table above says the same at the short end: 8.2% at T = 60, h = 1. | Stay on the analytic *t* / NW. Do not reach for the bootstrap to rescue a short sample; shorten the stride or lengthen the history instead. |
| Long strided series (≥ ~120 periods) whose distribution is in doubt (heavy tails, skew) | No warning | On iid input `STATIONARY_BOOTSTRAP` sits at the same 7–9% baseline as NW (φ = 0 row above) — it buys no *size*. On the spread series it measures 4.8–9.0% across the grid above, with its worst cell at the long horizon (9.0% at T = 120, h = 21) where NW is conservative instead (2.4%). Its case is distributional: it is the only member that does not assume asymptotic normality of the mean (§1). | Read `STATIONARY_BOOTSTRAP` alongside the analytic p — on `ic` and on all three spread metrics, which admit it on the strength of the table above — when tails or skew are the doubt. A documented option, not a default. |
| Heavy-tailed *and* short | `unreliable_se_short_periods` | The *t* is size-robust to tails — 3–4% on t(3) input, i.e. conservative — while the small-n bootstrap is not (6–14%). | Keep the *t*. Tails are not a reason to bootstrap a short series. |
| Thin cross-section (few names per leg) | `few_assets` | Each leg mean rests on a handful of names: a noisier estimate, not a differently-distributed one. The automatic bootstrap switch this once triggered rejected 8–20% against the *t*'s 7–9% and keyed on the wrong axis; it was removed. | Keep the requested member; read `n_assets` and treat the spread as fragile. |
| Overlapping panel scored by `common_asymmetry` / `common_quantile_spread`, per-period factor persistent once strided | `serial_correlation_detected` | The single-restriction HAR reference takes both metrics to 3.3–8.0% on a non-persistent common factor across `T × h`, but a φ = 0.9 common factor leaves them at 13.0% / 16.3% at `T = 60, h = 5`, clearing by `T = 240` (6.0% / 7.7%). The screen fires on 51–100% of those draws and on 0–8% of the calibrated ones. | Read the p against a raised hurdle or lengthen the sample. Changing `newey_west_lags` does not fix it — the bandwidth is not what is wrong. |
| `common_asymmetry` / `common_quantile_spread` with fewer than 10 periods after the `overlap_periods` stride | `unreliable_se_short_periods` | The effective sample, not the raw period count, is what the HAC standard error rests on: at `T = 120, h = 21` five independent observations carry a bandwidth-40 kernel. The persistence screen withholds itself there for the same reason. | Shorten the horizon or lengthen the history; the p carries little information at that effective count. |
| Joint test on K ≥ 3 short slices | `slice_period_joint_test` warning | 8–9% for K = 5 on 50–90-period slices, converging by T ≈ 150; the bootstrap path inherits it (12%). K = 2 is calibrated throughout. | Read the pairwise contrasts on the same slices (5–6%) rather than the joint p, or lengthen the slices. |
| Few event periods after the stride | `few_events` | Power-thin, not size-inflated for `caar` / `corrado_rank`; `bmp_z` is ~10% at 8 effective periods, ~7% at 15, clearing by ~30. | Read borderline p-values cautiously; extend the event history rather than switching estimator — all three count the same event periods. |
| 3–19 portfolio periods in `top_concentration` | `borderline_portfolio_periods` | `top_concentration` publishes no p at any sample size — the withdrawn one-sided `t` against `ratio ≥ 0.5` never rejected, because the null diversification ratio is ~0.91 (0 of 300 draws at both 10 and 48 tested periods). The code is about the precision of the mean, not a test. | Read `value` and `ratio_eff_to_total` as a noisy mean over few periods; lengthen the history before comparing panels. |
| Overlapping panel read through `common_asymmetry` / `common_quantile_spread` at `overlap_periods > 1` | No warning | Not tabled: the Wald family's HAC bandwidth floor is `overlap_periods - 1`, not the `3(h - 1)` the scalar HAR path uses, and the measured over-rejection does not shrink with `T` (section above). | Read `value` descriptively, or read the p at `h = 1` only, until the bandwidth floor is settled. |

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

