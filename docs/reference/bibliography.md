---
title: Bibliography
---

Paper SSOT for factrix. Every reference cited from a metric docstring,
the [Statistical methods](statistical-methods.md) reference page, or
the design-notes page in Development lives here once,
with an explicit anchor that
[mkdocs-autorefs](https://mkdocstrings.github.io/autorefs/) resolves
into reference-style links such as `[Newey-West 1987][newey-west-1987]`.

Sections are organised by methodological role rather than chronology.
Within a section, ordering follows topical relevance to the
implementation rather than alphabetical author order.

---

## Time-series regression and heteroskedasticity-and-autocorrelation-consistent (HAC) inference

### Newey & West (1987)
[](){ #newey-west-1987 }

Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55(3), 703–708.

Bartlett-kernel HAC variance estimator; underlies every Newey-West (NW) HAC t-test
in factrix.

### Newey & West (1994)
[](){ #newey-west-1994 }

Newey, W. K. & West, K. D. (1994). "Automatic Lag Selection in
Covariance Matrix Estimation." *Review of Economic Studies* 61(4),
631–653.

Data-adaptive plug-in bandwidth selection. Cited as background;
factrix uses the simpler Andrews (1991) Bartlett growth rate
$\lfloor T^{1/3} \rfloor$ floored against the Hansen-Hodrick overlap rule.

### Andrews (1991)
[](){ #andrews-1991 }

Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation
Consistent Covariance Matrix Estimation." *Econometrica* 59(3),
817–858.

Optimal Bartlett growth rate $T^{1/3}$; the basis of factrix's default
NW lag rule.

### Andrews & Monahan (1992)
[](){ #andrews-monahan-1992 }

Andrews, D. W. K. & Monahan, J. C. (1992). "An Improved
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix
Estimator." *Econometrica* 60(4), 953–966.

Prewhitening refinement to NW HAC; cited as background, not
implemented (factrix keeps the unprewhitened Bartlett kernel for
deterministic lag selection).

### Hansen & Hodrick (1980)
[](){ #hansen-hodrick-1980 }

Hansen, L. P. & Hodrick, R. J. (1980). "Forward Exchange Rates as
Optimal Predictors of Future Spot Rates: An Econometric Analysis."
*Journal of Political Economy* 88(5), 829–853.

K-period forecast residuals carry MA(K−1) structure under
non-overlapping innovations. Underpins two pieces of factrix's HAC
machinery: (i) a standalone rectangular-kernel HAC SE estimator on
sample means, available to callers who prefer an overlap-targeted
rectangular kernel over the Bartlett kernel; (ii) the `h−1` lag
floor that factrix combines with the Andrews rule when running NW
HAC under overlapping forward returns.

### Hodrick (1992)
[](){ #hodrick-1992 }

Hodrick, R. J. (1992). "Dividend Yields and Expected Stock Returns:
Alternative Procedures for Inference and Measurement." *Review of
Financial Studies* 5(3), 357–386.

Reverse-regression *t*-statistic for long-horizon return
predictability with persistent regressors. The "1B" form solves the
size distortion of NW / Hansen-Hodrick (1980) under heavy overlap
(`h / T` not small) by re-expressing the long-horizon regression as
a one-period regression on a moving-average of the predictor — the
test statistic is then size-correct in finite samples even when the
implied MA(h−1) overlap is severe.

Cited as background, not implemented. Hodrick 1B reformulates the
long-horizon regression as a one-period regression of `r_{t,t+1}` on
the predictor sum `X_t = Σ_{j=0}^{h-1} x_{t-j}`, swapping which side
carries the moving average. The coefficient has a different
interpretation than the standard `r_{t,t+h} ~ x_t` slope, and factrix
prefers non-overlapping resampling on the `Individual × Continuous`
cell as the literature-standard mitigation for overlap-driven size
distortion. See
[Statistical methods § HAC SE](statistical-methods.md#1-hac-se-under-overlapping-returns)
for the comparison among NW / HH-1980 / Hodrick-1992.

### White (1980)
[](){ #white-1980 }

White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix
Estimator and a Direct Test for Heteroskedasticity." *Econometrica*
48(4), 817–838.

HC0 sandwich estimator; the heteroskedasticity-only ancestor of NW
HAC. Cited as background.

### Hansen (1982)
[](){ #hansen-1982 }

Hansen, L. P. (1982). "Large Sample Properties of Generalized Method
of Moments Estimators." *Econometrica* 50(4), 1029–1054.

GMM framework and the over-identifying-restrictions J-statistic;
underlies factrix's two-step efficient GMM moment-estimator path.

---

## Cross-section and panel pricing

### Treynor & Black (1973)
[](){ #treynor-black-1973 }

Treynor, J. L. & Black, F. (1973). "How to Use Security Analysis to
Improve Portfolio Selection." *Journal of Business* 46(1), 66–86.

Original derivation of the alpha-over-residual-risk appraisal-ratio
optimisation for active management — the single-asset appraisal
ratio that [Grinold (1989)][grinold-1989] later generalised across
many independent bets to derive the
$\mathrm{IR} \approx \mathrm{IC} \times \sqrt{\mathrm{breadth}}$
identity. Conceptual ancestor of the breadth decomposition; the
identity itself is Grinold's, not Treynor-Black's.

### Grinold (1989)
[](){ #grinold-1989 }

Grinold, R. C. (1989). "The Fundamental Law of Active Management."
*Journal of Portfolio Management* 15(3), 30–37.

$\mathrm{IR} \approx \mathrm{IC} \times \sqrt{\mathrm{breadth}}$; motivates information coefficient (IC) as the canonical signal-quality
measure and IR/ICIR as its time-stability normalisation.

### Grinold & Kahn (2000)
[](){ #grinold-kahn-2000 }

Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management: A
Quantitative Approach for Producing Superior Returns and Controlling
Risk* (2nd ed.). McGraw-Hill.

Textbook treatment of IC, IR, breadth, and active-management
diagnostics that factrix implements at the metric layer.

### Fama & MacBeth (1973)
[](){ #fama-macbeth-1973 }

Fama, E. F. & MacBeth, J. D. (1973). "Risk, Return, and Equilibrium:
Empirical Tests." *Journal of Political Economy* 81(3), 607–636.

Two-stage λ procedure: per-date cross-sectional regression then
time-series t-test on E[λ]. The FM cell uses this with NW HAC at
stage 2.

### Black, Jensen & Scholes (1972)
[](){ #black-jensen-scholes-1972 }

Black, F., Jensen, M. C. & Scholes, M. (1972). "The Capital Asset
Pricing Model: Some Empirical Tests." In Jensen, M. (ed.), *Studies
in the Theory of Capital Markets*. Praeger.

Beta-sorted-portfolio time-series test of the zero-beta CAPM:
sort assets into beta-ranked portfolios, then run a time-series
regression of each portfolio's excess return on the market. The
contribution is the time-series-then-cross-section aggregation order
(per-asset / per-portfolio time series first, then cross-asset
inspection) that factrix's `common_continuous` cell adopts; factrix's
cross-asset $t$-test on the mean of per-asset β is a simplified
analogue of this aggregation order rather than a replication of BJS's
grouped-portfolio intercept test.

### Petersen (2009)
[](){ #petersen-2009 }

Petersen, M. A. (2009). "Estimating Standard Errors in Finance Panel
Data Sets: Comparing Approaches." *Review of Financial Studies* 22(1),
435–480.

Comparison of FM, clustered, and two-way SE under firm/time
correlation; supports FM + Newey-West as the default for time-effect
panels and motivates the two-way (date+asset) clustered SE option in
factrix's FM pooled-OLS path.

### Cameron, Gelbach & Miller (2011)
[](){ #cameron-gelbach-miller-2011 }

Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). "Robust
Inference With Multiway Clustering." *Journal of Business & Economic
Statistics* 29(2), 238–249.

Two-way clustering formula `V_AB = V_A + V_B − V_{A∩B}`; backs the
date+asset two-way clustered SE option in factrix's FM pooled-OLS
path and the standalone two-way-cluster Wald estimator.

### Thompson (2011)
[](){ #thompson-2011 }

Thompson, S. B. (2011). "Simple Formulas for Standard Errors that
Cluster by Both Firm and Time." *Journal of Financial Economics*
99(1), 1–10.

Finite-sample correction `df = min(G_A, G_B) − 1` used in factrix's
two-way clustered SE path.

### Shanken (1992)
[](){ #shanken-1992 }

Shanken, J. (1992). "On the Estimation of Beta-Pricing Models."
*Review of Financial Studies* 5(1), 1–33.

Errors-in-variables correction for the Fama-MacBeth stage-2 $t$-stat
when the stage-1 regressor is itself an estimated quantity (rolling
$\beta$, PCA score, ML predictor). Applied to the FM stage-2 SE in
factrix's FM path when the stage-1 regressor is itself estimated.
The general multi-factor multiplicative term
$1 + \lambda'\Sigma_f^{-1}\lambda$ collapses to
$1 + \hat\lambda^2/\sigma^2_f$ in the single-factor case factrix
implements; factrix's simplification additionally drops the full
variance's additive $+\sigma^2_f/T$ term and is therefore honest only
for large $T$.

### Kan & Zhang (1999)
[](){ #kan-zhang-1999 }

Kan, R. & Zhang, C. (1999). "Two-Pass Tests of Asset Pricing Models
with Useless Factors." *Journal of Finance* 54(1), 203–235.

Useless-factor diagnostics for two-pass cross-sectional tests: weak
or unidentified factors yield spuriously significant Fama-MacBeth
$t$-stats on risk premia even when the factor has no true pricing
power. Cautionary background on factor validity in factrix's
Fama-MacBeth path, separate from the errors-in-variables
sampling-error correction (which is the
[Shanken (1992)][shanken-1992] single-factor case that factrix's
estimated-factor correction implements).

### Fama & French (1992)
[](){ #fama-french-1992 }

Fama, E. F. & French, K. R. (1992). "The Cross-Section of Expected
Stock Returns." *Journal of Finance* 47(2), 427–465.

Empirical anchor for size and value as cross-sectional risk premia;
prototypical `Individual × Continuous` factor study in the
Fama-MacBeth tradition. Catalog-only reference in factrix; no inline
citation site at present.

### Fama & French (1993)
[](){ #fama-french-1993 }

Fama, E. F. & French, K. R. (1993). "Common Risk Factors in the
Returns on Stocks and Bonds." *Journal of Financial Economics*
33(1), 3–56.

Three-factor model; prototypical multi-factor spanning baseline of
the kind that factrix's spanning-alpha procedure evaluates a
candidate factor against. Catalog-only reference in factrix; no
inline citation site at present.

---

## Event study

### Brown & Warner (1985)
[](){ #brown-warner-1985 }

Brown, S. J. & Warner, J. B. (1985). "Using Daily Stock Returns: The
Case of Event Studies." *Journal of Financial Economics* 14(1),
3–31.

Simulation framework against which event-study tests are evaluated.
The cross-sectional `t` on CAAR is reasonably specified at moderate
`K` under no event-induced variance and is mis-specified under
variance inflation around the event date — the BW1985 documentation
of this failure is the motivation for the
[Boehmer-Musumeci-Poulsen 1991][boehmer-musumeci-poulsen-1991]
standardised AR test that factrix implements.

### Brown & Warner (1980)
[](){ #brown-warner-1980 }

Brown, S. J. & Warner, J. B. (1980). "Measuring Security Price
Performance." *Journal of Financial Economics* 8(3), 205–258.

Earlier monthly-frequency precursor; cited as background for the
mean-adjusted return convention.

### Fama, Fisher, Jensen & Roll (1969)
[](){ #fama-fisher-jensen-roll-1969 }

Fama, E. F., Fisher, L., Jensen, M. C. & Roll, R. (1969). "The
Adjustment of Stock Prices to New Information." *International
Economic Review* 10(1), 1–21.

Founding event-study paper; cited as historical anchor for the
`individual_sparse` cell's lineage.

### MacKinlay (1997)
[](){ #mackinlay-1997 }

MacKinlay, A. C. (1997). "Event Studies in Economics and Finance."
*Journal of Economic Literature* 35(1), 13–39.

Standardised event-window / estimation-window vocabulary; followed by
factrix's event-configuration schema.

### Campbell, Lo & MacKinlay (1997)
[](){ #campbell-lo-mackinlay-1997 }

Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The
Econometrics of Financial Markets*. Princeton University Press.

Textbook treatment of event-study test statistics; cited for the
path-excursion / horizon-scaling vocabulary that factrix's MFE / MAE
path-excursion metric adopts.

### Boehmer, Musumeci & Poulsen (1991)
[](){ #boehmer-musumeci-poulsen-1991 }

Boehmer, E., Musumeci, J. & Poulsen, A. B. (1991). "Event-study
Methodology Under Conditions of Event-induced Variance." *Journal of
Financial Economics* 30(2), 253–272.

BMP standardised AR test. factrix's implementation is a
**BMP-style** variant using mean-adjusted (not market-model) abnormal
returns; the prediction-error variance correction
$\sigma_i \sqrt{1 + 1/T_{\mathrm{est}}}$ of the original BMP
formulation is opt-in and off by default, so default-setting results
will not match a textbook BMP byte-for-byte.

### Patell (1976)
[](){ #patell-1976 }

Patell, J. M. (1976). "Corporate Forecasts of Earnings Per Share and
Stock Price Behavior: Empirical Tests." *Journal of Accounting
Research* 14(2), 246–276.

Patell standardised abnormal return; ancestor of the BMP test that
factrix actually implements.

### Corrado (1989)
[](){ #corrado-1989 }

Corrado, C. J. (1989). "A Nonparametric Test for Abnormal
Security-price Performance in Event Studies." *Journal of Financial
Economics* 23(2), 385–395.

Rank test on event-window abnormal returns; implemented in factrix
with a direction-adjusted two-sided extension.

### Corrado & Zivney (1992)
[](){ #corrado-zivney-1992 }

Corrado, C. J. & Zivney, T. L. (1992). "The Specification and Power
of the Sign Test in Event Study Hypothesis Tests Using Daily Stock
Returns." *Journal of Financial and Quantitative Analysis* 27(3),
465–478.

Sign test for event-study abnormal returns and a modified rank-test
variant for two-sided / cumulative inference (re-rank within the
event window); the source of the direction-adjustment idea adopted
by factrix's Corrado rank-test path.

### Kolari & Pynnönen (2010)
[](){ #kolari-pynnonen-2010 }

Kolari, J. W. & Pynnönen, S. (2010). "Event Study Testing with
Cross-sectional Correlation of Abnormal Returns." *Review of
Financial Studies* 23(11), 3996–4025.

Clustering-adjustment option on factrix's BMP-style test, scaling
the BMP $z$ by
$\sqrt{(1 - \hat r)/(1 + (N_{\mathrm{eff}}-1)\,\hat r)}$ to absorb
same-date abnormal-return cross-correlation. Recommended when
factrix's event-date clustering HHI diagnostic flags high
concentration.

### Sefcik & Thompson (1986)
[](){ #sefcik-thompson-1986 }

Sefcik, S. E. & Thompson, R. (1986). "An Approach to Statistical
Inference in Cross-Sectional Models with Security Abnormal Returns
as Dependent Variable." *Journal of Accounting Research* 24(2),
316–334.

Per-event cross-sectional regression of abnormal return on a
continuous event characteristic; the magnitude-weighted CAAR factrix
computes when an event carries a continuous factor value is a
per-event regression-slope statistic in this lineage, distinct from
the equal-weighted MacKinlay-style CAAR.

### Jaffe (1974)
[](){ #jaffe-1974 }

Jaffe, J. F. (1974). "Special Information and Insider Trading."
*Journal of Business* 47(3), 410–428.

Calendar-time portfolio approach to event studies — recasts
event-indexed inference onto a calendar grid by forming each
calendar period's portfolio of all firms with a recent event and
analysing portfolio returns. Historical anchor for factrix's
dense-calendar CAAR HAC t-test, which adapts the calendar-time idea
by zero-filling non-event dates on the per-event series rather than
forming a calendar-period portfolio across event firms.

### Mandelker (1974)
[](){ #mandelker-1974 }

Mandelker, G. (1974). "Risk and Return: The Case of Merging Firms."
*Journal of Financial Economics* 1(4), 303–335.

Independent contemporaneous calendar-time portfolio paper; cited
alongside Jaffe (1974) as the joint origin of the calendar-time
inference idea that factrix's sparse-panel CAAR adapts (factrix's
zero-fill densification on the per-event series is a related but
distinct operation from the original cross-event calendar-portfolio
construction).

### Fama (1998)
[](){ #fama-1998 }

Fama, E. F. (1998). "Market Efficiency, Long-term Returns, and
Behavioral Finance." *Journal of Financial Economics* 49(3), 283–306.

Methodological comparison of calendar-time portfolio averaging
(average abnormal returns / cumulative abnormal returns) against
buy-and-hold abnormal returns (BHARs), strongly recommending the
calendar-time approach on the grounds that monthly returns are less
susceptible to the bad-model problem and that monthly portfolio
formation automatically absorbs cross-correlations of event-firm
abnormal returns. The modern reference for the densification
rationale underlying factrix's sparse-panel CAAR HAC path.

---

## Multiple-testing correction and selection inference

### Benjamini & Hochberg (1995)
[](){ #benjamini-hochberg-1995 }

Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery
Rate: A Practical and Powerful Approach to Multiple Testing."
*Journal of the Royal Statistical Society: Series B* 57(1), 289–300.

FDR concept and step-up procedure. factrix does not use BH directly
because factor pools are typically dependent — see Benjamini-Hochberg-Yekutieli (BHY) below.

### Benjamini & Yekutieli (2001)
[](){ #benjamini-yekutieli-2001 }

Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False
Discovery Rate in Multiple Testing under Dependency." *Annals of
Statistics* 29(4), 1165–1188.

BH under arbitrary dependence with the `c(m) = Σ 1/i` correction;
provides the dependence-robust FDR adjustment that factrix's
multi-factor BHY path consumes.

### Simes (1986)
[](){ #simes-1986 }

Simes, R. J. (1986). "An Improved Bonferroni Procedure for Multiple
Tests of Significance." *Biometrika* 73(3), 751–754.

Simes' global test combining ordered p-values; used as the group
representative in factrix's hierarchical FDR procedures.

### Benjamini & Heller (2008)
[](){ #benjamini-heller-2008 }

Benjamini, Y. & Heller, R. (2008). "Screening for Partial Conjunction
Hypotheses." *Biometrics* 64(4), 1215–1222.

Partial-conjunction test for "at least r of K hypotheses are true";
backs factrix's partial-conjunction p-value path.

### Yekutieli (2008)
[](){ #yekutieli-2008 }

Yekutieli, D. (2008). "Hierarchical False Discovery Rate-Controlling
Methodology." *Journal of the American Statistical Association*
103(481), 309–316.

Hierarchical FDR with Simes as group representative; cited as the
theoretical context for the BHY + Simes composition.

### Benjamini & Bogomolov (2014)
[](){ #benjamini-bogomolov-2014 }

Benjamini, Y. & Bogomolov, M. (2014). "Selective Inference on
Multiple Families of Hypotheses." *Journal of the Royal Statistical
Society: Series B* 76(1), 297–318.

Selective-inference framework for partitioning a hypothesis set into
families and controlling FDR per family. The paper's recommended
form inflates the within-family level by `R/m` (the fraction of
families flagged by an outer selection step); factrix's
family-partition entry point adopts the family-partition idea but
applies plain per-bucket BHY without the BB14 selection-adjusted
inflation — cross-bucket selection-bias control is the caller's
responsibility (e.g. via factrix's hierarchical BHY procedure).

### Harvey, Liu & Zhu (2016)
[](){ #harvey-liu-zhu-2016 }

Harvey, C. R., Liu, Y. & Zhu, H. (2016). "…and the Cross-Section of
Expected Returns." *Review of Financial Studies* 29(1), 5–68.

Empirical case that conventional single-factor `t ≥ 2.0` is too lax
once the cross-section of tried factors and horizons is accounted
for; HLZ argues a meaningfully higher threshold (typically `t ≳ 3`)
under multiplicity-aware procedures. Motivation for factrix's
BHY-first multi-factor discipline and its family-wise error rate
(FWER)-across-horizons ∘ FDR-within-horizon stack on the
horizon-shopping correction axis.

### Harvey (2017)
[](){ #harvey-2017 }

Harvey, C. R. (2017). "Presidential Address: The Scientific Outlook
in Financial Economics." *Journal of Finance* 72(4), 1399–1440.

p-hacking and replication crisis in finance; cited as motivation for
factrix's pre-registered procedures and BHY-first stance.

### Harvey, C. R. & Liu, Y. (2020)
[](){ #harvey-liu-2020 }

Harvey, C. R. & Liu, Y. (2020). "False (and Missed) Discoveries in
Financial Economics." *Journal of Finance* 75(5), 2503–2553.

Double-bootstrap procedure that jointly calibrates Type I (FDR) and
Type II (miss-rate) error in asset-pricing multiple tests; the
"missed-discovery" axis complements Harvey-Liu-Zhu (2016)'s
Type-I-only focus by adding power-aware hurdles.

### Holm (1979)
[](){ #holm-1979 }

Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test
Procedure." *Scandinavian Journal of Statistics* 6(2), 65–70.

FWER step-down procedure that uniformly dominates Bonferroni under
the same dependence assumptions; the default slice-test adjustment
in factrix's multiple-testing path when callers do not supply a
bootstrap distribution.

### Romano & Wolf (2005)
[](){ #romano-wolf-2005 }

Romano, J. P. & Wolf, M. (2005). "Stepwise Multiple Testing as
Formalized Data Snooping." *Econometrica* 73(4), 1237–1282.

Bootstrap-based FWER step-down exploiting the joint dependence of
test statistics; factrix's bootstrap-based FWER step-down option for
the date-shared slice-test setting (e.g. universe pairwise IC).

### White (2000)
[](){ #white-2000 }

White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*
68(5), 1097–1126.

Bootstrap test for data-snooping bias in model-selection settings;
the canonical correction factrix's greedy forward-selection path
does *not* apply (the selection path is documented as a survivor
screen, not as inference).

### Hansen (2005)
[](){ #hansen-2005 }

Hansen, P. R. (2005). "A Test for Superior Predictive Ability."
*Journal of Business & Economic Statistics* 23(4), 365–380.

Refinement of the White (2000) reality check; cited as background
for the SPA family that factrix does not implement.

### Berk, Brown, Buja, Zhang & Zhao (2013)
[](){ #berk-brown-buja-zhang-zhao-2013 }

Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. (2013). "Valid
Post-Selection Inference." *Annals of Statistics* 41(2), 802–837.

PoSI inference after greedy selection; background for the known
invalidity of post-selection p-values from factrix's greedy
forward-selection path.

### Leeb & Pötscher (2005)
[](){ #leeb-potscher-2005 }

Leeb, H. & Pötscher, B. M. (2005). "Model Selection and Inference:
Facts and Fiction." *Econometric Theory* 21(1), 21–59.

Theoretical case against routine post-selection inference; cited
alongside Berk et al. (2013).

### Efron (2010)
[](){ #efron-2010 }

Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods
for Estimation, Testing, and Prediction*. Cambridge University Press.

Empirical-Bayes alternative to FDR for large-scale multiple testing;
reference work on local-fdr and shrinkage estimation as the
empirical-Bayes counterpart to factrix's frequentist BHY stance.
Catalog-only reference in factrix; no inline citation site at
present.

### Bailey & López de Prado (2014)
[](){ #bailey-lopez-de-prado-2014 }

Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting, and
Non-Normality." *Journal of Portfolio Management* 40(5), 94–107.

Parallel multiple-trials correction operating on the Sharpe rather
than the p-value: deflates an observed Sharpe by the expected
maximum under a number-of-trials null. Related literature for the
horizon-shopping multiple-trials problem on the Sharpe axis; not an
implemented procedure in factrix.

---

## Robust statistics, scale, and bootstrap

### Huber (1964)
[](){ #huber-1964 }

Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
*Annals of Mathematical Statistics* 35(1), 73–101.

M-estimator framework for robust location estimation under
contaminated-normal models. Foundational for the broader "robustify
the central tendency before it gates downstream inference" stance
that factrix applies to per-date cross-sections — the MAD-as-scale
lineage itself runs through [Hampel (1974)][hampel-1974], not this
paper.

### Huber (1981)
[](){ #huber-1981 }

Huber, P. J. (1981). *Robust Statistics*. Wiley.

Textbook treatment of robust scale; reference for the
`1.4826 × MAD` consistency factor used in factrix winsorisation.

### Hampel (1974)
[](){ #hampel-1974 }

Hampel, F. R. (1974). "The Influence Curve and its Role in Robust
Estimation." *Journal of the American Statistical Association*
69(346), 383–393.

Influence-function framework for local-robustness analysis and the
canonical reference popularising the median absolute deviation as a
robust scale estimator (attributing the MAD itself to Gauss).
Supplies the conceptual language factrix uses for per-date
robustification of scale and for breakdown-point claims on
estimators such as Theil-Sen (the breakdown-point concept itself
predates this paper — Hampel 1968 / 1971 — but the 1974 paper places
breakdown-point reasoning and the influence function on the same
conceptual map).

### Rousseeuw & Croux (1993)
[](){ #rousseeuw-croux-1993 }

Rousseeuw, P. J. & Croux, C. (1993). "Alternatives to the Median
Absolute Deviation." *Journal of the American Statistical
Association* 88(424), 1273–1283.

Sn / Qn estimators as MAD alternatives; cited as the "considered
but not implemented" alternative robust scale.

### Künsch (1989)
[](){ #kunsch-1989 }

Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General
Stationary Observations." *Annals of Statistics* 17(3), 1217–1241.

Fixed-block (moving-block) bootstrap for stationary time series;
underlies factrix's deterministic block-bootstrap scheme.

### Politis & Romano (1994)
[](){ #politis-romano-1994 }

Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap."
*Journal of the American Statistical Association* 89(428),
1303–1313.

Stationary block bootstrap with geometric block lengths; underlies
factrix's stationary block-bootstrap scheme.

### Politis & White (2004)
[](){ #politis-white-2004 }

Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection
for the Dependent Bootstrap." *Econometric Reviews* 23(1), 53–70.

Data-driven block-length selector for stationary / circular
bootstraps; supplies the automatic block-length choice in factrix's
bootstrap path when callers do not pass one.

### Sen (1968)
[](){ #sen-1968 }

Sen, P. K. (1968). "Estimates of the Regression Coefficient Based on
Kendall's Tau." *Journal of the American Statistical Association*
63(324), 1379–1389.

Theil-Sen median-slope estimator; the basis of factrix's
breakdown-robust IC-trend slope.

---

## Unit-root, predictive regression, and persistence

### Dickey & Fuller (1979)
[](){ #dickey-fuller-1979 }

Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators
for Autoregressive Time Series with a Unit Root." *Journal of the
American Statistical Association* 74(366), 427–431.

Dickey-Fuller unit-root test on $H_0: \beta = 0$ in
$\Delta y_t = \alpha + \beta\, y_{t-1} + \varepsilon$; the foundational
unit-root null underlying factrix's ADF persistence diagnostic. The
*augmented* form factrix actually applies (lagged differences added
to whiten serially-correlated errors) is [Said-Dickey
(1984)][said-dickey-1984], not this 1979 paper.

### Said & Dickey (1984)
[](){ #said-dickey-1984 }

Said, S. E. & Dickey, D. A. (1984). "Testing for Unit Roots in
Autoregressive-Moving Average Models of Unknown Order." *Biometrika*
71(3), 599–607.

Approximates ARMA errors by a long autoregression and proves the
Dickey-Fuller $t$-statistic retains its limit distribution at
lag-order rate $o(T^{1/3})$. Justifies the augmentation form that
factrix's ADF persistence diagnostic relies on; concrete data-driven
lag-selection rules (AIC / BIC / Ng-Perron) sit in a separate
literature.

### MacKinnon (1996)
[](){ #mackinnon-1996 }

MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit
Root and Cointegration Tests." *Journal of Applied Econometrics*
11(6), 601–618.

Response-surface critical values for ADF; factrix interpolates
linearly against the constant-only specification when converting the
ADF statistic to a p-value.

### Stambaugh (1999)
[](){ #stambaugh-1999 }

Stambaugh, R. F. (1999). "Predictive Regressions." *Journal of
Financial Economics* 54(3), 375–421.

Finite-sample bias of ordinary least squares (OLS) $\hat\beta$ in
predictive regressions when the predictor is persistent *and* its
innovation is correlated with the return innovation (both conditions
are necessary); factrix flags the persistence channel via ADF rather
than auto-correcting.

### Campbell & Yogo (2006)
[](){ #campbell-yogo-2006 }

Campbell, J. Y. & Yogo, M. (2006). "Efficient Tests of Stock Return
Predictability." *Journal of Financial Economics* 81(1), 27–60.

Bonferroni Q-test built on a DF-GLS confidence interval for the
persistence parameter; a corrective-inference alternative to
flag-only diagnostics for predictive regressions under near-unit-root
predictors.

### Phillips & Magdalinos (2009)
[](){ #phillips-magdalinos-2009 }

Phillips, P. C. B. & Magdalinos, T. (2009). "Econometric Inference
in the Vicinity of Unity." Working paper, Singapore Management
University.

Introduces IVX: mildly-integrated internal instruments yielding
pivotal chi-square inference for predictive regression across
integrated, near-integrated, and mildly-integrated regressors. The
theoretical foundation for the Kostakis-Magdalinos-Stamatogiannis
(2015) empirical implementation.

### Kostakis, Magdalinos & Stamatogiannis (2015)
[](){ #kostakis-magdalinos-stamatogiannis-2015 }

Kostakis, A., Magdalinos, T. & Stamatogiannis, M. P. (2015). "Robust
Econometric Inference for Stock Return Predictability." *Review of
Financial Studies* 28(5), 1506–1553.

Empirical IVX-Wald test for stock-return predictability robust to
regressor persistence (stationary through nonstationary), supporting
multivariate and long-horizon predictive specifications; the
practical complement to [Phillips-Magdalinos
(2009)][phillips-magdalinos-2009] on the inference axis adjacent to
factrix's ADF persistence flag.

### Richardson & Stock (1989)
[](){ #richardson-stock-1989 }

Richardson, M. & Stock, J. H. (1989). "Drawing Inferences from
Statistics Based on Multiyear Asset Returns." *Journal of Financial
Economics* 25(2), 323–348.

Alternative asymptotic theory for overlapping multiyear-return
statistics under a horizon-grows-with-sample limit, where
conventional asymptotics misrepresent the finite-sample distribution
of long-horizon predictive coefficients. The limit-theory backstop
for the HAC and sub-sampling fixes factrix applies on long-horizon
paths (the sub-sampling / lag-floor *remedies* themselves trace to
[Hansen-Hodrick (1980)][hansen-hodrick-1980] and the broader HAC
literature, not to this paper).

### Stock & Watson (1988)
[](){ #stock-watson-1988 }

Stock, J. H. & Watson, M. W. (1988). "Variable Trends in Economic
Time Series." *Journal of Economic Perspectives* 2(3), 147–174.

Practitioner-oriented review of the consequences of unit roots in
macroeconomic time series; cited as general background for why an
ADF flag matters in factrix's IC-trend persistence path. The specific
p-value threshold factrix uses for unit-root suspicion is folklore
from the unit-root literature (closer to Stock 1994 *Handbook of
Econometrics* §III) rather than a direct prescription from this
paper.

### Fama & French (1988)
[](){ #fama-french-1988 }

Fama, E. F. & French, K. R. (1988). "Dividend Yields and Expected
Stock Returns." *Journal of Financial Economics* 22(1), 3–25.

Canonical direct long-horizon predictive regression: summed log
returns `r_{t→t+N} = Σ log(P_{t+k}/P_{t+k−1})` regressed on the
dividend yield at horizons 1–4 years. Linear-additive across
horizons by construction, no compounding bias. The academic-standard
alternative to factrix's `÷N` arithmetic per-period normalisation —
factrix uses `÷N` for scale comparability across horizons, not
because Fama-French (1988) recommends it.

### Boudoukh, Richardson & Whitelaw (2008)
[](){ #boudoukh-richardson-whitelaw-2008 }

Boudoukh, J., Richardson, M. & Whitelaw, R. F. (2008). "The Myth of
Long-Horizon Predictability." *Review of Financial Studies* 21(4),
1577–1605.

Under the null and a persistent regressor (most factor signals), OLS
slope estimators across horizons are highly correlated — approaching
unity between adjacent horizons at dividend-yield-like persistence —
and `R²` is roughly proportional to horizon. Across-horizon test
statistics are not independent and BHY's positive regression
dependence on a subset (PRDS) assumption fails, so factrix uses an
FWER (independence-free) inner step before BHY in its multi-factor
BHY path — the FWER prescription is factrix's response to BRW's
correlation result, not BRW's own recommendation. Also motivates
treating per-period scaling as separate from inference in factrix's
forward-return path: the across-horizon dependence BRW documents is
not addressed by any normalisation choice.

---

## Factor zoo, replication, and out-of-sample (OOS) decay

### McLean & Pontiff (2016)
[](){ #mclean-pontiff-2016 }

McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy
Stock Return Predictability?" *Journal of Finance* 71(1), 5–32.

Empirical ~32% post-publication decay in factor returns; the
canonical OOS-decay benchmark underlying factrix's multi-split
OOS-decay procedure.

### Hou, Xue & Zhang (2020)
[](){ #hou-xue-zhang-2020 }

Hou, K., Xue, C. & Zhang, L. (2020). "Replicating Anomalies."
*Review of Financial Studies* 33(5), 2019–2133.

Large-scale replication of published anomalies under NYSE-breakpoint,
value-weighted testing that jointly mitigates microcap influence —
~65% of 452 anomalies fail $|t| \geq 1.96$ once both microcap
mitigations are applied. The headline reason factrix prefers
value-weighted spreads in capacity-constrained settings.

### Chen & Zimmermann (2022)
[](){ #chen-zimmermann-2022 }

Chen, A. Y. & Zimmermann, T. (2022). "Open Source Cross-Sectional
Asset Pricing." *Critical Finance Review* 11(2), 207–264.

Open-source reproduction of 300+ cross-sectional anomalies with
public data and code; the empirical reproducibility anchor for
factrix's downstream slice-conditional IC analyses (regime, universe,
or other slice-axis stratifications layered on top of a reproducible
characteristic set).

### López de Prado (2018)
[](){ #lopez-de-prado-2018 }

López de Prado, M. (2018). *Advances in Financial Machine Learning*.
Wiley.

CPCV (Combinatorial Purged CV) and broader ML-aware backtesting
discipline; the robust train/test alternative to factrix's
multi-split OOS-decay path.

### Green, Hand & Zhang (2017)
[](){ #green-hand-zhang-2017 }

Green, J., Hand, J. R. M. & Zhang, X. F. (2017). "The Characteristics
that Provide Independent Information about Average U.S. Monthly
Stock Returns." *Review of Financial Studies* 30(12), 4389–4436.

Simultaneous Fama-MacBeth regression of ~90 firm characteristics on
cross-sectional returns, identifying the small subset that retains
independent explanatory power and documenting the post-2003 collapse
of the broader characteristic zoo. Empirical anchor for taking
characteristic redundancy seriously in multi-factor screening (not
for univariate IC ranking, which is the simpler — and weaker —
methodology factrix exposes alongside the multi-factor BHY path).

### Jensen, Kelly & Pedersen (2023)
[](){ #jensen-kelly-pedersen-2023 }

Jensen, T. I., Kelly, B. & Pedersen, L. H. (2023). "Is There a
Replication Crisis in Finance?" *Journal of Finance* 78(5),
2465–2518.

Bayesian hierarchical re-evaluation of the factor zoo; concludes
most published anomalies replicate after appropriate priors.
Cited in design notes as the "why not Bayesian" comparison.

### Lou & Polk (2022)
[](){ #lou-polk-2022 }

Lou, D. & Polk, C. (2022). "Comomentum: Inferring Arbitrage
Activity from Return Correlations." *Review of Financial Studies*
35(7), 3272–3302.

Comomentum measures crowding from cross-asset return correlations
within momentum winners and losers. A suggestive *crowding*
explanation for downward-sloping IC over time in factrix's IC-trend
interpretation; the paper's direct subject is return comomentum, not
IC slope — [McLean-Pontiff 2016][mclean-pontiff-2016] is the cleaner
cite for post-publication IC decay.

---

## Factor spanning, selection, and active-management heuristics

### Barillas & Shanken (2017)
[](){ #barillas-shanken-2017 }

Barillas, F. & Shanken, J. (2017). "Which Alpha?" *Review of
Financial Studies* 30(4), 1316–1338.

Spanning-test framework for traded-factor model comparison: model
comparison reduces to whether each model prices the *other* model's
factors (left-hand-side = competing factors, not test assets), and
the result applies to nested and non-nested comparisons alike. The
methodology behind factrix's spanning-alpha procedure.

### Barillas & Shanken (2018)
[](){ #barillas-shanken-2018 }

Barillas, F. & Shanken, J. (2018). "Comparing Asset Pricing Models."
*Journal of Finance* 73(2), 715–754.

Bayesian model-comparison alternative; cited in design notes as the
"why not Bayesian" comparison for spanning tests.

### Feng, Giglio & Xiu (2020)
[](){ #feng-giglio-xiu-2020 }

Feng, G., Giglio, S. & Xiu, D. (2020). "Taming the Factor Zoo: A
Test of New Factors." *Journal of Finance* 75(3), 1327–1370.

Two-pass model-selection-corrected stochastic discount factor (SDF)
estimator for testing whether a candidate factor adds incremental
pricing power; the principled alternative to greedy forward
selection that factrix's spanning toolkit notes but does not
implement.

### Gibbons, Ross & Shanken (1989)
[](){ #gibbons-ross-shanken-1989 }

Gibbons, M. R., Ross, S. A. & Shanken, J. (1989). "A Test of the
Efficiency of a Given Portfolio." *Econometrica* 57(5), 1121–1152.

GRS test for joint α = 0 across portfolios; the parametric ancestor
of factrix's spanning toolkit.

### Kozak, Nagel & Santosh (2020)
[](){ #kozak-nagel-santosh-2020 }

Kozak, S., Nagel, S. & Santosh, S. (2020). "Shrinking the Cross
Section." *Journal of Financial Economics* 135(2), 271–292.

Bayesian shrinkage on the SDF coefficients; cited in design notes
as the "why not Bayesian" comparison for factor selection.

### Patton & Timmermann (2010)
[](){ #patton-timmermann-2010 }

Patton, A. J. & Timmermann, A. (2010). "Monotonicity in Asset
Returns: New Tests with Applications to the Term Structure, the
CAPM, and Portfolio Sorts." *Journal of Financial Economics* 98(3),
605–625.

Formal monotonicity tests for portfolio sorts; the rigour benchmark
above factrix's per-date Spearman monotonicity metric.

### Novy-Marx & Velikov (2016)
[](){ #novy-marx-velikov-2016 }

Novy-Marx, R. & Velikov, M. (2016). "A Taxonomy of Anomalies and
Their Trading Costs." *Review of Financial Studies* 29(1), 104–147.

Turnover, generalised buy/hold spreads, and breakeven-cost analysis
of anomaly portfolios; the lineage for turnover-aware and
breakeven-cost diagnostics.

### DeMiguel, Martin-Utrera, Nogales & Uppal (2020)
[](){ #demiguel-martin-utrera-nogales-uppal-2020 }

DeMiguel, V., Martin-Utrera, A., Nogales, F. J. & Uppal, R. (2020).
"A Transaction-cost Perspective on the Multitude of Firm
Characteristics." *Review of Financial Studies* 33(5), 2180–2222.

Transaction-cost-aware factor selection; the structural reason
gross-spread metrics need a cost-deduction companion in factrix's
spread/cost-net path.

### DeMiguel, Garlappi & Uppal (2009)
[](){ #demiguel-garlappi-uppal-2009 }

DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive
Diversification: How Inefficient is the 1/N Portfolio Strategy?"
*Review of Financial Studies* 22(5), 1915–1953.

Equal-weight benchmark; cited in design notes as the "why no
optimisation" comparison.

### Asness, Moskowitz & Pedersen (2013)
[](){ #asness-moskowitz-pedersen-2013 }

Asness, C. S., Moskowitz, T. J. & Pedersen, L. H. (2013). "Value and
Momentum Everywhere." *Journal of Finance* 68(3), 929–985.

Cross-asset factor evaluation methodology; cited in design notes as
the cross-market evaluation pattern factrix does *not* aggregate at
its scope boundary.

### Ambachtsheer (1977)
[](){ #ambachtsheer-1977 }

Ambachtsheer, K. P. (1977). "Where Are the Customers' Alphas?"
*Journal of Portfolio Management* 4(1), 52–56.

Early operational use of IC-based alpha attribution in pension-fund
performance discussion; the appraisal-ratio ancestor of the formal
$\mathrm{IR} \approx \mathrm{IC} \times \sqrt{\mathrm{breadth}}$
decomposition is Treynor & Black (1973), and the breadth identity
itself is canonically derived in [Grinold 1989][grinold-1989].

---

## Methodology critique and composite-score design

### Goodhart (1984)
[](){ #goodhart-1984 }

Goodhart, C. A. E. (1984). "Problems of Monetary Management: The UK
Experience." In *Monetary Theory and Practice*. Macmillan.

"When a measure becomes a target, it ceases to be a good measure";
cited in design notes as the structural argument against composite
factor scores.

### Cochrane (2005)
[](){ #cochrane-2005 }

Cochrane, J. H. (2005). *Asset Pricing* (Revised ed.). Princeton
University Press.

Textbook treatment of the SDF framework underlying factrix's
spanning and FM tests.

### Cochrane (2011)
[](){ #cochrane-2011 }

Cochrane, J. H. (2011). "Presidential Address: Discount Rates."
*Journal of Finance* 66(4), 1047–1108.

The "factor zoo" critique that motivates BHY-first multi-factor
discipline.

### Herfindahl (1950)
[](){ #herfindahl-1950 }

Herfindahl, O. C. (1950). *Concentration in the U.S. Steel
Industry*. PhD dissertation, Columbia University.

Origin of the HHI used by factrix's top-concentration and
clustering-diagnostic metrics.

### Hirschman (1945)
[](){ #hirschman-1945 }

Hirschman, A. O. (1945). *National Power and the Structure of
Foreign Trade*. University of California Press.

Independent earlier formulation of HHI; cited alongside Herfindahl
(1950).

### Jacquier, Kane & Marcus (2003)
[](){ #jacquier-kane-marcus-2003 }

Jacquier, E., Kane, A. & Marcus, A. J. (2003). "Geometric or
Arithmetic Mean: A Reconsideration." *Financial Analysts Journal*
59(6), 46–53.

Compounding at the arithmetic mean is an upward-biased estimator of
cumulative wealth; the geometric mean is itself biased; the unbiased
estimator is a horizon-weighted blend of the two with weights
depending on the forecast horizon / sample-length ratio. The
compounding-bias caveat for factrix's `÷N` per-period forward-return
normalisation — the bias affects signed-return mean and t-tests at
large `N`, but is negligible for rank-based IC.
