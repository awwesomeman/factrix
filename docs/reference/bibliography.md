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
in factrix (`_newey_west_se`, `_newey_west_t_test`).

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
machinery: (i) the rectangular-kernel HAC SE on a sample mean,
exposed as the `factrix.stats.HansenHodrick` estimator
(`_hansen_hodrick_se` / `_hansen_hodrick_t_test`) for callers who
prefer overlap-targeted rectangular kernel over the Bartlett kernel;
(ii) the `forward_periods − 1` lag floor that factrix combines with
the Andrews rule when running NW HAC under overlapping forward
returns.

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
underlies the two-step efficient `GMM` ``MomentEstimator``
(`factrix.stats.gmm`).

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
inspection) that factrix's `common_continuous` cell adopts; the
`ts_beta` cross-asset t on the mean of per-asset β is a simplified
analogue of this aggregation order rather than a replication of BJS's
grouped-portfolio intercept test.

### Petersen (2009)
[](){ #petersen-2009 }

Petersen, M. A. (2009). "Estimating Standard Errors in Finance Panel
Data Sets: Comparing Approaches." *Review of Financial Studies* 22(1),
435–480.

Comparison of FM, clustered, and two-way SE under firm/time
correlation; supports FM + Newey-West as the default for time-effect
panels and motivates the two-way (date+asset) clustered SE option
exposed in `pooled_ols` and `factrix.stats.WaldTwoWayCluster`.

### Cameron, Gelbach & Miller (2011)
[](){ #cameron-gelbach-miller-2011 }

Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). "Robust
Inference With Multiway Clustering." *Journal of Business & Economic
Statistics* 29(2), 238–249.

Two-way clustering formula `V_AB = V_A + V_B − V_{A∩B}`; backs the
date+asset two-way clustered SE option in `pooled_ols`
(`two_way_cluster_col`) and the `factrix.stats.WaldTwoWayCluster`
estimator.

### Thompson (2011)
[](){ #thompson-2011 }

Thompson, S. B. (2011). "Simple Formulas for Standard Errors that
Cluster by Both Firm and Time." *Journal of Financial Economics*
99(1), 1–10.

Finite-sample correction `df = min(G_A, G_B) − 1` used by `pooled_ols`
when two-way clustering is enabled.

### Shanken (1992)
[](){ #shanken-1992 }

Shanken, J. (1992). "On the Estimation of Beta-Pricing Models."
*Review of Financial Studies* 5(1), 1–33.

Errors-in-variables correction for the Fama-MacBeth stage-2 $t$-stat
when the stage-1 regressor is itself an estimated quantity (rolling
$\beta$, PCA score, ML predictor). Used in `fama_macbeth` when
`is_estimated_factor=True`. The general multi-factor multiplicative
term $1 + \lambda'\Sigma_f^{-1}\lambda$ collapses to
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
power. Cited from ``fama_macbeth`` as cautionary background on factor
validity, separate from the errors-in-variables sampling-error
correction (which is the [Shanken (1992)][shanken-1992] single-factor
case that factrix's ``is_estimated_factor`` flag implements).

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

Three-factor model; the prototypical multi-factor spanning baseline
that `spanning_alpha` evaluates against.

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
standardised AR test (factrix `bmp_test`).

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
factrix's `EventConfig` schema.

### Campbell, Lo & MacKinlay (1997)
[](){ #campbell-lo-mackinlay-1997 }

Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The
Econometrics of Financial Markets*. Princeton University Press.

Textbook treatment of event-study test statistics; cited from
`mfe_mae` for the path-excursion vocabulary and elsewhere as a
general econometrics reference.

### Boehmer, Musumeci & Poulsen (1991)
[](){ #boehmer-musumeci-poulsen-1991 }

Boehmer, E., Musumeci, J. & Poulsen, A. B. (1991). "Event-study
Methodology Under Conditions of Event-induced Variance." *Journal of
Financial Economics* 30(2), 253–272.

BMP standardised AR test. factrix's `bmp_test` is a **BMP-style**
simplification: it uses mean-adjusted abnormal returns and omits the
prediction-error correction of the original BMP formulation, so
results will not match a textbook BMP implementation byte-for-byte.

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

Rank test on event-window abnormal returns; implemented as
`corrado_rank_test` with a direction-adjusted two-sided extension.

### Corrado & Zivney (1992)
[](){ #corrado-zivney-1992 }

Corrado, C. J. & Zivney, T. L. (1992). "The Specification and Power
of the Sign Test in Event Study Hypothesis Tests Using Daily Stock
Returns." *Journal of Financial and Quantitative Analysis* 27(3),
465–478.

Sign-test variant of the Corrado rank test; cited as the source of
the direction-adjustment idea adopted by `corrado_rank_test`.

### Kolari & Pynnönen (2010)
[](){ #kolari-pynnonen-2010 }

Kolari, J. W. & Pynnönen, S. (2010). "Event Study Testing with
Cross-sectional Correlation of Abnormal Returns." *Review of
Financial Studies* 23(11), 3996–4025.

Clustering-adjusted BMP variant; `EventConfig.adjust_clustering=
'kolari_pynnonen'` is reserved for this but the adjustment is not
yet implemented (high-Herfindahl-Hirschman index (HHI) events should fall back to manual
calendar-block bootstrap until it ships).

### Sefcik & Thompson (1986)
[](){ #sefcik-thompson-1986 }

Sefcik, S. E. & Thompson, R. (1986). "An Approach to Statistical
Inference in Cross-Sectional Models with Security Abnormal Returns
as Dependent Variable." *Journal of Accounting Research* 24(2),
316–334.

Per-event cross-sectional regression of abnormal return on a
continuous event characteristic; the `factor × return` magnitude-
weighted CAAR factrix computes (`compute_caar` with continuous
`factor`) is a per-event regression-slope statistic in this lineage,
distinct from the equal-weighted MacKinlay-style CAAR.

### Jaffe (1974)
[](){ #jaffe-1974 }

Jaffe, J. F. (1974). "Special Information and Insider Trading."
*Journal of Business* 47(3), 410–428.

Calendar-time portfolio approach to event studies; densifying the
event-indexed return series to a dense calendar grid (zero-fill on
non-event dates) so HAC-style inference can assume a fixed period
between observations. Cited from `_CAARSparsePanelProcedure` as the
historical anchor for factrix's dense-calendar CAAR HAC t-test.

### Mandelker (1974)
[](){ #mandelker-1974 }

Mandelker, G. (1974). "Risk and Return: The Case of Merging Firms."
*Journal of Financial Economics* 1(4), 303–335.

Independent contemporaneous application of the calendar-time
portfolio approach; cited alongside Jaffe (1974) as the joint origin
of the densification convention factrix's sparse-panel CAAR adopts.

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
abnormal returns. Cited as the modern reference for the densification
rationale in `_CAARSparsePanelProcedure`.

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
implemented as `bhy_adjust` and consumed by `multi_factor.bhy`.

### Simes (1986)
[](){ #simes-1986 }

Simes, R. J. (1986). "An Improved Bonferroni Procedure for Multiple
Tests of Significance." *Biometrika* 73(3), 751–754.

Simes' global test combining ordered p-values; used by `simes_p` as
the group representative in hierarchical FDR procedures.

### Benjamini & Heller (2008)
[](){ #benjamini-heller-2008 }

Benjamini, Y. & Heller, R. (2008). "Screening for Partial Conjunction
Hypotheses." *Biometrics* 64(4), 1215–1222.

Partial-conjunction test for "at least r of K hypotheses are true";
backs `partial_conjunction_p`.

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
independent families with per-family FDR control; underlies the
explicit ``expand_over`` per-bucket step-up convention in
`multi_factor.bhy` (per-bucket adjusted p-values, no global
cross-bucket adjustment).

### Harvey, Liu & Zhu (2016)
[](){ #harvey-liu-zhu-2016 }

Harvey, C. R., Liu, Y. & Zhu, H. (2016). "…and the Cross-Section of
Expected Returns." *Review of Financial Studies* 29(1), 5–68.

Empirical case for raising t-thresholds in factor research; backs
the single-factor `t ≥ 2.0` threshold and the BHY-first multi-factor
discipline in `greedy_forward_selection`, and the family-wise error rate (FWER)-across-horizons
∘ FDR-within-horizon discipline at `factrix.multi_factor.bhy` and the
"Horizon-shopping correction" section of
`docs/guides/batch-screening.md`.

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

FDR-adjusted t-thresholds for asset-pricing tests; complements
Harvey-Liu-Zhu (2016).

### Holm (1979)
[](){ #holm-1979 }

Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test
Procedure." *Scandinavian Journal of Statistics* 6(2), 65–70.

FWER step-down procedure that uniformly dominates Bonferroni under
the same dependence assumptions; the default slice-test adjustment
in `_stats.multiple_testing` when callers do not supply a bootstrap
distribution.

### Romano & Wolf (2005)
[](){ #romano-wolf-2005 }

Romano, J. P. & Wolf, M. (2005). "Stepwise Multiple Testing as
Formalized Data Snooping." *Econometrica* 73(4), 1237–1282.

Bootstrap-based FWER step-down exploiting the joint dependence of
test statistics; implemented in `_stats.multiple_testing` for the
date-shared slice-test setting (e.g. universe pairwise IC).

### White (2000)
[](){ #white-2000 }

White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*
68(5), 1097–1126.

Bootstrap test for data-snooping bias in model-selection settings;
cited from `greedy_forward_selection` as the canonical correction
factrix does *not* apply (the function inflates t-stats by design
and is documented as not for inference).

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

PoSI inference after greedy selection; cited as background for the
known invalidity of post-selection p-values from
`greedy_forward_selection`.

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

Empirical-Bayes alternative to FDR; cited in design notes as the
"why not Bayesian" comparison anchor.

### Bailey & López de Prado (2014)
[](){ #bailey-lopez-de-prado-2014 }

Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting, and
Non-Normality." *Journal of Portfolio Management* 40(5), 94–107.

Parallel multiple-trials correction operating on the Sharpe rather
than the p-value: deflates an observed Sharpe by the expected
maximum under a number-of-trials null. Cited in the
"Horizon-shopping correction" section of
`docs/guides/batch-screening.md` as related literature for the
multiple-trials problem on the Sharpe axis; not an implemented
procedure in factrix.

---

## Robust statistics, scale, and bootstrap

### Huber (1964)
[](){ #huber-1964 }

Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
*Annals of Mathematical Statistics* 35(1), 73–101.

Foundation for MAD-based robust scale; underlies `mad_winsorize`.

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

Influence-function framework underlying the 29.3% breakdown point
factrix advertises for Theil-Sen.

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
underlies the deterministic block scheme in `_stats.bootstrap`.

### Politis & Romano (1994)
[](){ #politis-romano-1994 }

Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap."
*Journal of the American Statistical Association* 89(428),
1303–1313.

Stationary block bootstrap with geometric block lengths; underlies
the stationary scheme in `_stats.bootstrap`.

### Politis & White (2004)
[](){ #politis-white-2004 }

Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection
for the Dependent Bootstrap." *Econometric Reviews* 23(1), 53–70.

Data-driven block-length selector for stationary / circular
bootstraps; supplies the automatic `L` choice in `_stats.bootstrap`
when callers do not pass one.

### Sen (1968)
[](){ #sen-1968 }

Sen, P. K. (1968). "Estimates of the Regression Coefficient Based on
Kendall's Tau." *Journal of the American Statistical Association*
63(324), 1379–1389.

Theil-Sen median-slope estimator; the basis of `ic_trend`'s
breakdown-robust slope.

---

## Unit-root, predictive regression, and persistence

### Dickey & Fuller (1979)
[](){ #dickey-fuller-1979 }

Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators
for Autoregressive Time Series with a Unit Root." *Journal of the
American Statistical Association* 74(366), 427–431.

Augmented Dickey-Fuller (ADF) test on $H_0: \beta = 0$ in $\Delta y_t = \alpha + \beta\, y_{t-1} + \varepsilon$; the basis of
factrix's `_adf` persistence diagnostic.

### Said & Dickey (1984)
[](){ #said-dickey-1984 }

Said, S. E. & Dickey, D. A. (1984). "Testing for Unit Roots in
Autoregressive-Moving Average Models of Unknown Order." *Biometrika*
71(3), 599–607.

ADF extension to ARMA errors; cited as background for the lag
selection inside `_adf`.

### MacKinnon (1996)
[](){ #mackinnon-1996 }

MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit
Root and Cointegration Tests." *Journal of Applied Econometrics*
11(6), 601–618.

Response-surface critical values for ADF; `_adf_pvalue_interp` does
linear interpolation against the constant-only specification.

### Stambaugh (1999)
[](){ #stambaugh-1999 }

Stambaugh, R. F. (1999). "Predictive Regressions." *Journal of
Financial Economics* 54(3), 375–421.

Bias of ordinary least squares (OLS) $\hat\beta$ in predictive regressions when the predictor is
persistent; factrix flags via ADF rather than auto-correcting.

### Campbell & Yogo (2006)
[](){ #campbell-yogo-2006 }

Campbell, J. Y. & Yogo, M. (2006). "Efficient Tests of Stock Return
Predictability." *Journal of Financial Economics* 81(1), 27–60.

Bonferroni Q-test for predictive regressions under near-unit-root
predictors; cited as the proper-inference alternative factrix does
not implement.

### Phillips & Magdalinos (2009)
[](){ #phillips-magdalinos-2009 }

Phillips, P. C. B. & Magdalinos, T. (2009). "Econometric Inference
in the Vicinity of Unity." Working paper, Singapore Management
University.

IVX predictive regression; foundation for the Kostakis et al. (2015)
implementation that factrix does not include.

### Kostakis, Magdalinos & Stamatogiannis (2015)
[](){ #kostakis-magdalinos-stamatogiannis-2015 }

Kostakis, A., Magdalinos, T. & Stamatogiannis, M. P. (2015). "Robust
Econometric Inference for Stock Return Predictability." *Review of
Financial Studies* 28(5), 1506–1553.

Practical IVX implementation cited as background for the
predictive-regression bias factrix flags but does not auto-correct.

### Richardson & Stock (1989)
[](){ #richardson-stock-1989 }

Richardson, M. & Stock, J. H. (1989). "Drawing Inferences from
Statistics Based on Multiyear Asset Returns." *Journal of Financial
Economics* 25(2), 323–348.

Asymptotic theory for overlapping multiyear returns; cited as
background for the Hansen-Hodrick lag floor.

### Stock & Watson (1988)
[](){ #stock-watson-1988 }

Stock, J. H. & Watson, M. W. (1988). "Variable Trends in Economic
Time Series." *Journal of Economic Perspectives* 2(3), 147–174.

Practitioner-oriented review of the consequences of unit roots in
macroeconomic time series; cited as general background for why an
ADF flag matters in `ic_trend`. The specific `p > 0.10 ⇒ unit-root
suspect` threshold is folklore from the unit-root literature
(closer to Stock 1994 *Handbook of Econometrics* §III) rather than a
direct prescription from this paper.

### Fama & French (1988)
[](){ #fama-french-1988 }

Fama, E. F. & French, K. R. (1988). "Dividend Yields and Expected
Stock Returns." *Journal of Financial Economics* 22(1), 3–25.

Canonical direct long-horizon predictive regression: summed log
returns `r_{t→t+N} = Σ log(P_{t+k}/P_{t+k−1})` regressed on the
dividend yield at horizons 1–4 years. Linear-additive across
horizons by construction, no compounding bias. Cited at
`compute_forward_return` as the academic-standard alternative to
factrix's `÷N` arithmetic per-period normalization — factrix uses
`÷N` for scale comparability across horizons, not because
Fama-French (1988) recommends it.

### Boudoukh, Richardson & Whitelaw (2008)
[](){ #boudoukh-richardson-whitelaw-2008 }

Boudoukh, J., Richardson, M. & Whitelaw, R. F. (2008). "The Myth of
Long-Horizon Predictability." *Review of Financial Studies* 21(4),
1577–1605.

Under the null and a persistent regressor (most factor signals),
OLS slope estimators across horizons are highly correlated —
approaching unity between adjacent horizons at dividend-yield-like
persistence — and `R²` is roughly proportional to horizon. Cited at
`factrix.multi_factor.bhy` and the "Horizon-shopping correction"
section of `docs/guides/batch-screening.md`: across-horizon test
statistics are not independent and BHY's positive regression dependence on a subset (PRDS) assumption fails, so
factrix uses an FWER (independence-free) inner step before BHY —
the FWER prescription is factrix's response to BRW's correlation
result, not BRW's own recommendation. Cited at
`compute_forward_return` to motivate treating per-period scaling
as separate from inference: the across-horizon dependence BRW
documents is not addressed by any normalization choice.

---

## Factor zoo, replication, and out-of-sample (OOS) decay

### McLean & Pontiff (2016)
[](){ #mclean-pontiff-2016 }

McLean, R. D. & Pontiff, J. (2016). "Does Academic Research Destroy
Stock Return Predictability?" *Journal of Finance* 71(1), 5–32.

Empirical ~32% post-publication decay in factor returns; the
canonical OOS-decay benchmark cited from `multi_split_oos_decay`.

### Hou, Xue & Zhang (2020)
[](){ #hou-xue-zhang-2020 }

Hou, K., Xue, C. & Zhang, L. (2020). "Replicating Anomalies."
*Review of Financial Studies* 33(5), 2019–2133.

Large-scale replication of published anomalies under value-weighted
testing; cited from `quantile_spread_vw` as the headline reason for
preferring VW over EW spreads in capacity-constrained settings.

### Chen & Zimmermann (2022)
[](){ #chen-zimmermann-2022 }

Chen, A. Y. & Zimmermann, T. (2022). "Open Source Cross-Sectional
Asset Pricing." *Critical Finance Review* 11(2), 207–264.

Open-source replication of 200+ anomalies with code; cited as the
empirical motivation for regime-stratified IC analysis (now expressed
via `slice_pairwise_test(ic, ..., label="regime")`).

### López de Prado (2018)
[](){ #lopez-de-prado-2018 }

López de Prado, M. (2018). *Advances in Financial Machine Learning*.
Wiley.

CPCV (Combinatorial Purged CV) and broader ML-aware
backtesting discipline; cited from `multi_split_oos_decay` as the
robust train/test alternative factrix does not yet implement.

### Green, Hand & Zhang (2017)
[](){ #green-hand-zhang-2017 }

Green, J., Hand, J. R. M. & Zhang, X. F. (2017). "The Characteristics
that Provide Independent Information about Average U.S. Monthly
Stock Returns." *Review of Financial Studies* 30(12), 4389–4436.

Cross-sectional IC ranking of ~100 firm characteristics; cited as
empirical anchor for IC-based screening.

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
within momentum winners and losers. Cited from `ic_trend` as a
suggestive *crowding* explanation for downward-sloping IC over time;
the paper's direct subject is return comomentum, not IC slope —
[McLean-Pontiff 2016][mclean-pontiff-2016] is the cleaner cite for
post-publication IC decay.

---

## Factor spanning, selection, and active-management heuristics

### Barillas & Shanken (2017)
[](){ #barillas-shanken-2017 }

Barillas, F. & Shanken, J. (2017). "Which Alpha?" *Review of
Financial Studies* 30(4), 1316–1338.

Spanning-test framework for nested factor models; the methodology
behind `spanning_alpha`.

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

Two-pass model-selection-corrected stochastic discount factor (SDF) estimator for testing whether
a candidate factor adds incremental pricing power. Cited from
`spanning.py` as the principled alternative to greedy forward
selection.

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

Formal monotonicity tests for portfolio sorts; cited as the rigour
benchmark above factrix's per-date Spearman `monotonicity` metric.

### Novy-Marx & Velikov (2016)
[](){ #novy-marx-velikov-2016 }

Novy-Marx, R. & Velikov, M. (2016). "A Taxonomy of Anomalies and
Their Trading Costs." *Review of Financial Studies* 29(1), 104–147.

Notional-turnover $\tau$ and breakeven-cost framework; the source of
`notional_turnover` and `breakeven_cost`.

### DeMiguel, Martin-Utrera, Nogales & Uppal (2020)
[](){ #demiguel-martin-utrera-nogales-uppal-2020 }

DeMiguel, V., Martin-Utrera, A., Nogales, F. J. & Uppal, R. (2020).
"A Transaction-cost Perspective on the Multitude of Firm
Characteristics." *Review of Financial Studies* 33(5), 2180–2222.

Transaction-cost-aware factor selection; cited from `net_spread`
as the structural reason gross-spread metrics need a cost-deduction
companion.

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

Origin of the HHI used by `top_concentration` and
`clustering_diagnostic`.

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
cumulative wealth; the geometric mean is itself biased; the
unbiased estimator is a horizon-weighted blend of the two with
weights depending on the forecast horizon / sample-length ratio.
Cited at `compute_forward_return` for the compounding-bias caveat
of factrix's `÷N` per-period normalization — the bias affects
signed-return mean and t-tests at large `N`, but is negligible for
rank-based IC.
