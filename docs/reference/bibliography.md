# Bibliography

Core references behind the registered procedures, statistical
estimators, and multiple-testing corrections shipped in factrix. Use
this page as the citation target when reporting results downstream.

For the broader literature review (design rationale, alternatives
considered, and dimensions outside factrix's signal-validation scope)
see [Development § Methodology](../development/methodology.md).

Anchors below are consumed as cross-references from
[Statistical methods](statistical-methods.md).

---

## Time-series regression and HAC inference

### <a id="newey-west-1987"></a>Newey & West (1987)

Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55(3), 703–708.

Bartlett-kernel HAC variance estimator; underlies every NW HAC t-test
in factrix (`_newey_west_se`, `_newey_west_t_test`).

### <a id="newey-west-1994"></a>Newey & West (1994)

Newey, W. K. & West, K. D. (1994). "Automatic Lag Selection in
Covariance Matrix Estimation." *Review of Economic Studies* 61(4),
631–653.

Data-adaptive plug-in bandwidth selection; cited as background.
factrix uses the simpler Andrews (1991) Bartlett growth rate
`⌊T^(1/3)⌋` floored against the Hansen-Hodrick overlap rule.

### <a id="andrews-1991"></a>Andrews (1991)

Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation
Consistent Covariance Matrix Estimation." *Econometrica* 59(3),
817–858.

Optimal Bartlett growth rate `T^(1/3)`; the basis of factrix's default
NW lag rule.

### <a id="hansen-hodrick-1980"></a>Hansen & Hodrick (1980)

Hansen, L. P. & Hodrick, R. J. (1980). "Forward Exchange Rates as
Optimal Predictors of Future Spot Rates: An Econometric Analysis."
*Journal of Political Economy* 88(5), 829–853.

K-period forecast residuals carry MA(K−1) structure; the source of
the `forward_periods − 1` lag floor that factrix combines with the
Andrews rule under overlapping forward returns.

### <a id="white-1980"></a>White (1980)

White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix
Estimator and a Direct Test for Heteroskedasticity." *Econometrica*
48(4), 817–838.

HC0 sandwich estimator; the heteroskedasticity-only ancestor of NW HAC.

---

## Cross-section and panel pricing

### <a id="grinold-1989"></a>Grinold (1989)

Grinold, R. C. (1989). "The Fundamental Law of Active Management."
*Journal of Portfolio Management* 15(3), 30–37.

`IR ≈ IC × √breadth`; motivates IC as the canonical signal-quality
measure and IR/ICIR as its time-stability normalisation.

### <a id="fama-macbeth-1973"></a>Fama & MacBeth (1973)

Fama, E. F. & MacBeth, J. D. (1973). "Risk, Return, and Equilibrium:
Empirical Tests." *Journal of Political Economy* 81(3), 607–636.

Two-stage λ procedure: per-date cross-sectional regression then
time-series t-test on E[λ]. The FM cell uses this with NW HAC at
stage 2.

### <a id="black-jensen-scholes-1972"></a>Black, Jensen & Scholes (1972)

Black, F., Jensen, M. C. & Scholes, M. (1972). "The Capital Asset
Pricing Model: Some Empirical Tests." In Jensen, M. (ed.), *Studies
in the Theory of Capital Markets*. Praeger.

Per-asset time-series β then cross-asset t on E[β]. The
`common_continuous` cell mirrors this aggregation order.

### <a id="petersen-2009"></a>Petersen (2009)

Petersen, M. A. (2009). "Estimating Standard Errors in Finance Panel
Data Sets: Comparing Approaches." *Review of Financial Studies* 22(1),
435–480.

Comparison of FM, clustered, and two-way SE under firm/time
correlation; supports FM + Newey-West as the default for time-effect
panels and motivates the future clustered-SE extension noted in
architecture.

---

## Event study

### <a id="brown-warner-1985"></a>Brown & Warner (1985)

Brown, S. J. & Warner, J. B. (1985). "Using Daily Stock Returns: The
Case of Event Studies." *Journal of Financial Economics* 14(1),
3–31.

Daily-frequency event-study methodology; t-test on CAAR is well
specified at standard sample sizes — backs the parametric path used
by `individual_sparse`.

### <a id="mackinlay-1997"></a>MacKinlay (1997)

MacKinlay, A. C. (1997). "Event Studies in Economics and Finance."
*Journal of Economic Literature* 35(1), 13–39.

Standardised event-window / estimation-window vocabulary; followed by
factrix's `EventConfig` schema.

### <a id="boehmer-musumeci-poulsen-1991"></a>Boehmer, Musumeci & Poulsen (1991)

Boehmer, E., Musumeci, J. & Poulsen, A. B. (1991). "Event-study
Methodology Under Conditions of Event-induced Variance." *Journal of
Financial Economics* 30(2), 253–272.

BMP standardised AR test; implemented as `bmp_test` for the
mean-adjusted, prediction-error-omitted simplification.

---

## Multiple-testing correction

### <a id="benjamini-hochberg-1995"></a>Benjamini & Hochberg (1995)

Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery
Rate: A Practical and Powerful Approach to Multiple Testing." *Journal
of the Royal Statistical Society: Series B* 57(1), 289–300.

FDR concept and step-up procedure. factrix does not use BH directly
because factor pools are typically dependent — see BHY below.

### <a id="benjamini-yekutieli-2001"></a>Benjamini & Yekutieli (2001)

Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False
Discovery Rate in Multiple Testing under Dependency." *Annals of
Statistics* 29(4), 1165–1188.

BH under arbitrary dependence with the `c(m) = Σ 1/i` correction;
implemented as `bhy_adjust` and consumed by `multi_factor.bhy`.

---

## Robust statistics and bootstrap

### <a id="huber-1964"></a>Huber (1964)

Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
*Annals of Mathematical Statistics* 35(1), 73–101.

Foundation for MAD-based robust scale; underlies `mad_winsorize`.

### <a id="politis-romano-1994"></a>Politis & Romano (1994)

Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap."
*Journal of the American Statistical Association* 89(428),
1303–1313.

Stationary block bootstrap. Cited as background; not implemented
inside factrix — see "Evaluated but deliberately not implemented" in
[Statistical methods](statistical-methods.md).

---

## Unit-root and persistence diagnostics

### <a id="dickey-fuller-1979"></a>Dickey & Fuller (1979)

Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators
for Autoregressive Time Series with a Unit Root." *Journal of the
American Statistical Association* 74(366), 427–431.

ADF test on `H₀: β = 0` in `Δy_t = α + β y_{t-1} + ε`; the basis of
factrix's `_adf` persistence diagnostic.

### <a id="mackinnon-1996"></a>MacKinnon (1996)

MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit
Root and Cointegration Tests." *Journal of Applied Econometrics*
11(6), 601–618.

Response-surface critical values for ADF; `_adf_pvalue_interp` does
linear interpolation against the constant-only specification.
