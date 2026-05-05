# Design notes

A record of choices factrix **deliberately did not implement**, with
the trade-off and the literature anchor that grounds each decision.
Reading this page is the fastest way to understand what factrix is
*not*: it explains the negative space around the scope so contributors
do not re-litigate settled boundaries. Positive design lives in
[Architecture](architecture.md); the literature targets are in the
[Bibliography](../reference/bibliography.md).

The seven sections below correspond to the seven structural
"considered but not done" decisions that recur in design discussions.

---

## 1. No composite factor score

factrix evaluates each factor through a list of separately reported
metrics — IC, FM-λ, quantile spread, monotonicity, OOS decay,
turnover, etc. — and renders a binary `verdict()`. It deliberately
does **not** collapse these into a single weighted score.

- A composite score becomes a target to be optimised against. Once
  factor generators tune to the composite, the score stops measuring
  what it nominally measures — Goodhart's law in its original
  monetary-policy form ([Goodhart 1984][goodhart-1984]) and as
  rediscovered in the asset-pricing context.
- Different metrics test different null hypotheses (mean,
  monotonicity, persistence, capacity). Folding them into one
  weighted scalar implicitly assigns a price to each null, which is
  not a problem the data can answer.
- Naive equal weighting beats most "optimal" weight schemes when
  estimation error swamps optimisation gain — the same finding that
  motivates equal-weight portfolios over mean-variance under realistic
  parameter uncertainty ([DeMiguel-Garlappi-Uppal
  2009][demiguel-garlappi-uppal-2009]).
- Per-metric pass/fail keeps the user in the inference loop. A score
  can hide which dimension failed; a binary verdict per metric makes
  it explicit.

---

## 2. No factor selection or Stage-2 ranking

factrix does not decide *which* factors should make it into a final
multi-factor portfolio. Outputs are per-factor diagnostics; the
selection step is left to the user (or to a separate stage built on
factrix's outputs).

- Single-factor quality and incremental contribution to a model are
  separable problems. Conflating them is what the
  [shrinkage][kozak-nagel-santosh-2020] and
  [zoo-taming][feng-giglio-xiu-2020] literatures explicitly warn
  against.
- Greedy forward selection is implemented as a diagnostic
  (`greedy_forward_selection`) but documented as **not for
  inference** — t-stats from greedy paths are inflated by selection,
  and post-selection inference requires PoSI machinery
  ([Berk-Brown-Buja-Zhang-Zhao 2013][berk-brown-buja-zhang-zhao-2013],
  [Leeb-Pötscher 2005][leeb-potscher-2005]) that factrix does not
  ship.
- Spanning tests ([Barillas-Shanken 2017][barillas-shanken-2017])
  give the *honest* incremental-α answer for a fixed pair of factor
  sets. They are exposed as `spanning_alpha`; building a selection
  pipeline on top is a downstream concern.

---

## 3. No capacity / size-of-trade scoring

factrix reports `notional_turnover`, `breakeven_cost`, `net_spread`,
and `top_concentration` as **profile diagnostics** — values the user
inspects, not gates that affect the verdict.

- Capacity is a property of the portfolio implementation, not of the
  factor signal. A factor with concentrated long-leg α can be
  excellent for a $10M book and unusable for a $10B book; factrix
  cannot know which the user is sizing for.
- Linear cost models (the breakeven formulation) hold only at small
  scale. Above a few percent of average daily volume, market impact
  follows a square-root law (Almgren & Chriss, "Optimal Execution of
  Portfolio Transactions," *Journal of Risk* 2001 — cited as
  background; not in bibliography) and the linear approximation
  understates cost.
- The trading-cost-aware factor selection in
  [DeMiguel-Martin-Utrera-Nogales-Uppal
  2020][demiguel-martin-utrera-nogales-uppal-2020] internalises cost
  into the selection objective. factrix exposes the inputs
  (`net_spread`) but stops short of taking the selection decision —
  same reason as section 2.

---

## 4. No cross-market aggregation

factrix evaluates a factor on the panel the user supplies. It does
not provide a primitive for "is this factor globally pervasive across
N markets" beyond what the user can run by calling `evaluate()` per
market and comparing the returned `FactorProfile`s.

- The global-vs-local pricing debate is unsettled empirically
  ([Asness-Moskowitz-Pedersen 2013][asness-moskowitz-pedersen-2013]
  argues for substantial commonality; the Fama-French international
  evidence ([Fama-French 1993][fama-french-1993] and the 2012/2017
  international tests, the latter not in bibliography) shows local
  models price local cross-sections better). Either aggregation
  choice would foreclose a research path.
- Standardisation across markets has industry conventions
  (country-specific mean + global standard deviation in MSCI
  Barra-style models — not in bibliography) but no canonical academic
  answer. Picking one inside the library would impose a prior the
  user did not opt into.
- A separate cross-market layer can be built on factrix outputs
  without library support: each `FactorProfile.metrics` is
  serialisable and easy to combine externally.

---

## 5. BHY rather than Bayesian multiple-testing

factrix's `multi_factor.bhy` runs Benjamini-Yekutieli
([Benjamini-Yekutieli 2001][benjamini-yekutieli-2001]) FDR control
with the `c(m) = Σ 1/i` dependence correction. It does not implement
Bayesian alternatives such as
[Harvey-Liu 2020][harvey-liu-2020] "Lucky Factors" or
[Bryzgalova-Huang-Julliard 2023][barillas-shanken-2018] BMA SDF
search.

- BHY is a frequentist recipe with a deterministic threshold given
  `(p, m)`. It composes cleanly with downstream business logic that
  needs a yes/no answer — the user reports `bhy_passed` to a
  governance step without negotiating priors.
- Bayesian methods integrate dependence structure more naturally and
  can yield posterior probabilities of being a true factor. The cost
  is an explicit prior over factor returns and a sampler / variational
  fit per evaluation. factrix's lean-dependency policy (numpy + polars
  only in the hot path) does not pay this cost.
- The case that BHY is *empirically* close enough is supported by
  Chordia, Goyal & Saretto (2020) "Anomalies and False Rejections,"
  *Review of Financial Studies* 33(5) (cited as background; not in
  bibliography) and
  [Jensen-Kelly-Pedersen 2023][jensen-kelly-pedersen-2023]: under
  realistic factor pools, BHY rejects the marginally significant
  exactly where Bayesian methods do, and the disagreement is
  concentrated in the corners of the decision space where neither is
  decisive.
- Users who want Bayesian inference can run it externally on the
  per-factor `MetricOutput.stat` series; factrix does not block that
  path.

---

## 6. Panel-aware FM rather than pooled OLS

The `Individual × Continuous` cell with `Metric.FM` runs Fama-MacBeth
([Fama-MacBeth 1973][fama-macbeth-1973]) with NW HAC at stage 2
([Newey-West 1987][newey-west-1987]), not pooled OLS with clustered
SE.

- Pooled OLS understates SE in the time-effect-dominant panels where
  factrix factor evaluations typically run
  ([Petersen 2009][petersen-2009]). The FM aggregation order — date
  cross-section first, then time-series t — is robust to
  contemporaneous cross-sectional dependence by construction.
- The future extension to two-way clustering
  ([Cameron-Gelbach-Miller 2011][cameron-gelbach-miller-2011],
  [Thompson 2011][thompson-2011]) is acknowledged in the
  architecture; it is not the default because the typical factrix
  panel exhibits stronger date-correlation than asset-correlation,
  and the two-way correction adds complexity without changing the
  point estimate.
- `pooled_ols` is exposed as an explicit comparison metric so a user
  can see when FM and pooled disagree (the
  `fm_pooled_sign_mismatch`-style diagnostic surfaced by
  `profile.diagnose()`). This is information for the user, not a
  reason to default to pooled.

---

## 7. Per-metric registered procedures rather than a unified test

factrix dispatches each `Scope × Signal × Metric` cell to a
**registered procedure** that runs a fixed pipeline (sample guards →
cross-section step → significance test → diagnostics). It does not
combine cell-canonical scalars into a single procedure-level F-test
or χ² of the form "is anything in this profile significant?".

- A unified composite test forces a hypothesis structure that the
  underlying data does not have. IC and FM-λ test the same cell with
  different aggregation orders; combining them double-counts the
  same evidence under positive correlation.
- Event-study tests ([Brown-Warner 1985][brown-warner-1985],
  [MacKinlay 1997][mackinlay-1997]) and cross-sectional tests have
  fundamentally different null hypotheses. A unified statistic that
  blends them would obscure which null actually rejected.
- Registered procedures keep the per-metric audit trail readable.
  A reviewer can see exactly which test fired, on what subsample,
  with which lag rule — the same auditability principle that motivates
  pre-registered analysis plans more broadly
  ([Harvey 2017][harvey-2017]).
- The cost is that factrix produces a vector of evidence rather than
  one number. The verdict layer reduces the vector to a binary — but
  the vector remains accessible for users who need to inspect the
  components.
