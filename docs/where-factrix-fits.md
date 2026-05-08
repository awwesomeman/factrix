# Where factrix fits

This page is the depth companion to the README's "Where factrix fits"
block. It assumes you have read the hero claim and routing table on
the README; here we expand the design philosophy, walk through the
pipeline and internals, draw scope boundaries, compare against
same-purpose peers, show adjacent-tool integration, and disclose
honest weaknesses.

## 1. What factrix is

factrix is a **factor verdict surface**: given a candidate factor and
a forward return, it answers *is the predictive power real?* and
returns a structured profile of evidence. It is the first Python
framework to dispatch primary statistical tests by factor type rather
than applying one uniform formula to every factor.

Three factor types each get a primary test fitted to their
data-generating process:

- **Cross-sectional factors** — Information Coefficient (IC) and
  Fama-MacBeth (FM), both with Newey-West HAC standard errors and a
  Hansen-Hodrick lag floor for overlapping forward returns.
- **Event factors** — Cumulative Average Abnormal Return (CAAR) on
  the dense event-time calendar, with NW HAC inference and an
  overlap diagnostic when consecutive events sit within twice the
  forward horizon.
- **Common factors** — a factor whose realisation is shared across
  the cross-section in a given period (Fama-French market / size /
  value, or a macro variable). factrix tests these as a panel
  exposure, falling back to single-series β when only one asset is
  available.

Each type also runs a multi-metric *diagnostic battery* — never
collapsed into a single score. This is a deliberate design choice,
not an oversight. A composite score becomes its own optimisation
target the moment it ships (Goodhart 1984), and weighted aggregation
across heterogeneous nulls implicitly prices each null
(DeMiguel-Garlappi-Uppal 2009; Harvey 2017). See
[design notes §1](development/design-notes.md#1-no-composite-factor-score)
and
[§7](development/design-notes.md#7-per-metric-registered-procedures-rather-than-a-unified-test)
for the full citation chain.

## 2. Where factrix sits

### 2.1 Ecosystem pipeline

factrix is **Stage 1** of a multi-stage workflow. It is not a
competitor to portfolio construction, backtesting, or execution
tools — it sits upstream of them and produces the input they assume.

```mermaid
flowchart LR
    DATA[Raw data] --> CONSTR[Factor construction<br/>zipline Pipeline · self-roll]
    CONSTR --> FX[<b>factrix verdict</b><br/>Stage 1 — kill fakes]
    FX --> PORT[Portfolio construction<br/>skfolio · PyPortfolioOpt · riskfolio-lib]
    PORT --> BT[Backtest<br/>vectorbt · zipline-reloaded · bt]
    BT --> LIVE[Live execution<br/>lumibot · nautilus_trader]
    classDef here fill:#3670A0,color:#fff,stroke:#234060,stroke-width:2px;
    class FX here
```

### 2.2 factrix internals

Inside factrix the call graph is small. A factor panel plus an
`AnalysisConfig` enters `evaluate()`, which routes to the registered
procedure for the (scope, signal, metric, mode) cell, runs the
primary test plus diagnostics, and returns a `FactorProfile`.
Multiple profiles flow into `multi_factor.bhy()` for cross-test FDR
control on the surviving subset.

```mermaid
flowchart LR
    PANEL[Factor panel<br/>+ forward return] --> EVAL[evaluate panel cfg]
    CFG[AnalysisConfig<br/>scope · signal · metric] --> EVAL
    EVAL --> DISP{Type-routed<br/>dispatch}
    DISP -->|individual_continuous| CS[IC + FM<br/>+ diagnostics]
    DISP -->|individual_sparse| EV[CAAR<br/>+ diagnostics]
    DISP -->|common_continuous| CO[ts_beta<br/>+ diagnostics]
    CS --> PROF[FactorProfile<br/>stats · primary_p · verdict]
    EV --> PROF
    CO --> PROF
    PROF --> BHY[multi_factor.bhy<br/>FDR within family]
    BHY --> SURV[Surviving profiles]
```

The dispatch arrow is the single line that distinguishes factrix
from peers that apply one uniform formula across factor types
(see [§4](#4-vs-same-purpose-peers)).

## 3. Scope boundaries

What factrix deliberately does **not** do, and the canonical tool for
each. This is a commitment, not a TODO. To expand factrix scope,
update
[ARCHITECTURE.md Invariants](development/architecture.md#invariants)
first.

| Out of scope | Use instead |
|---|---|
| Portfolio optimisation (MVO / HRP / risk parity) | [skfolio](https://skfolio.org/), [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt), [riskfolio-lib](https://github.com/dcajasn/Riskfolio-Lib), cvxpy |
| ML signal layer | xgboost + shap |
| Regime detection methodology (HMM / threshold) | [hmmlearn](https://hmmlearn.readthedocs.io/), self-roll |
| Structural break detection (Chow / Bai-Perron) | [ruptures](https://centre-borelli.github.io/ruptures-docs/) |
| GARCH / wild-bootstrap SE | [arch](https://github.com/bashtage/arch) |
| Persistent-predictor auto-correction (IVX / Stambaugh) | [arch](https://github.com/bashtage/arch), R `ivx` (factrix flags via ADF; does not auto-correct) |
| Backtest / execution / slippage / margin | [vectorbt](https://github.com/polakowo/vectorbt), [bt](https://github.com/pmorissette/bt), [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded), backtrader |
| Intraday / HFT (tick-level) | dedicated tooling |
| Cross-factor signal combiner | self-roll, scikit-learn |
| Composite factor scoring across dimensions | [AlphaEval](https://github.com/LeoDingggg/AlphaEval) (different design philosophy — see [§4.4](#44-vs-alphaeval)) |
| Deflated / probabilistic / Haircut Sharpe | [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) (commercial); roadmap gap for factrix — see [§7](#7-honest-weaknesses) |
| Cross-sectional factor *construction* DSL | [zipline-reloaded Pipeline](https://github.com/stefan-jansen/zipline-reloaded); factrix consumes Pipeline output |
| Returns-level tear-sheet (downstream of factrix) | [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded) |

### 3.1 Rationale for the controversial rows

Three rows surprise readers most often. Their rationale is anchored
in design notes rather than restated here.

**Composite factor scoring** — rejected for the reason a single
weighted score becomes its own target the instant it ships
(Goodhart 1984), and weighted aggregation across heterogeneous
nulls implicitly prices each null without disclosing the price.
factrix exposes per-metric pass/fail and keeps the user in the
inference loop. See
[design notes §1](development/design-notes.md#1-no-composite-factor-score).

**ML signal layer** — out of scope as a deliberate boundary. The
signal-generation problem is well served by xgboost + shap, and
folding model fit into factrix would change the page's hero claim
from "verdict on a hypothesised factor" to "verdict on a fitted
model" — those need different statistical machinery (cross-validation
schemes, leakage tests). qlib already covers the integrated
pipeline; we leave that branch to qlib.

**Persistent-predictor flagging only, not auto-correction** —
when the cross-sectional or time-series predictor is highly
persistent (ADF p > 0.10), factrix raises
`PERSISTENT_REGRESSOR` and notes that the β estimate may carry
Stambaugh (1999) bias. It does not silently swap in IVX
(Phillips-Magdalinos), Stambaugh-correction, or sign-restricted
inference, because the right correction depends on the
researcher's economic prior — IVX assumes a near-unit-root
predictor; Stambaugh requires a specified innovation-correlation
sign. Auto-correcting would mask the modelling choice. Reach for
[arch](https://github.com/bashtage/arch) or R `ivx` when the flag
fires.
