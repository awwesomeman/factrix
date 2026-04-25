# Changelog

All notable changes to **factrix** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR**: API-breaking changes (field rename, signature change, removed metric)
- **MINOR**: New metrics, new Profile fields, new optional parameters
- **PATCH**: Bug fixes, docstring/test fixes, internal refactors

While the version is below `1.0.0`, the public API should be considered
unstable — breaking changes may occur in **MINOR** bumps. Consumers are
expected to pin by commit SHA (e.g., via git submodule) rather than relying
on semver range constraints until `1.0.0` is cut.

---


### Note
First standalone release. Extracted from the `awwesomeman/factor-analysis`
research workspace via `git filter-repo`; 149 commits of prior development
history are preserved in this repository's git log.

Earlier version tags (`3.x`) existed only in the originating workspace and
are not reproduced here — version numbering restarts from `0.1.0` to
honestly reflect that the API is still iterating. The pre-extraction
snapshot is anchored in the source workspace as the tag
`pre-extraction-backup`.

### State at extraction
- **555 tests passing** across four factor types (cross-sectional, event
  signal, macro panel, macro common)
- `fl.evaluate` / `fl.evaluate_batch` / `fl.factor()` public API
- Typed `Profile` dataclasses (`frozen=True, slots=True`) with binary
  `verdict()` (PASS / FAILED) and `diagnose()` diagnostic list
- `ProfileSet` container with BHY multiple-testing correction
  (`multiple_testing_correct`)
- Artifacts retention via `return_artifacts=True` and `evaluate_batch(keep_artifacts=, compact=)`
- External rule registration: `register_rule` / `clear_custom_rules`
- `on_result` callback supports `bool | None` early-stopping
- Level 2 helpers integrated into Profile pipeline: `regime_ic`,
  `multi_horizon_ic`, `spanning_alpha`, orthogonalization
- Shared tradability primitives (`turnover` / `breakeven_cost` /
  `net_spread`) hoisted to `Factor` base class with per-call override
  support

### Known caveats
- `examples/demo.ipynb` stored outputs may reflect earlier quantile-spread
  field names (`q1_q5_spread`, then `long_short_spread` — rename landed
  2026-04-20). Current field is `quantile_spread` / `spread_tstat` /
  `spread_p`; rerun the notebook to regenerate outputs against live code.
- `Factor Signal Analyzer` positioning: `turnover` / `breakeven_cost` /
  `net_spread` are idealized proxies (equal-weight, zero slippage) and do
  not represent tradable returns

## v0.4.0 (2026-04-25)

Trading-cost arithmetic overhaul: separates rank-stability turnover
from notional position turnover, fixes per-period vs per-rebalance
unit mismatch in the bps formulas, and renames `turnover_jaccard` →
`notional_turnover` to describe the concept (Novy-Marx & Velikov τ)
rather than the implementation (Jaccard set-similarity).

### BREAKING CHANGE

- `breakeven_cost` / `net_spread` now require `forward_periods`
  (kw-only) so per-period spread and per-rebalance turnover stay on
  the same time scale. Without it, h ≥ 2 factors were over-charged
  by N× and breakeven understated by N×. No default — a default of
  1 would silently reproduce the buggy answer.
- `breakeven_cost` / `net_spread` numeric values shift on every
  factor that previously hit the rank-stability-turnover bug
  (CrossSectionalProfile from prior release; MacroPanelProfile and
  `Factor.evaluate()` in this release). Direction is optimistic for
  CS (breakeven rises) and pessimistic for MP (breakeven falls,
  because the rank-stability turnover overstated churn). Consumers
  comparing against stored thresholds must re-calibrate.
- `MacroPanelProfile` gains a required `notional_turnover: float`
  field; direct kwarg-construction breaks until callers add it.
- Identifier rename across the public API:
  `turnover_jaccard` → `notional_turnover` (primitive, MetricOutput
  name / cache key, `Factor` session method, Profile dataclass fields
  on both CS and MP, public export). Migration is mechanical
  find-and-replace.
- Rule code rename: `cs.high_turnover_jaccard` →
  `cs.high_notional_turnover`; `macro_panel.high_turnover_jaccard` →
  `macro_panel.high_notional_turnover`.

### Feat

- **tradability**: `notional_turnover` (Novy-Marx & Velikov 2016 τ)
  separated from rank-stability `turnover` (1 − Spearman ρ). The two
  measure different things — middle-rank shuffling counts as turnover
  but not as notional churn — so only `notional_turnover`'s units
  align with the bps cost arithmetic. (Originally landed as
  `turnover_jaccard` in 2d005ff; renamed in this release.)
- **tradability**: `Factor.notional_turnover()` session method with
  the standard `n_groups` override + cache shape; mirrors
  `quantile_spread`. `n_groups` override on `breakeven_cost` /
  `net_spread` now also reroutes the turnover bucketing so spread
  and turnover stay consistent during a sensitivity sweep.
- **tradability**: `notional_turnover` exported from
  `factrix.metrics.__all__` (the prior `turnover_jaccard` name was
  never threaded into the public surface).
- **diagnostics**: `cs.high_notional_turnover` and
  `macro_panel.high_notional_turnover` rules (severity `warn`,
  threshold `notional_turnover > 0.5`). Sibling to the existing
  `cs.high_turnover`; both rules can fire independently — a factor
  with high mid-rank noise but stable Q1/Qn has high `turnover` yet
  low `notional_turnover` (still implementable).

### Fix

- **tradability**: per-period vs per-rebalance unit mismatch in the
  bps formulas. `gross_spread` (per-period) was being subtracted
  from `2·cost·turnover` (per-N-period rebalance) — different time
  scales. `breakeven_cost ×= forward_periods`, `net_spread`'s
  `cost_drag /= forward_periods` to align both sides on the
  per-period scale.
- **tradability**: `MacroPanelProfile.from_artifacts` and
  `Factor.breakeven_cost` / `Factor.net_spread` were still feeding
  rank-stability `turnover` into the bps formulas despite the
  primitive's docstring forbidding it. Routed through
  `notional_turnover` instead — completes the wiring that 2d005ff
  applied only to CrossSectionalProfile.
- **tradability**: `MacroPanelProfile.turnover` is now sampled at
  `config.forward_periods` stride (was defaulting to lag-1).
  Mirrors `CrossSectionalProfile`. Diagnostic-only field; bps
  formulas are unaffected (they consume `notional_turnover`).

### Refactor

- **tradability**: rename `turnover_jaccard` → `notional_turnover`
  throughout the public API. Renamed mechanically; no logic change.
  See BREAKING above for migration surface.

## v0.3.0 (2026-04-23)

### BREAKING CHANGE

- callers passing adf_check=False must migrate to
adf_threshold=None; callers passing adf_check=True can drop the
argument (default unchanged) or pass adf_threshold=0.10.
- metadata n_dates -> n_pairs; short-circuit reason
no_valid_rank_autocorrelation -> insufficient_pairs.

### Feat

- **mfe-mae**: expose min_estimation_samples kwarg
- **trend**: adf_threshold replaces adf_check bool
- **hit-rate**: exact binomial p-value for small samples
- **concentration**: add alpha-contribution hhi variant
- **ic-trend**: flag unit-root suspicion via adf pre-check
- **ic,caar**: bhy per bucket; horizon-aware sample guards
- **stats**: add stationary bootstrap utility
- **mfe-mae**: normalize excursions by estimation sigma
- **spanning**: flag inflated t-stats from forward selection
- **quantile-spread**: lag vw weights by default
- **bmp-test**: add kolari-pynnönen cross-sectional adjustment
- **stats**: enforce forward_periods floor on newey-west
- **fama-macbeth**: shanken eiv + two-way cluster
- **profileset**: warn on raw-p batch decisions
- **metrics**: align turnover with forward horizon

### Fix

- **trend**: short-circuit nan-only ic series before lstsq

### Refactor

- **ic**: drop parallel list in regime_ic bhy step
- **_types**: centralise metric option literals

## v0.2.0 (2026-04-21)

### BREAKING CHANGE

- consumers must update `import factorlib` →
`import factrix` and `pip install factorlib` → `pip install factrix`.
- prepared panels now carry _fl_preprocess_sig (String)
instead of _fl_forward_periods (Int32). Downstream code reading that
specific column name must update.
- fl.datasets.make_cs_panel(..., forward_periods=5)
and fl.datasets.make_event_panel(..., forward_periods=5) must be
updated to signal_horizon=5. Package is at 0.1.0 with no external
users.

### Feat

- **profileset**: add diagnose_all, with_canonical, and layered logging
- **profiles**: add PASS_WITH_WARNINGS verdict and alternative p-values
- **preprocess**: widen strict gate to all preprocess-time fields
- **datasets**: add synthetic CS / event panels with calibrated IC

### Fix

- **metrics**: guard ts_beta_sign_consistency at N<2
- **preprocess,evaluate**: strict-gate safety + fallback visibility

### Refactor

- rename package from factorlib to factrix
- drop streamlit; lean pyproject deps; editable install
- **datasets**: rename forward_periods to signal_horizon

## v0.1.0 (2026-04-20)

### BREAKING CHANGE

- describe_profile_values(profile, artifacts, *,
include_detail) -> describe_profile_values(profile). Call-sites in
tests / demo / README / docstrings updated to the single-arg form.
Demo notebook rerun end-to-end: 56/56 cells green.
- EventProfile field bmp_sar_mean renamed to
bmp_test_mean; metric_outputs cache keys "mfe_mae" -> "mfe_mae_summary"
and "bmp_sar" -> "bmp_test".
- Artifacts.metric_outputs cache keys renamed from
"ic_trend" to "caar_trend" (EventProfile) and "beta_trend"
(MacroPanel/MacroCommon). CrossSectionalProfile unaffected.
- Profile fields long_short_spread / long_short_spread_vw
renamed to quantile_spread / quantile_spread_vw. Cache keys and
MetricOutput.name identifiers renamed in lockstep.
- to test a different forward_periods, rebuild the
Factor session with a new config (fl.preprocess + fl.factor).
- q1_q5_spread → long_short_spread, q1_concentration →
top_concentration, plus intermediate columns q1_return/q5_return →
top_return/bottom_return. Old names hard-coded Q1/Q5 but n_groups is
configurable (CS default 10) — a quant reading q1_q5_spread under
n_groups=10 was literally wrong. Prose / charts / diagnose messages
propagated.
- metrics/oos.multi_split_oos_decay returns MetricOutput
(was OOSResult); event_around_return/multi_horizon_hit_rate/mfe_mae_summary
return short-circuit MetricOutput instead of None.
- `validate_factor_data` previously required the date
column to be exactly `pl.Datetime("ms")` naive; it now accepts `pl.Date`
or any `pl.Datetime(time_unit, time_zone)` variant. Callers relying on
the strict-ms rejection must pre-cast themselves.
- `CrossSectionalConfig.q_top` removed. `q1_concentration`
now uses `q_top=1/n_groups` so its Q1 bucket matches the Q1 in
`q1_q5_spread`. Previously the default `n_groups=10` + `q_top=0.2` made
the two Q1 metrics disagree (concentration used top 20%, spread used
top 10%) — a silent inconsistency that users had to notice and tune out.
- `Profile.from_artifacts(artifacts)` now returns
`tuple[Self, dict[str, MetricOutput]]` instead of `Self`. Direct callers
(subclass authors, test helpers) must tuple-unpack.
- BaseConfig.multi_horizon_periods removed.
- CrossSectionalConfig.orthogonalize removed;
CrossSectionalProfile.orthogonalize_applied removed;
cs.orthogonalize_not_applied diagnose rule removed.

### Feat

- **reporting**: normalize describe_profile type column and NaN labels
- **factor**: override advisory via UserWarning
- **metrics**: tie_policy diagnostic and short-circuit NaN
- **preprocess**: embed forward_periods marker
- **factor**: add event/macro factor session subclasses
- **factor**: add fl.factor() session API and unify MetricOutput contract
- **evaluation**: date dtype consistency check at join boundaries
- **adapt**: auto-promote pl.Date to pl.Datetime(ms)
- **reporting**: describe_profile_values drill-down api
- **stats**: support two-stage screening BHY with n_total
- **adapt**: support OHLCV canonical renames
- **profiles**: wire Level 2 metrics into CrossSectionalProfile (T3.S2)
- **preprocess**: wire orthogonalize into the CS pipeline (T3.S1)
- **profile_set**: add with_extra_columns for user-defined columns
- **api**: allow on_result callback to signal early stop
- **diagnostics**: support register_rule for external rule injection
- **api**: support keep_artifacts and compact in evaluate_batch
- **api**: expose artifacts via evaluate(return_artifacts=True)
- **profiles**: surface insufficient data via profile field and diagnose rule
- **metrics**: structure short-circuit metadata with reason/n_observed
- **factorlib**: add multi-horizon event metrics and docs
- **factorlib**: add event_ic metric for continuous signal strength
- **factorlib**: implement event_signal factor type
- **factorlib**: implement macro_common factor type (Future B)
- **factorlib**: implement macro_panel factor type (Future A)
- **factors**: add generate_price_intention factor
- **adapt**: accept pandas DataFrame and add fill_forward option
- add split_by_group utility and comparison charts
- **factors**: add industry dummy encoding for orthogonalization
- **charts**: add reusable Plotly chart builders for factor analysis
- **ic**: expose std_ic in regime_ic metadata for downstream charts
- **regression**: add spanning regression with greedy forward selection
- **experiment**: add log_evaluation for gate-based MLflow logging
- **gates**: add gate pipeline for factor evaluation
- **tools**: add modular tools layer and preprocessing pipeline
- bootstrap factor analysis framework

### Fix

- **demo**: complete P2/COV coverage gaps
- **demo**: regime shift(1) and market_cap size proxy
- **api**: raise on silent config / factor_type / compact misuse
- **metrics**: mirror input date dtype in event output frames
- **profile_set**: make runtime-guard error actionable
- **redundancy**: surface degenerate inputs instead of silently zeroing
- **profile_set**: re-check canonical invariant at runtime BHY call
- **profiles**: verdict uses t-distribution; extract _diagnose helper
- **protocol**: _CompactedPrepared blocks every container/truth-testing path
- **metrics**: redundancy_matrix uses mean(|rho|) not |mean(rho)|
- **factorlib**: show gate results in EvaluationResult repr
- **factorlib**: use t+1 entry for forward return computation
- **factorlib**: improve UX/DX after routing refactor
- **concentration**: change t-stat to test H₀: ratio ≥ 0.5
- **tools**: fix OOS median, spanning null-fill, quantile ties, and add guards

### Refactor

- **metrics**: adopt _short_circuit_output helper at all sites
- **reporting**: drop artifacts arg from describe_profile_values
- **factor**: unify mfe_mae_summary and bmp_test names
- **factor**: unify caar_trend / beta_trend names
- **profile**: unify on quantile_spread across API
- **factor**: drop forward_periods= override
- **factor**: consolidate shared helpers from review pass
- **api**: drop Factor base class from top-level __all__
- **profile**: rename q1 fields and polish factor session
- **metrics**: extract single-asset ts_beta fallback helper
- **api**: require preprocessed input (strict gate)
- **validation**: accept any Datetime time_unit / timezone
- **config**: drop q_top, derive from n_groups
- **evaluation**: stash MetricOutput, pure from_artifacts tuple return
- **config**: add OrthoConfig sub-dataclass + DataFrame shortcut
- **api**: extract _evaluate_one shared helper
- **phase1-2**: apply review findings
- **profiles**: remove orthogonalize false provenance
- **profile_set**: resolve empty-set dtypes via typing.get_type_hints
- drop gate-era evaluation layer for FactorProfile API
- **tests**: hoist make_macro_panel helper into conftest
- rename oos_decay to oos_survival_ratio
- **profiles**: restore config-aware diagnose rules via profile fields
- **api**: drop keep_artifacts no-op param and delete dead metric dicts
- **factorlib**: add top-level profile-era API and reflect describe_profile
- **factorlib**: add ProfileSet, multiple_testing module, redundancy_matrix
- **factorlib**: add diagnostic rules module and Artifacts compact mode
- **factorlib**: add per-type Profile dataclasses with canonical-p whitelist
- **factorlib**: add FactorProfile protocol and registry scaffold
- **metrics**: split event metrics by statistical question
- **factorlib**: unify quantile spread, add discoverability, fix cross-type consistency
- **factorlib**: add pipeline routing, API layer, and clean break exports
- **factorlib**: add FactorType enum, per-type config, and rich protocol types
- **factorlib**: restructure packages per v8 routing design
- **tools**: generalize statistics to p-value driven significance
- **tools**: implement P1 metric refactoring
- unify naming to snake_case and align with return types
- **ic**: split IC and IC_IR into separate metrics
- **preprocess**: accept PipelineConfig and preserve date dtype
- **preprocess**: rename preprocessing to preprocess
- unify canonical column names (date/asset_id/price) and add adapt()
- **gates**: publicize build_artifacts as standard API
- **preprocessing**: rename to preprocess_cs_factor and pass through price
- **tools**: use academic significance markers (***/**/*)
- **dashboard,scoring**: rename dimensions and enhance dashboard layout
- restructure project into modular packages
- extract scoring into package with registry pattern

### Perf

- **stats**: cache bhy correction factor
- **preprocess**: use partition_by for per-date OLS
- **redundancy**: vectorize pairwise_abs_spearman via numpy.corrcoef
- **profile_set**: column-wise DataFrame build and tidy error f-string
