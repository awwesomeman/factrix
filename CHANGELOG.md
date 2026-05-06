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

## [Unreleased]

PRs accumulate WHY-narrative bullets here under `### Added` / `### Changed` /
`### Fixed` / `### Migration`. A release commit (`cz bump --changelog`) renames
this section to the next versioned heading and adds a fresh `[Unreleased]`
above it. This decouples per-PR cadence from per-tag cadence — see
CONTRIBUTING §7 (Release workflow).

### Added

- **`bmp_test(include_prediction_error_variance=False)`** — opt-in
  strict BMP (1991) denominator $\sigma_i \cdot \sqrt{1 + 1/T_{\mathrm{est}}}$
  for the mean-adjusted residual forecast. Default off preserves the
  prior simplified denominator. Under a single ``estimation_window``
  the correction scales every SAR by the same constant, so
  ``mean_SAR`` / ``std_SAR`` shrink by the same ratio but the $z$
  statistic is invariant; the flag documents the strict standardiser
  rather than moving inference. Per-event $T_i$ variation (which
  would move $z$) requires a market-model extension and remains out
  of scope. (#48)
- **`WarningCode.SPARSE_MAGNITUDE_WEIGHTED`** — emitted by
  `compute_caar` (as a `UserWarning` on the primitive) and surfaced
  in `FactorProfile.warnings` from the `(INDIVIDUAL, SPARSE, PANEL)`
  / `(COMMON, SPARSE, PANEL)` procedures when the sparse `factor`
  column is mixed-sign and not a clean ±1 ternary
  (e.g. `{-2.5, 0, +1.3}`). The CAAR / sparse-panel statistic in
  that regime is the Sefcik-Thompson (1986) magnitude-weighted
  variant rather than the MacKinlay (1997) signed CAAR — a different
  estimator at finite samples when negative- and positive-leg vols
  disagree. ``{-1, 0, +1}`` does not trigger (sign and weight
  coincide numerically); all-non-negative inputs do not trigger (no
  flip ambiguity). Helper `_is_sparse_magnitude_weighted` (single
  `.unique()` + sign distribution) is the shared check. (#48)
- **`compute_ic` per-date `tie_ratio` column** — output schema widened
  from `(date, ic)` to `(date, ic, tie_ratio)`, where
  `tie_ratio = 1 - n_unique / n` per date in `[0, 1]`. Aggregated as
  the median across dates and surfaced via
  `MetricOutput.metadata["tie_ratio"]` for `ic`, `ic_newey_west`, and
  `ic_ir`. Motivation: at high tie rates Spearman ρ on average ranks
  is biased relative to the tie-corrected formula (Kendall-Stuart
  §31), and the previous primitive contract gave callers no way to
  detect bucketed / categorical signals without re-inspecting the
  raw input. Parallels the existing `top_concentration` tie diagnostic.
  **Migration:** code that asserts the exact column list of
  `compute_ic` output (e.g. `df.columns == ["date", "ic"]`) needs to
  accept the third column; column-by-name access is unaffected. (#48)
- **`WarningCode.SMALL_CROSS_SECTION_N`** + **`BORDERLINE_CROSS_SECTION_N`**
  — emitted by the `common_continuous` PANEL procedure and by
  `suggest_config` based on `n_assets`. `2 ≤ n_assets < 10` → SMALL
  (df=`n_assets`-1 ≤ 8, t_crit inflated 18%–548% vs asymptotic 1.96);
  `10 ≤ n_assets < 30` → BORDERLINE (residual inflation 5%–15%);
  `n_assets ≥ 30` → no warning. Two-tier mirrors the existing
  `n_periods` structure (`MIN_PERIODS_HARD` / `MIN_PERIODS_RELIABLE`).
  SMALL implies BORDERLINE so only the more severe code emits per
  profile. Procedure still runs at all `n_assets ≥ 2` — warnings
  surface the inference-power decay rather than blocking execution.
  (#17)
- `MIN_ASSETS = 10` and `MIN_ASSETS_RELIABLE = 30` constants in
  `factrix/_stats/constants.py`, alongside `MIN_PERIODS_HARD` /
  `MIN_PERIODS_RELIABLE`. Naming deliberately omits `_HARD` for
  `MIN_ASSETS` because the `n_assets` axis only warns — re-using the
  `n_periods` `_HARD` (which means "raise") would mislead. (#17)
- **`factrix.metrics`** module docstring gains a fifth category,
  Time-Series / Standalone Diagnostic, listing the `ts_beta` family +
  `ts_quantile_spread` + `ts_asymmetry`. `help(factrix.metrics)` now
  surfaces them; previously the docstring categorisation hid them
  despite being fully exported. (#19)
- **`SuggestConfigResult.detected: dict[str, Any]`** — new field
  carrying the structured panel observations behind the suggestion
  (`scope`, `signal`, `mode`, `n_assets`, `n_periods`, `sparsity`).
  All keys always present, type-stable. AI agents and pipeline gates
  branch on these without parsing the `reasoning` strings or
  re-deriving observations from the raw panel. `reasoning`
  (human-readable narrative) and `warnings` unchanged. (#21)

### Changed

- **`examples/` reorganised from monolithic demo into focused per-recipe
  notebooks.** `demo.py` (369-line `# %%`-script) and `demo.ipynb`
  (separately generated by a 531-line `scripts/build_demo.py`) collapsed
  into hand-edited SSOT notebooks: `multi_factor_screening.ipynb` (BHY
  family partitioning + cross-family pitfall — only learnable from
  example) and `stock_factor_evaluation.ipynb` (`individual_continuous`
  IC, ~80% use case). Each notebook's first markdown cell carries a
  `Factor type` block (factory call + three-axis enum values) and the
  standardised three-label header (`Use this when` / `What it tests` /
  `Output to read`). `scripts/sync_examples.py` regenerates
  `docs/examples/index.md` from each notebook's H1 + first
  `Use this when` bullet, eliminating the hand-maintained listing as a
  drift source. CI gains an `examples` job that executes every recipe
  via `nbconvert --execute` so API drift breaks the build before
  reaching docs. (#14)
- **Two-tier sample-size guards on `fama_macbeth`, `caar`, and
  `top_concentration`.** Each now distinguishes a math-validity floor
  (``_HARD`` → short-circuit to NaN ``MetricOutput``) from a
  literature/power floor (``_WARN`` → return the stat with a Python
  ``UserWarning`` and the relevant ``WarningCode.value`` surfaced in
  ``metadata["warning_codes"]``). Pre-#48 these primitives short-
  circuited at a single conservative threshold, refusing to report
  anything in the borderline regime — UX regression every user hit
  when fewer than ~30 events / periods were available even though the
  math was perfectly defined. The new contract is *warn, don't
  refuse*. Constants: ``MIN_FM_PERIODS = 20`` →
  ``MIN_FM_PERIODS_HARD = 4`` / ``MIN_FM_PERIODS_WARN = 30``;
  ``MIN_EVENTS = 10`` → ``MIN_EVENTS_HARD = 4`` /
  ``MIN_EVENTS_WARN = 30`` (Brown-Warner 1985 convention);
  ``MIN_PORTFOLIO_PERIODS = 5`` → ``MIN_PORTFOLIO_PERIODS_HARD = 3``
  / ``MIN_PORTFOLIO_PERIODS_WARN = 20``. Two new ``WarningCode``
  values: ``FEW_EVENTS_BROWN_WARNER`` (caar) and
  ``BORDERLINE_PORTFOLIO_PERIODS`` (top_concentration); FM reuses the
  existing ``UNRELIABLE_SE_SHORT_PERIODS``. Descriptive metrics
  (``clustering`` / ``corrado`` / ``event_horizon`` / ``event_quality``
  / ``mfe_mae`` / ``quantile`` / ``ts_quantile`` / ``ts_asymmetry``)
  switch to the new ``_HARD`` constant only — they have no formal H0
  so the WARN tier would be noise; they now accept smaller-n inputs
  they previously refused, by design. (#48)
- **`multi_split_oos_decay` is descriptive-only.** ``stat`` is
  ``None`` (was already), and ``metadata["p_value"]`` is now omitted
  entirely (was ``1.0``) — the multi-split decomposition (``per_split``
  + ``sign_flipped`` + ``status``) is the message, and a t-stat at
  ``MIN_OOS_PERIODS = 5`` would have power ≈ 0. Dropping ``p_value``
  prevents callers from accidentally routing the diagnostic into BHY
  / gate logic that expects a probability. ``MIN_OOS_PERIODS`` stays
  single-tier (no HARD/WARN split needed when there is no hypothesis
  test). (#48)
- **`compute_caar` per-row formula: `return × sign(factor)` →
  `return × factor`.** Magnitude is now preserved as a weight rather
  than being silently dropped via `.sign()` coercion. `{0, 1}` and
  `{-1, 0, +1}` callers see no behaviour change (sign was identity);
  `{0, R}` non-ternary callers — previously flagged by
  `WarningCode.SPARSE_MAGNITUDE_DROPPED` as wrong — now get the
  magnitude-weighted statistic they were trying to compute. Callers
  wanting ternary semantics on a non-ternary input apply `.sign()` to
  the input column themselves before calling. See `compute_caar`
  docstring for the input-form behaviour table. (#12)
- README §樣本守門 重寫：新增「factory × `n_assets` regime 行為矩陣」表 +
  「計算順序對照」段 + 「兩軸守門對稱」表，明確區分
  `individual_continuous`（cross-section first → time-series）與
  `common_continuous`（time-series first → cross-asset）的計算順序差異——
  使用者誤以為兩者皆「先橫斷面再時序」是 `common_continuous` N=1 退化
  與 small-`n_assets` 結果不可信的根因。修正先前 L247「`n_assets` < 10
  切 FM」誤導建議——FM 在 `n_assets` = 2..9 同樣不可靠。 (#16)
- ARCHITECTURE.md 增補 §Cross-sectional guards (`n_assets`)（two-tier
  threshold 設計理由 + t_crit 衰減表）與 §Procedure pipelines（每個 PANEL
  continuous procedure 的計算管線、small-`n_assets` failure mode、threshold
  對應），把行為矩陣背後的 statistical rationale 集中到一處。 (#16)
- **`MIN_IC_PERIODS` → `MIN_ASSETS_PER_DATE_IC`** (in `factrix/_types.py`).
  The "PERIODS" suffix was misleading — the value has always been
  checked against per-date asset counts, not period counts. **Migration:**
  update the import; no deprecation alias kept (pre-1.0 + single-consumer
  convention; the factor-analysis workspace pins by SHA). (#19)
- **`WarningCode.UNRELIABLE_SE_SHORT_SERIES` → `UNRELIABLE_SE_SHORT_PERIODS`**.
  Vocabulary aligned with the `n_periods` parameter name canonicalised in
  PR #16. Both Python identifier and serialised string value change to
  `"unreliable_se_short_periods"`. **Migration:** update imports + any
  string-based filters / log queries that match the old serialised value;
  no alias kept. (#19)

### Fixed

- **`(INDIVIDUAL, SPARSE, None, PANEL)` NW HAC lag rule.** The procedure
  previously fed `compute_caar`'s event-date-indexed series straight into
  NW HAC, but the `forward_periods - 1` lag floor assumes consecutive
  observations are 1 calendar period apart. On the event-only filtered
  series that assumption breaks: sparse events (calendar gap >
  `forward_periods`) over-corrected an MA(h-1) overlap that did not
  exist (deflating t / inflating p); clustered events
  (gap < `forward_periods`) under-corrected the real overlap structure
  (inflating t / deflating p). The procedure now reindexes the CAAR
  series to the full calendar and zero-fills non-event dates before NW
  HAC — the **calendar-time portfolio approach** (Jaffe 1974; Mandelker
  1974; Fama 1998 §2). Mathematically the t-statistic is invariant to
  the dense reframing in the iid limit (`mean_dense × n_total =
  mean_event × n_event`), so the canonical p is unchanged where the lag
  rule was already valid; only the previously-biased regimes shift. All
  four NW-HAC PANEL procedures (IC / FM / CAAR / common-sparse) now run
  on calendar-dense series with the same `_resolve_nw_lags` machinery.
  **Output contract:** `FactorProfile.n_obs` and
  `StatCode.NW_LAGS_USED` now report the dense-series counts (was
  event-date counts); `StatCode.CAAR_MEAN` continues to report the
  per-event-date mean (user-facing statistic unchanged). (#24)
- **`(COMMON, SPARSE, None, PANEL)` event-count guard.** The procedure
  previously checked only ``n_periods`` (via per-asset
  ``MIN_TS_OBS = 20`` in ``compute_ts_betas``); a broadcast dummy with
  a single non-zero event would still produce a β driven entirely by
  that one observation, with no warning. Two-tier guard added on the
  event count: ``n_events < MIN_BROADCAST_EVENTS_HARD = 5`` raises
  ``InsufficientSampleError``;
  ``MIN_BROADCAST_EVENTS_HARD ≤ n_events < MIN_BROADCAST_EVENTS_RELIABLE = 20``
  emits the new ``WarningCode.SPARSE_COMMON_FEW_EVENTS``. Mirrors the
  existing ``n_periods`` two-tier (``MIN_PERIODS_HARD`` / ``_RELIABLE``).
  Constants live in ``factrix/_stats/constants.py``; the
  ``BROADCAST_`` prefix disambiguates from the CAAR
  ``MIN_EVENTS = 10`` in ``factrix/_types.py`` (different statistic).
  Empty-panel sparse-PANEL behaviour shifts from silent
  ``primary_p = 1.0`` to an explicit raise. (#29)
### Removed

- **`WarningCode.SPARSE_MAGNITUDE_DROPPED`** enum value + description.
  The warning existed to flag callers that the dispatched sparse
  procedure would drop magnitude via `.sign()`. With the
  `compute_caar` semantic shift above, no routing drops magnitude any
  more — the warning has nothing left to warn about. (#12)
- **`SuggestConfigResult.detected["magnitude_dropped"]`** key removed;
  `DETECTED_KEYS` reduced from 7 to 6. `_detect_signal` returns a
  3-tuple `(signal, reason, sparsity)` (was 4-tuple including
  `has_nonternary_magnitudes`). The `suggest_config` `magnitude_dropped`
  predicate, scope/mode gating, and `reasoning["signal"]` `.sign()`
  addendum are all gone — same root cause. **Migration:** delete any
  branch that reads `result.detected["magnitude_dropped"]` or
  membership-checks `WarningCode.SPARSE_MAGNITUDE_DROPPED`; on `{0, R}`
  inputs, `compute_caar` now does the right thing without warning. (#12)

## v0.7.0 (2026-05-04)

Closes the silent-coercion gap in sparse-procedure dispatch. Until now,
a user feeding a sparse-but-continuous signal (SUE z-score, ratings
notch delta, event-day return, order-flow imbalance burst, earnings
revision delta — anything where magnitude is the research target) was
silently routed to `Signal.SPARSE` purely on zero-ratio, then had their
magnitude information discarded inside `compute_caar` / `bmp_test` via
`pl.col(factor).sign()`. No warning, no info note, no way to know
without reading the source. This release makes the coercion *visible*
without changing it; the broader axis-design question — whether to add
a magnitude-weighted sparse procedure family — is tracked separately
(#12) and intentionally **not** bundled here.

### Added

- **`WarningCode.SPARSE_MAGNITUDE_DROPPED`** — emitted by
  `suggest_config(...)` when `_detect_signal` detects a SPARSE-shaped
  factor whose non-zero values are not strictly in {-1, +1}. Users see,
  before running anything, that CAAR / BMP will collapse magnitude to
  sign, and can rescale to ±1, route to a continuous procedure, or
  knowingly accept the sign-only semantics (#8).

### Changed (docs)

- **`compute_caar` docstring** now states the `.sign()` coercion in a
  dedicated `Note:` block and updates the `factor_col` argument
  description. Behavior is unchanged — the sign-only semantics has
  always been the contract; the docstring just no longer hides it.

### Migration

No code changes required. If `suggest_config(...).warnings` now
contains `WarningCode.SPARSE_MAGNITUDE_DROPPED`, your factor is being
treated sign-only by CAAR / BMP — this was already the behavior in
prior releases, you just couldn't see it. To preserve sign-only
semantics: ignore the warning. To use magnitude: pre-multiply your
factor to ±1 by another rule, or wait for the magnitude-weighted
sparse procedure tracked in #12.

## v0.6.0 (2026-05-03)

Time-series shape diagnostics + a statistical infrastructure layer that
makes them, and future Wald-based metrics, p-value-comparable with the
existing `ts_beta` family. Plus a quiet but load-bearing FDR-control
fix for batch BHY: `forward_periods` is now part of the family key, so
mixing horizons in a single `bhy()` call no longer silently dilutes the
step-up threshold.

### Added

- **`ts_quantile_spread` + `ts_asymmetry`** standalone diagnostics for
  `(COMMON, CONTINUOUS, *)` cells (#5). Both supplement the linear,
  symmetric OLS β assumed by `ts_beta_t_nw` — the first catches
  U-shape / inverted-U / extreme-only response via top-bottom bucket
  Wald, the second catches long-side ≠ short-side via either
  conditional means (method A) or piecewise slopes (method B). Three
  applicability gates (`distinct ≥ n_groups×2`, `兩側存在`, `雙側內變異`)
  short-circuit with `metadata["reason"]` + redirect hint instead of
  silent NaN.
- **NW HAC multivariate OLS + Wald helpers** (`factrix/_stats/__init__.py`)
  — the joint-regression infrastructure under the new diagnostics, with
  HAC variance and joint Wald χ² so all three (`ts_beta_t_nw`,
  `ts_quantile_spread`, `ts_asymmetry`) emit p-values from the same
  framework and stay cross-metric comparable.
- **`docs/metric_applicability.md`** §`ts_quantile_spread / ts_asymmetry`
  applicability matrix and gate definitions; **README** §文件導引 link
  to the new section.
- **README** use-case → factory reverse-lookup table for users not yet
  fluent in the three-axis vocabulary, plus a worked Bonferroni-then-BHY
  recipe for horizon-shopping correction.

### Fixed

- **`multi_factor.bhy()` family partitioning** now splits on
  `forward_periods` in addition to `(scope, signal, metric)`. Each
  horizon has its own null distribution and effective sample size;
  pooling them across horizons silently broke FDR control. Mixing
  horizons in one `bhy()` call now produces correctly-partitioned
  families.

### Changed (docs)

- Clarified that `forward_periods` is **rows on the time axis**, not
  calendar time — factrix is frequency-agnostic and shifts by row count.
  Aligned wording across README smoke-test callout, `AnalysisConfig`
  class + attribute docstrings, and `compute_forward_return` so IDE
  hover and README give the same answer. (Frequent confusion: users
  defaulted to a daily reading even on weekly / intraday panels.)
- Documented the **metric tier convention** (registry procedure vs
  standalone diagnostic) and softened user-facing terminology around
  cells / modes.

## v0.5.0 (2026-05-01)

Three-axis orthogonal API rewrite. Replaces the four `factor_type` strings
+ four parallel `Profile` dataclasses + `preprocess` / `factor` session /
`ProfileSet` triad with a single `AnalysisConfig` (4 factory methods over
`FactorScope × Signal × Metric`), a single `FactorProfile` result type,
and a registry-SSOT dispatch (`factrix/_registry.py`). PANEL (panel,
N≥2) and TIMESERIES (N=1) are now first-class equals — `(COMMON,
*, N=1)` and `(INDIVIDUAL, SPARSE, N=1)` produce real `primary_p`,
no longer pinned to `1.0`. Single-phase rip-and-replace per
`docs/plans/refactor_api.md` §8 — no alias
or deprecation cycle.

### BREAKING CHANGE

- **Public surface**: removed `fl.preprocess`, `fl.evaluate_batch`,
  `fl.factor()`, `fl.adapt`, `fl.validate_factor_data`,
  `fl.describe_profile`, `fl.describe_profile_values`, `fl.ProfileSet`,
  `fl.register_rule` / `fl.clear_custom_rules`. The new minimal surface
  is `fl.AnalysisConfig` + `fl.evaluate(panel, config)` +
  `fl.multi_factor.bhy(profiles, *, threshold=0.05)`.
- **Config**: `CrossSectionalConfig` / `EventConfig` / `MacroPanelConfig`
  / `MacroCommonConfig` removed. Construct via `AnalysisConfig.individual_continuous(metric=Metric.IC|Metric.FM)`,
  `.individual_sparse()`, `.common_continuous()`, `.common_sparse()`.
  `metric=Metric.FM` replaces `factor_type="macro_panel"` (the old name
  conflated data shape with research question).
- **New cell**: `(COMMON, SPARSE, None)` (`AnalysisConfig.common_sparse()`)
  was a coverage hole in v0.4 — now first-class for FOMC / policy / index
  rebalance broadcast events.
- **Profile**: `CrossSectionalProfile` / `EventProfile` /
  `MacroPanelProfile` / `MacroCommonProfile` collapsed into a single
  `FactorProfile` dataclass. Cell-specific scalars now live in
  `profile.stats: Mapping[StatCode, float]` keyed by enum (not string).
- **Field rename**: `Profile.canonical_p` → `FactorProfile.primary_p`.
  `Diagnostic` / `DiagnosticSeverity` removed; structured warnings now
  travel as `frozenset[WarningCode]` on `profile.warnings` (verdict-neutral).
- **Verdict**: `PASS_WITH_WARNINGS` removed. `Verdict` is binary `PASS`
  / `FAIL`. `warnings` / `info_notes` are surfacing-only — they never
  auto-rebind `primary_p` or upgrade `verdict()`.
- **TIMESERIES first class**: `(COMMON, *, N=1)` and `(INDIVIDUAL, SPARSE, N=1)`
  no longer return `primary_p = 1.0`. Real NW HAC t-tests on the
  underlying time series; `(INDIVIDUAL, SPARSE)` with the same N=1 user
  config and `(COMMON, SPARSE)` with N=1 collapse to the same procedure
  via the internal `_SCOPE_COLLAPSED` sentinel and tag the profile with
  `InfoCode.SCOPE_AXIS_COLLAPSED`.
- **PANEL invalid combos**: `(INDIVIDUAL, CONTINUOUS, *) × N=1` is
  mathematically undefined and now raises `ModeAxisError` with
  `suggested_fix=AnalysisConfig.common_continuous(...)` instead of
  silently degrading. `(INDIVIDUAL, *)` no longer accepts N=1 panels for
  CONTINUOUS metrics.
- **BHY**: `ProfileSet.multiple_testing_correct(p_source=, fdr=)` →
  `fl.multi_factor.bhy(profiles, *, threshold=0.05, gate=None)`.
  Family partitioning is automatic from the config triple — user no
  longer passes a group key; cross-family p mixing is structurally
  prevented.
- **Sample guards**: per-metric `MIN_FM_PERIODS = 20` / `MIN_TS_OBS = 20`
  unified into `MIN_PERIODS_HARD = 20` (raise `InsufficientSampleError`) and
  `MIN_PERIODS_RELIABLE = 30` (warn `UNRELIABLE_SE_SHORT_SERIES`) in
  `factrix/_stats/constants.py`. Procedures never silently produce a
  result on `n_periods < MIN_PERIODS_HARD`.
- **Errors**: `FactrixError` hierarchy — `ConfigError` →
  `{IncompatibleAxisError, ModeAxisError, InsufficientSampleError}`.
- **Removed v0.4 modules**: `_api.py`, `factor.py`, `config.py`,
  `validation.py`, `reporting.py`, `evaluation/pipeline.py`,
  `evaluation/profiles/`, `evaluation/profile_set.py`,
  `evaluation/diagnostics/`, `preprocess/pipeline.py`, `factors/`,
  `integrations/`, `charts/`, `metrics/redundancy.py`. `factrix/metrics/*`
  primitives kept — they back the v0.5 procedures.

### Added

- **API**: `factrix.AnalysisConfig` — three-axis frozen dataclass with
  4 type-safe factory methods. `__post_init__` runs every construction
  path (factory, direct, `from_dict`) through one validation gate.
- **API**: `factrix.evaluate(panel, config) -> FactorProfile` — single
  dispatch entry point. Panel schema: `(date, asset_id, factor, forward_return)`;
  Mode is derived from `panel["asset_id"].n_unique()`.
- **API**: `factrix.multi_factor.bhy` — Benjamini-Yekutieli step-up FDR
  correction with automatic family partitioning. Same-test-family
  enforced by config triple, not user discipline.
- **Introspection**: `factrix.describe_analysis_modes(format="text"|"json")`
  reverse-queries the registry to print all legal cells + procedures +
  references. `factrix.suggest_config(panel)` heuristic-picks a factory
  call from a raw panel.
- **Codes**: `WarningCode`, `InfoCode`, `StatCode`, `Verdict` StrEnums
  (`factrix/_codes.py`) — structured replacements for stringly-typed
  diagnostic / metadata payloads.
- **Registry SSOT**: `_DispatchKey(scope, signal, metric, mode)` →
  `_RegistryEntry(procedure, use_case, refs)` mapping. Adding a cell
  touches one `register(...)` call. Bootstrap import at the bottom of
  `_registry.py` populates the registry before any first query.
- **Procedures**: 7 `FactorProcedure` classes in `factrix/_procedures.py`
  covering 5 PANEL cells (IC, FM, CAAR, COMMON×CONT, COMMON×SPARSE) +
  2 TIMESERIES cells (TS-β CONTINUOUS, TS dummy SPARSE via
  `_SCOPE_COLLAPSED`).
- **Stats**: Hansen-Hodrick (1980) overlap floor
  `max(auto_bartlett(T), forward_periods - 1)` applied across all panel
  and timeseries cells with overlapping forward returns. Newey-West (1994)
  `auto_bartlett(T) = max(1, int(4 · (T/100)^(2/9)))` lag rule.

### Hardened (post-cut review fixes)

Applied during the v0.5 cut window before the surface was made public:

- `FactorProfile.n_assets: int` — panel cross-section width surfaced
  alongside `n_obs`. Disambiguates "small effective sample" between
  short series and thin cross-section. Visible in `diagnose()`.
- `multi_factor.bhy(gate=...)` requires a p-value `StatCode` and raises
  `ValueError` otherwise. Closes a footgun where `gate=StatCode.IC_T_NW`
  silently fed t-stats into BHY step-up. New `StatCode.is_p_value`
  property supports the validation.
- `multi_factor.bhy` emits `RuntimeWarning` when a batch yields ≥2
  size-1 families (= no FDR correction power) — surfaces the
  cross-family no-op anti-pattern.
- `WarningCode` / `InfoCode` gain `.description` glosses,
  `IncompatibleAxisError` leads with the actionable factory list,
  registry adds a `_SCOPE_COLLAPSED` metric guard + post-import
  invariant assert (catches silent registration drift).
- `_route_scope(scope, signal, mode)` SSOT for the §5.4.1 sparse-
  TIMESERIES scope-collapse rule; `_evaluate`, `_describe`, and
  `_multi_factor.bhy` all reverse-call it (no parallel implementations).

### Renamed (terminology disambiguation)

Pre-1.0 readability sweep — no behaviour change:

- `MIN_T_HARD` / `MIN_T_RELIABLE` → `MIN_PERIODS_HARD` /
  `MIN_PERIODS_RELIABLE`. `InsufficientSampleError` kwargs `actual_T` /
  `required_T` → `actual_periods` / `required_periods`. Disambiguates
  `T` (time-series length) from `t` (Student's t-statistic) used in
  `*_T_NW` `StatCode` enums. `auto_bartlett(T)` and `*_T_NW` keep the
  literal `T` (direct citations of NW1994 and Student's t).
- `describe_analysis_modes(format="json")` row keys
  `mode_a_panel` / `mode_b_timeseries` → `panel` / `timeseries`,
  matching the `Mode.PANEL` / `Mode.TIMESERIES` enum values that
  already drove dispatch.
- README / ARCHITECTURE / docstrings drop the `Mode A` / `Mode B`
  marketing label in favour of the enum names; procedure code uses
  `n_periods` / `n_assets` consistently for dimension counts.

### Migration

| v0.4                                     | v0.5                                                                       |
|------------------------------------------|----------------------------------------------------------------------------|
| `fl.preprocess(raw, config=cfg)`         | _(no preprocess step)_ caller attaches `forward_return` via `factrix.preprocess.returns.compute_forward_return` then evaluates |
| `fl.evaluate(prepared, name, config=cfg)` | `fl.evaluate(panel, cfg)` (no `name` — name belongs in caller's bookkeeping) |
| `factor_type="cross_sectional"`          | `AnalysisConfig.individual_continuous()` (default `metric=IC`)              |
| `factor_type="macro_panel"`              | `AnalysisConfig.individual_continuous(metric=Metric.FM)`                    |
| `factor_type="event_signal"`             | `AnalysisConfig.individual_sparse()`                                        |
| `factor_type="macro_common"`             | `AnalysisConfig.common_continuous()`                                       |
| _(coverage hole)_                        | `AnalysisConfig.common_sparse()`                                            |
| `Profile.canonical_p`                    | `profile.primary_p`                                                        |
| `profile.diagnose() -> list[Diagnostic]` | `profile.diagnose() -> dict[str, Any]` + `profile.warnings: frozenset[WarningCode]` |
| `ProfileSet.multiple_testing_correct(...)`  | `fl.multi_factor.bhy(profiles, threshold=0.05)`                          |
| `Profile.verdict()` ∈ `{PASS, PASS_WITH_WARNINGS, FAILED}` | `profile.verdict()` ∈ `{Verdict.PASS, Verdict.FAIL}` |
| `(COMMON, *) × N=1` → `primary_p = 1.0`  | TIMESERIES first-class — real NW HAC t-test                                    |
| _(no n_assets exposure)_                 | `FactorProfile.n_assets` (cross-section width)                              |
| `MIN_T_HARD` / `MIN_T_RELIABLE`          | `MIN_PERIODS_HARD` / `MIN_PERIODS_RELIABLE`                                |
| `InsufficientSampleError(actual_T=, required_T=)` | `InsufficientSampleError(actual_periods=, required_periods=)`         |

### Note

v0.4 → v0.5 was a single-phase rip-and-replace breaking change with no
alias or deprecation cycle. Pin to commit SHA across the boundary.

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
