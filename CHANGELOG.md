# Changelog

All notable changes to **factrix** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR**: API-breaking changes (field rename, signature change, removed metric)
- **MINOR**: New metrics, new Profile fields, new optional parameters
- **PATCH**: Bug fixes, docstring/test fixes, internal refactors

While the version is below `1.0.0`, the public API should be considered unstable — breaking changes may occur in **MINOR** bumps. Consumers are expected to pin by commit SHA (e.g., via git submodule) rather than relying on semver range constraints until `1.0.0` is cut.

---

## [Unreleased]

### Added

- **`factrix.metrics.slice_pairwise_test` / `slice_joint_test`** (#176). Cross-slice statistical-test verb pair. `slice_pairwise_test` reports K(K−1)/2 pairwise Wald contrasts with Holm / Romano-Wolf / Bonferroni adjusted p; `slice_joint_test` reports the single omnibus Wald χ² that all slice means are equal. Default estimator `WaldNWCluster` (joint NW HAC over the per-date K-vector panel) covers analytic inference; `BlockBootstrap` triggers the joint bootstrap path with Romano-Wolf as the default multiple-testing adjustment. Both verbs require the metric's module to declare a `per_date_series` capability — `ic` / `fama_macbeth` / `hit_rate` ship with it; metrics without it raise `TypeError`. See `docs/api/slice-test.md`.
- **`factrix.metrics._metric_capabilities`** (#176). Resolver helpers (`resolve_per_date_series`, `resolve_min_assets_per_group`) plus a `PerDateSeries` Protocol and a `per_date_series_rename` factory. Centralises capability lookup so inference verbs reuse one resolver instead of grovelling `sys.modules` directly.

### Deprecated

- **`factrix.metrics.regime_ic`** (#176). Use `slice_pairwise_test(ic, ic_df.join(regime_labels, on="date"), label="regime")` for the same analysis. The new verb's pairwise contrast frame (Wald χ² + Holm / Romano-Wolf adjusted p) replaces `regime_ic`'s BHY-on-min-|t| summary shape; downstream code consuming `per_regime` metadata should migrate before the function is removed in a future minor. Output shape frozen for one more minor as the migration window.

### Removed

- **`factrix.metrics.multi_horizon_ic` / `multi_horizon_hit_rate`** (breaking, #186). Deprecated in v0.11.0; the in-metric horizon loop conflicted with `FactorProfile.identity` carrying `forward_periods` (the #160 anti-shopping defense) and ran a second BHY path inside the metric in parallel to `multi_factor.bhy(profiles, expand_over=["forward_periods"])`, the FDR SSOT. Both names are no longer importable from `factrix.metrics`; direct references raise `ImportError`. Code reaching them via `list_metrics` / `run_metrics` was never wired (already excluded via `_AUTO_DISCOVER_EXCLUDED` in v0.11). The `_HorizonICEntry` TypedDict and the `_metric_index._DEPRECATED` set are removed alongside the functions. Migration recipes (descriptive `run_metrics` per horizon + `pl.concat` of `bundle.to_frame()`; inferential `evaluate` per horizon + `bhy(expand_over=["forward_periods"])`) remain in `docs/api/multi-horizon.md` and apply unchanged from the v0.11.0 deprecation window.

  ```python
  # before (v0.11.0, deprecated)
  fx.metrics.multi_horizon_ic(panel, periods=[1, 5, 10, 20])

  # after (descriptive sweep)
  bundles = [
      fx.run_metrics(panel, cfg.replace(forward_periods=h))
      for h in [1, 5, 10, 20]
  ]
  table = pl.concat([b.to_frame() for b in bundles])

  # after (FDR-controlled inference)
  profiles = [fx.evaluate(panel, cfg.replace(forward_periods=h)) for h in [1, 5, 10, 20]]
  survivors = fx.multi_factor.bhy(profiles, expand_over=["forward_periods"])
  ```

## v0.11.0 (2026-05-11)

### Added

- **`factrix.stats.Estimator`** — runtime-checkable Protocol for inference-method instances. Implementations supply `name` / `description` / `applicable_to(scope, signal)` / `emits_for(scope, signal, metric)`; the family-verb resolution layer dispatches via these to a `StatCode` key in `profile.stats`. The protocol is selection-only — no `compute()` method — so cell-internal estimator swap stays a future `ComputableEstimator(Estimator)` extension and the surface does not pre-commit to a return shape that NW (`SE+t+p`) and GMM (`J-stat+df+p`) cannot share. (#170)
- **`factrix.stats.NeweyWest`** — reference Estimator naming the procedure-emitted Newey-West HAC inference path. Carries no compute logic; the underlying NW HAC math stays in `factrix._stats` and is invoked by each cell procedure during `evaluate()`. Constructor takes no arguments in v0.11; `lag` / `kernel` / `overlap_floor` knobs are tracked for a future enhancement. (#170)
- **`factrix.list_estimators(scope, signal, *, format, with_import)`** — mirrors `list_metrics` shape so the pre-flight pattern ("which scalars and which inference methods does this cell admit") is one API. Returned rows track the `_ESTIMATOR_REGISTRY` (`NeweyWest`, `HansenHodrick`); GMM follows in #191. (#170)

- **`factrix.stats.HansenHodrick`** — Hansen-Hodrick (1980) rectangular-kernel HAC `Estimator` for the IC PANEL / FM PANEL cells where overlapping h-period forward returns induce MA(h-1) structure. Closed-form `Var(mean) = (γ₀ + 2 Σ_{j=1..h-1} γⱼ) / n`. Cell-agnostic dispatch via `StatCode.P_HH`; `applicable_to` restricted to `(INDIVIDUAL, CONTINUOUS)`. Procedure-side gate skips the emission when `forward_periods == 1` (no overlap → HH collapses to iid SE), so `bhy(estimator=HansenHodrick())` on a non-overlapping profile lands on a missing-stat error instead of aliasing NW. (#184)
- **`StatCode.T_HH`** — Hansen-Hodrick t-statistic, sibling of `T_NW`. Emitted as a pair with `P_HH` so HH inference reproducibility is symmetric with NW. (#184)
- **`WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE`** — fired by HH (and any future rectangular-kernel HAC variant) when the variance estimate `γ₀ + 2 Σγⱼ` comes out negative on short / mildly anti-correlated samples (no PSD guarantee, Andrews 1991 §3). The primitive clamps variance to 0 → SE=0 → t=0 → p=1.0 (the conservative "cannot reject" direction); the procedure surfaces the flag in `profile.warnings` and mirrors `{"variance_clamped": True}` under both `metadata[StatCode.T_HH]` and `metadata[StatCode.P_HH]`. Generic name (rather than `HH_*`) so future rectangular-kernel HAC variants can reuse it. (#184)
- **IC PANEL / FM PANEL procedures emit the `(T_HH, P_HH)` pair when `forward_periods > 1`** — populates HH-pure t-stat + p-value alongside the existing NW-derived `(T_NW, P_NW)`, with shared `metadata = {"kernel": "rectangular", "variance_clamped": <bool>}` mirrored under both StatCode keys. `EMITS_STATS` lists both as conditionally emitted; downstream readers should consult `profile.stats` membership rather than assume universal presence. (#184)
- **`StatCode` grammar locked in for inference primary stats** — module docstring now spells out the `<TEST_STAT_KIND>_<ALGO>` / `P_<ALGO>` shape (`T_NW` / `T_HH`, `P_NW` / `P_HH` / `P_GMM`, future `J_GMM` / `WALD` / `F` / `LR`) and a redesign trigger: when ≥ 4 inference algorithms ship concurrently or ≥ 3 distinct test-statistic KINDs coexist, the flat enum yields to a structured `profile.inference[Algo.X]` shape. Below those thresholds the flat enum stays cheaper. (#184)

- **`FactorProfile.metadata: Mapping[StatCode, Mapping[str, Any]]`** — new field carrying hyperparameter records for each populated stat (#188). Symmetric with `stats`: for any populated entry, `stats[code]` is the value and `metadata[code]` is the inner dict of hyperparameters that produced it. Examples by cell:

  | Cell | Populated `metadata` keys | Inner |
  |---|---|---|
  | IC / FM / CAAR PANEL | `T_NW`, `P_NW` | `{"nw_lags": <resolved bandwidth>}` |
  | COMMON CONTINUOUS PANEL | `FACTOR_ADF_TAU`, `FACTOR_ADF_P` | `{"lag_order": 0}` |
  | COMMON CONTINUOUS TIMESERIES | `T_NW`, `P_NW`, `FACTOR_ADF_TAU`, `FACTOR_ADF_P` | NW lags + ADF lag_order |
  | TS-dummy SPARSE TIMESERIES | `T_NW`, `P_NW`, `RESID_LJUNG_BOX_Q`, `RESID_LJUNG_BOX_P`, `EVENT_HHI_VALUE` | NW lags + Ljung-Box `lag_h` + HHI `n_bins` |

  Stats with no hyperparameter (`MEAN`) are absent from the mapping rather than mapping to an empty dict. Tests that share a hyperparameter (NW populates `T_NW` + `P_NW` from one bandwidth; Ljung-Box populates `Q` + `P` from one `lag_h`) duplicate the inner dict under both keys to keep single-key lookup honest. `profile.diagnose()["metadata"]` serialises with `StatCode.value` strings as outer keys and plain dicts inside.

  This restores reproducibility for the NW lag count after #187 removed `StatCode.NW_LAGS_USED`, and surfaces previously-discarded hyperparameters (ADF `lag_order`, Ljung-Box `lag_h`, HHI `n_bins`) under the same uniform schema. The `_ljung_box` internal helper now returns `(h, Q, p)` instead of `(Q, p)` so callers receive the resolved lag count alongside the test output. (#188)

- **`factrix.run_metrics(panel, cfg, *, factor_col, metrics=None) -> MetricsBundle`** — descriptive twin of `evaluate` for running a cell's standalone metrics in one call. Same `(panel, cfg)` entry contract as `evaluate`; disjoint result type so inferential (`FactorProfile`) and descriptive (`MetricsBundle`) layers stay separate. Default auto-discovers panel-input metrics from `list_metrics(scope, signal)` and shares one stage-1 compute (e.g. `compute_ic`) across consumer metrics in the same module. `metrics=[name, ...]` runs an explicit subset; unknown / excluded names raise `UserInputError` with fuzzy suggestion (per #165). Stage-1 consumer metrics in non-IC cells (`caar`, `fama_macbeth`, `ts_beta`, `mfe_mae_summary`, plus series / spread consumers) carry an `_AUTO_DISCOVER_EXCLUDED` reason with explicit-call recipe; v1.x extends per-cell stage-1 wiring. (#147)
- **`factrix.MetricsBundle`** — frozen dataclass returned by `run_metrics`. Exposes dict-style access (`bundle["ic"]` → `MetricOutput`, `"name" in bundle`, `iter(bundle)`), `bundle.identity` (factor_id, forward_periods) for grouping, `bundle.context` (mapping of sample-restriction keys, populated by `slice_analysis` / regime consumers in v1.x), `bundle.skipped` (name → reason for `_AUTO_DISCOVER_EXCLUDED` hits and stage-1 failures), and `bundle.to_frame()` with a stable 8-column schema (`factor_id` / `forward_periods` / `metric` / `value` / `stat` / `significance` / `p_value` / `short_circuit_reason`). `__hash__ = None` because the bundle holds mutable `MetricOutput.metadata`; group bundles via `bundle.identity` (a hashable tuple). Errors split three ways — sample-floor / `InsufficientSampleError` becomes a short-circuit `MetricOutput` inside the bundle; user input mistakes raise `UserInputError`; metric-internal bugs wrap as `RunMetricsError(FactrixError)` with `cell` / `metric_name` / `stage` fields and `__cause__` chain. (#147)

- **`factrix.UserInputError`** — marker exception subclass of `FactrixError` and `ValueError` for user-input typos / type mismatches across the v1 verb surface. Multi-inherits `ValueError` so `except ValueError` ecosystem code (pytest fixtures, polars internals) still catches it. Constructor is keyword-only (`verb` / `field` / `value` / `candidates` / `docs_url`) and renders a canonical message: difflib fuzzy suggestion, sorted candidate list (capped at 15), value repr (capped at 120 chars), and a deployed-docs URL. Adopted by `bhy` (#161 duplicate-identity / non-p gate), `run_metrics` (#147 unknown / excluded metric names), and other v1 sub-issues; existing raise sites are not retrofitted. (#169)

### Changed

- **`StatCode.P` renamed to `StatCode.P_NW`** (breaking, #192). Primary inference codes are now uniformly `P_<algo>` (`P_NW` / `P_HH` / `P_GMM`), parallel to the existing `T_<algo>` convention (`T_NW` / `T_HH`). The bare `P` was the odd one out — algorithm provenance was implicit in the name and only carried by the description string, which fails grep / IDE auto-complete and rots silently. Migration: replace every `StatCode.P` lookup with `StatCode.P_NW`; `profile.diagnose()` JSON keys move from `"p"` to `"p_nw"`. Family verbs (`bhy(profiles)` without explicit `estimator=`) drive off `primary_p` and are unaffected; `bhy(profiles, estimator=NeweyWest())` continues to work because `NeweyWest.emits_for` was retargeted to `P_NW`.

  | Before | After |
  |---|---|
  | `StatCode.P` (value `"p"`) | `StatCode.P_NW` (value `"p_nw"`) |

  Out of scope (unchanged): `StatCode.MEAN` (no algorithm axis), `StatCode.T_NW` (already correctly named), diagnostic codes `FACTOR_ADF_P` / `RESID_LJUNG_BOX_P` (grammar `<TARGET>_<TEST>_P` is structural — TARGET distinguishes factor input / residual / event distribution; the asymmetry with primary `P_<algo>` is documented in the `StatCode` module docstring). (#192)

- **`factrix.multi_factor.bhy` accepts `estimator: Estimator | None` instead of `p_stat: StatCode | None`** (breaking, #170). The previous `p_stat=` kwarg was a placeholder landed in v0.10 alongside the family-verb refactor and is removed in v0.11. Migration:

  ```python
  # before (v0.10)
  fx.multi_factor.bhy(profiles, p_stat=fx.StatCode.IC_P)

  # after (v0.11)
  from factrix.stats import NeweyWest
  fx.multi_factor.bhy(profiles, estimator=NeweyWest())
  ```

  Default behaviour (`estimator=None`) is unchanged — each profile's `primary_p` drives the step-up. `StatCode.is_p_value` continues to gate `profile.verdict(gate=...)`; the family-verb path no longer consults it because an `Estimator` instance is implicitly a p-value source by construction (`emits_for` returns a probability `StatCode`). The `_STAT_DESCRIPTIONS[StatCode.*_T_NW]` entries are slimmed: kernel / bandwidth / overlap-floor implementation details now live on `NeweyWest` itself, while enum descriptions retain only cell-specific stat semantics and cross-ref the estimator class. (#170)

- **`factrix.multi_factor.bhy` returns `Survivors` instead of `list[FactorProfile]`** (breaking under v0.x). Migration: replace `survivors` with `survivors.profiles` for downstream list / iteration use. The new container exposes `.profiles` (input order, kept rows only), `.adj_q` (bucket-local BHY-adjusted p-values, aligned to `.profiles`), `.q`, `.expand_over` (tuple of partition keys), and `.n_total` (per-bucket `m`, keyed by `expand_over_values`). Internally `bhy` builds the survivor index as `{i : bhy_adjusted_p(p_array)[i] <= q}` per bucket and slices both `.profiles` and `.adj_q` to that set, so the survivor mask and the adjusted p-values downstream code reads come from the same `bhy_adjusted_p` call (the previous parallel `bhy_adjust` mask path is removed) — tie / boundary cases where two parallel implementations could disagree are eliminated by construction. `Survivors` ships `__repr__` / `_repr_html_` for Jupyter — three-column `identity | primary_p | adj_q` table, plus an `expand_over_values` column when buckets are declared. The container is procedure-agnostic; future Holm / Bonferroni / Romano-Wolf verbs will populate the same shape via their own `*_adjusted_p`. (#171)
- **`factrix.multi_factor.bhy` retired v0.4 auto-partition; caller now declares the family explicitly** (breaking, #161). The previous behaviour of auto-isolating buckets by dispatch cell × forward horizon is gone — `bhy(profiles)` treats the input list as **one** family and runs a single step-up. To run per-bucket independent step-ups (Benjamini & Bogomolov 2014 selective inference), declare the partition keys via `expand_over=[<context key>, ...]`. Mixed `forward_periods` without `expand_over` now emits a `RuntimeWarning` flagging the FDR-inflation foot-gun (silent pooling dilutes the per-rank threshold). The default `factor_id="factor"` across multiple cells now raises `UserInputError` (duplicate identity) instead of silently auto-splitting; the error hint suggests setting distinct `factor_id` or using `expand_over`. The `_resolve_family` layer p-stat validation (`StatCode.is_p_value`) is shared across `bhy` / `bonferroni` / `holm` / `partial_conjunction`, so every family verb enforces the same gatekeeping. Migration:

  ```python
  # before (v0.10) — auto-partition by cell × forward_periods
  fx.multi_factor.bhy(profiles)

  # after (v0.11) — declare buckets explicitly
  fx.multi_factor.bhy(profiles, expand_over=["forward_periods"])
  ```

  The `threshold=` and `gate=` aliases still accept input but emit `DeprecationWarning`; both will be removed next release. (#161)

- **`FactorProfile` gains `identity: tuple[str, int]` and `context: Mapping[str, Any]`** (#160 / #172). `identity = (factor_id, forward_periods)` is the v1 anti-shopping defense for multi-horizon factor research — MTC family forms naturally over `identity` (used by `bhy(expand_over=["forward_periods"])`), while sample-restriction / conditioning dimensions stay queryable via `profile.context[key]` (universe / regime entries populated by higher-level verbs through `dataclasses.replace`). `factor_id` is a real dataclass field (default `"factor"`); `forward_periods` is derived from `profile.config`; `identity` is a read-only property returning the tuple. `__hash__ = None` makes the unhashable contract explicit (group by `profile.identity` instead). `_evaluate` is the single stamp site — cell procedures stay schema-agnostic. New `__repr__` / `_repr_html_` render `identity` / mode / `primary_p` / sample sizes and unfold non-empty `context.<key>` rows in Jupyter; `_repr_html_` escapes user-supplied factor_id / context values via `html.escape()` to prevent injected HTML in notebook embeds. `profile.diagnose()` schema gains `identity` and `context` fields. (#160)

- **Terminology**: rename "Layer A" / "Layer B" to **dispatcher** / **curated wrapper** in module docstrings and user-facing docs. Public API names are unchanged. Older CHANGELOG entries below retain the original wording. (#157)
- **Docs convention**: switch the recommended import alias from `fl` to `fx` across README, mkdocs pages, notebooks, tests, and `llms-full.txt`. `fl` collided with the FinLab community convention (`import finlab as fl`) and carried no mnemonic tie to `factrix`; `fx` takes the first and last letters in the jax-as-`jnp` / polars-as-`pl` / networkx-as-`nx` style. Public API and importable package name (`factrix`) are unchanged — docs-only convention shift, not a breaking change. (#180)

- **`StatCode` naming flattened** (breaking, #187). Primary cell stats lose their metric-name prefix because cell identity already lives on `profile.config` (`scope` / `signal` / `metric`); diagnostics gain explicit prefixes (`FACTOR_` / `RESID_` / `EVENT_`) because their target sits outside `config`. Naming grammar is now `<TARGET>_<KIND>` where TARGET is empty for primary and explicit for diagnostics, KIND is one of `_MEAN` / `_VALUE` / `_<statistic>` / `_P` / `_P_<algorithm>`.

  Rename map (procedure emit + downstream readers):

  | Before | After |
  |---|---|
  | `IC_MEAN` / `FM_LAMBDA_MEAN` / `CAAR_MEAN` / `TS_BETA` | `MEAN` |
  | `IC_T_NW` / `FM_LAMBDA_T_NW` / `TS_BETA_T_NW` / `CAAR_T_NW` | `T_NW` |
  | `IC_P` / `FM_LAMBDA_P` / `TS_BETA_P` / `CAAR_P` | `P` |
  | `LJUNG_BOX_P` | `RESID_LJUNG_BOX_P` |
  | `EVENT_TEMPORAL_HHI` | `EVENT_HHI_VALUE` |

  New StatCodes shipped as part of the refactor:

  - `P_HH` / `T_HH` — landed in #184 (HH-pure rectangular-kernel HAC variant for IC / FM PANEL); the (T_HH, P_HH) pair mirrors the (T_NW, P) shape so HH and NW carry symmetric information. `P_GMM` reserved for #191 (Hansen 1982 GMM J-test); the matching `J_GMM` chi-square statistic lands together with the GMM procedure in that issue.
  - `FACTOR_ADF_TAU` / `RESID_LJUNG_BOX_Q` — the underlying ADF τ statistic and Ljung-Box Q statistic are now emitted alongside their existing p-values; the math was already computed inside the procedure but the value was previously discarded.

  `StatCode.is_p_value` widened from `value.endswith("_p")` to a tokenised check (`"p" in value.split("_")`) so bare `P` and algorithm variants `P_HH` / `P_GMM` qualify alongside `*_P` diagnostics.

  `Estimator.emits_for` simplified — `NeweyWest` no longer dispatches per-cell to a metric-specific `*_P`; it returns `StatCode.P` cell-agnostically. Future Estimator instances (HansenHodrick / GMM / DriscollKraay) return their own `P_*` value in one line, removing the N-cell × M-algorithm dispatch table the previous shape would have required.

  Downstream consumers of `profile.diagnose()` JSON: the `stats` sub-dict's keys move from `"ic_p"` / `"ic_mean"` / `"caar_p"` / etc. to flat `"p"` / `"mean"` / `"t_nw"`. Filtering / dashboard code that reads keys by their old metric-prefixed string needs the same rename map applied to its own logic. (#187)

### Deprecated

- **`factrix.metrics.multi_horizon_ic` / `multi_horizon_hit_rate`** — sweeping IC / hit-rate across `[1, 5, 10, 20]` forward periods is a dispatcher concern, not a per-cell metric. The in-metric horizon loop conflicted with `FactorProfile.identity` carrying `forward_periods` (#160 anti-shopping defense) and ran a second BHY path inside the metric (`metadata["p_adjusted_bhy"]`) parallel to `multi_factor.bhy(profiles, expand_over=["forward_periods"])`, the FDR SSOT. Both functions remain importable and runnable for one release cycle but emit `DeprecationWarning` on call and are excluded from `list_metrics` output (`_metric_index._DEPRECATED`). `run_metrics` auto-discover already skipped them via `_AUTO_DISCOVER_EXCLUDED` (per #147). Migration: `run_metrics(panel, cfg.replace(forward_periods=h))` per horizon → `compare(bundles)` for descriptive horizon-by-metric view, or `evaluate(...)` per horizon → `multi_factor.bhy(profiles, expand_over=["forward_periods"])` for FDR-controlled inference. Both paths are metric-agnostic — `mfe_mae` / `caar` / `oos` / `monotonicity` inherit horizon-sweep support automatically. Recipes in `docs/api/multi-horizon.md`. Removal version pinned at the next major-bump release-train. (#186)

### Removed

- **`factrix.multi_factor.bhy(p_stat=StatCode)` path** — replaced by `estimator=Estimator` (see Changed above). (#170)
- **`factrix.multi_factor.bhy(gate=...)` deprecation alias** — the v0.4 alias for `p_stat=` is removed alongside its successor; users still on `gate=` should jump directly to `estimator=NeweyWest()`. (#170)
- **`StatCode.NW_LAGS_USED`** — Newey-West auto-bandwidth lag count is no longer surfaced on `profile.stats`. The lag selection logic in `_resolve_nw_lags` is unchanged; the value just stops being externalised. Reinstating it under a dedicated `profile.metadata` (or sibling) channel is tracked as #188 — `_codes.py` was the wrong home (a hyperparameter-selection record, not a stat). (#187)

---

## v0.10.0 (2026-05-09)

### BREAKING CHANGE

- by_regime emits DeprecationWarning since v0.10.0;
removal scheduled for a future minor (separate sub-issue). Migrate to
by_slice with an explicit inner-join — see docs/api/by-regime.md.

### Feat

- **metrics**: deprecate by_regime in favor of by_slice (#154)
- **metrics**: add by_slice axis-agnostic dispatcher (#154)

### Fix

- **mfe_mae**: instantiate Polars dtypes in schema dict

### Refactor

- **ic**: TypedDict for per-regime / per-horizon entries

## v0.9.0 (2026-05-07)

### Feat

- **metrics**: by_regime regime dispatcher (#107) (#112)
- **metrics**: add by_regime regime dispatcher (#107)
- **describe**: SuggestConfigResult.diagnose() symmetry
- **codes**: StatCode.description for symmetry
- **evaluate**: factor_col= signal-column alias
- **introspection**: expose per-cell stats_keys via EMITS_STATS
- **preprocess**: expose compute_forward_return as public API (#91)

### Fix

- persona cross-cuts review followups (#110)
- **docs-hooks**: prune stale notebooks from docs/examples on build (#99) (#103)
- **ci**: fix CHANGELOG regex and add --latest to release job

### Refactor

- **ic**: share regime slicing primitive (#107)

## v0.8.0 (2026-05-07)

### Feat

- **brand**: add logo banner and icon (#86)
- **api**: list_metrics for per-cell standalone metric discovery (#76) (#79)
- **api**: list_metrics for per-cell standalone metric discovery (#76)
- **api**: friendly error when evaluate() called without config (#72)
- **metrics**: four-angle primitive polish (#48)
- **caar**: preserve magnitude in compute_caar (#12)
- **introspect**: SuggestConfigResult.detected (#20)
- **introspect**: two-tier n_assets guard

### Fix

- **readme**: point docs badge to correct workflow file
- **describe**: cross_section_tier on inference-stage N (#83)
- **ci**: use uv pip for import-from-wheel smoke
- **stats**: calendar-time CAAR for PANEL NW HAC (#24) (#37)
- **stats**: two-tier n_events guard on common_sparse (#25) (#29)
- **introspect**: scope-gate magnitude_dropped (#23) (#28)

### Refactor

- **scripts**: organise scripts/ by purpose (#65)
- **registry**: add _dispatch_key_for SSOT helper
- naming + discoverability hygiene sweep

## v0.7.0 (2026-05-07)

### Feat

- **introspect**: warn on dropped sparse magnitude

## v0.6.0 (2026-05-03)

### Feat

- **metrics**: add ts_quantile_spread + ts_asymmetry (#5)
- **stats**: add NW HAC multivariate OLS + Wald helpers

### Fix

- **multi_factor**: split bhy family on forward_periods

## v0.5.0 (2026-05-02)

### Feat

- **api**: expose n_assets on FactorProfile; gate-validate bhy
- **api**: complete v0.5 surface — three-axis dispatch refactor
- **api**: switch public surface to v0.5; rip out v0.4
- **api**: add v0.5 multi_factor.bhy with family partitioning
- **api**: add v0.5 describe + suggest_config helpers
- **api**: wire CAAR PANEL — 7/7 cells live
- **api**: wire COMMON PANEL procedures (cont + sparse)
- **api**: wire TS dummy SPARSE Mode B procedure
- **api**: wire TS β CONTINUOUS Mode B procedure
- **api**: wire FM PANEL procedure compute
- **api**: add v0.5 _evaluate dispatch wrapper
- **api**: wire IC PANEL procedure compute
- **api**: wire v0.5 dispatch registry SSOT
- **api**: scaffold v0.5 AnalysisConfig foundation

### Fix

- **build**: correct dependencies table location in pyproject.toml
- **api**: apply v0.5 review UX minors
- **api**: apply v0.5 review architecture minors

### Refactor

- rename T → n_periods to disambiguate from t-stat
- drop Mode A/B for PANEL/TIMESERIES

## v0.4.0 (2026-04-25)

### BREAKING CHANGE

- breakeven_cost and net_spread now require
forward_periods. Direct callers must update.

### Feat

- **diagnostics**: add high_turnover_jaccard rule
- **tradability**: separate cost driver from rank stability

### Fix

- **tradability**: align mp turnover stride with fp
- **tradability**: complete jaccard cost wiring
- **tradability**: scale cost by holding period

### Refactor

- **tradability**: rename jaccard to notional

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

  # before (v0.10)
  fx.multi_factor.bhy(profiles, p_stat=fx.StatCode.IC_P)

  # after (v0.11)
  from factrix.stats import NeweyWest
  fx.multi_factor.bhy(profiles, estimator=NeweyWest())
  ```

  Default behaviour (`estimator=None`) is unchanged — each profile's `primary_p` drives the step-up. `StatCode.is_p_value` continues to gate `profile.verdict(gate=...)`; the family-verb path no longer consults it because an `Estimator` instance is implicitly a p-value source by construction (`emits_for` returns a probability `StatCode`). The `_STAT_DESCRIPTIONS[StatCode.*_T_NW]` entries are slimmed: kernel / bandwidth / overlap-floor implementation details now live on `NeweyWest` itself, while enum descriptions retain only cell-specific stat semantics and cross-ref the estimator class. (#170)

- **`factrix.multi_factor.bhy` returns `Survivors` instead of `list[FactorProfile]`** (breaking under v0.x). Migration: replace `survivors` with `survivors.profiles` for downstream list / iteration use. The new container exposes `.profiles` (input order, kept rows only), `.adj_q` (bucket-local BHY-adjusted p-values, aligned to `.profiles`), `.q`, `.expand_over` (tuple of partition keys), and `.n_total` (per-bucket `m`, keyed by `expand_over_values`). Internally `bhy` builds the survivor index as `{i : bhy_adjusted_p(p_array)[i] <= q}` per bucket and slices both `.profiles` and `.adj_q` to that set, so the survivor mask and the adjusted p-values downstream code reads come from the same `bhy_adjusted_p` call (the previous parallel `bhy_adjust` mask path is removed) — tie / boundary cases where two parallel implementations could disagree are eliminated by construction. `Survivors` ships `__repr__` / `_repr_html_` for Jupyter — three-column `identity | primary_p | adj_q` table, plus an `expand_over_values` column when buckets are declared. The container is procedure-agnostic; future Holm / Bonferroni / Romano-Wolf verbs will populate the same shape via their own `*_adjusted_p`. (#171)
- **`factrix.multi_factor.bhy` retired v0.4 auto-partition; caller now declares the family explicitly** (breaking, #161). The previous behaviour of auto-isolating buckets by dispatch cell × forward horizon is gone — `bhy(profiles)` treats the input list as **one** family and runs a single step-up. To run per-bucket independent step-ups (Benjamini & Bogomolov 2014 selective inference), declare the partition keys via `expand_over=[<context key>, ...]`. Mixed `forward_periods` without `expand_over` now emits a `RuntimeWarning` flagging the FDR-inflation foot-gun (silent pooling dilutes the per-rank threshold). The default `factor_id="factor"` across multiple cells now raises `UserInputError` (duplicate identity) instead of silently auto-splitting; the error hint suggests setting distinct `factor_id` or using `expand_over`. The `_resolve_family` layer p-stat validation (`StatCode.is_p_value`) is shared across `bhy` / `bonferroni` / `holm` / `partial_conjunction`, so every family verb enforces the same gatekeeping. Migration:

  ```python
  # before (v0.10) — auto-partition by cell × forward_periods
  fx.multi_factor.bhy(profiles)

  # after (v0.11) — declare buckets explicitly
  fx.multi_factor.bhy(profiles, expand_over=["forward_periods"])
  ```

  The `threshold=` and `gate=` aliases still accept input but emit `DeprecationWarning`; both will be removed next release. (#161)

- **`FactorProfile` gains `identity: tuple[str, int]` and `context: Mapping[str, Any]`** (#160 / #172). `identity = (factor_id, forward_periods)` is the v1 anti-shopping defense for multi-horizon factor research — MTC family forms naturally over `identity` (used by `bhy(expand_over=["forward_periods"])`), while sample-restriction / conditioning dimensions stay queryable via `profile.context[key]` (universe / regime entries populated by higher-level verbs through `dataclasses.replace`). `factor_id` is a real dataclass field (default `"factor"`); `forward_periods` is derived from `profile.config`; `identity` is a read-only property returning the tuple. `__hash__ = None` makes the unhashable contract explicit (group by `profile.identity` instead). `_evaluate` is the single stamp site — cell procedures stay schema-agnostic. New `__repr__` / `_repr_html_` render `identity` / mode / `primary_p` / sample sizes and unfold non-empty `context.<key>` rows in Jupyter; `_repr_html_` escapes user-supplied factor_id / context values via `html.escape()` to prevent injected HTML in notebook embeds. `profile.diagnose()` schema gains `identity` and `context` fields. (#160)

- **Terminology**: rename "Layer A" / "Layer B" to **dispatcher** / **curated wrapper** in module docstrings and user-facing docs. Public API names are unchanged. Older CHANGELOG entries below retain the original wording. (#157)
- **Docs convention**: switch the recommended import alias from `fl` to `fx` across README, mkdocs pages, notebooks, tests, and `llms-full.txt`. `fl` collided with the FinLab community convention (`import finlab as fl`) and carried no mnemonic tie to `factrix`; `fx` takes the first and last letters in the jax-as-`jnp` / polars-as-`pl` / networkx-as-`nx` style. Public API and importable package name (`factrix`) are unchanged — docs-only convention shift, not a breaking change. (#180)

- **`StatCode` naming flattened** (breaking, #187). Primary cell stats lose their metric-name prefix because cell identity already lives on `profile.config` (`scope` / `signal` / `metric`); diagnostics gain explicit prefixes (`FACTOR_` / `RESID_` / `EVENT_`) because their target sits outside `config`. Naming grammar is now `<TARGET>_<KIND>` where TARGET is empty for primary and explicit for diagnostics, KIND is one of `_MEAN` / `_VALUE` / `_<statistic>` / `_P` / `_P_<algorithm>`.

  Rename map (procedure emit + downstream readers):

  | Before | After |
  |---|---|
  | `IC_MEAN` / `FM_LAMBDA_MEAN` / `CAAR_MEAN` / `TS_BETA` | `MEAN` |
  | `IC_T_NW` / `FM_LAMBDA_T_NW` / `TS_BETA_T_NW` / `CAAR_T_NW` | `T_NW` |
  | `IC_P` / `FM_LAMBDA_P` / `TS_BETA_P` / `CAAR_P` | `P` |
  | `LJUNG_BOX_P` | `RESID_LJUNG_BOX_P` |
  | `EVENT_TEMPORAL_HHI` | `EVENT_HHI_VALUE` |

  New StatCodes shipped as part of the refactor:

  - `P_HH` / `T_HH` — landed in #184 (HH-pure rectangular-kernel HAC variant for IC / FM PANEL); the (T_HH, P_HH) pair mirrors the (T_NW, P) shape so HH and NW carry symmetric information. `P_GMM` reserved for #191 (Hansen 1982 GMM J-test); the matching `J_GMM` chi-square statistic lands together with the GMM procedure in that issue.
  - `FACTOR_ADF_TAU` / `RESID_LJUNG_BOX_Q` — the underlying ADF τ statistic and Ljung-Box Q statistic are now emitted alongside their existing p-values; the math was already computed inside the procedure but the value was previously discarded.

  `StatCode.is_p_value` widened from `value.endswith("_p")` to a tokenised check (`"p" in value.split("_")`) so bare `P` and algorithm variants `P_HH` / `P_GMM` qualify alongside `*_P` diagnostics.

  `Estimator.emits_for` simplified — `NeweyWest` no longer dispatches per-cell to a metric-specific `*_P`; it returns `StatCode.P` cell-agnostically. Future Estimator instances (HansenHodrick / GMM / DriscollKraay) return their own `P_*` value in one line, removing the N-cell × M-algorithm dispatch table the previous shape would have required.

  Downstream consumers of `profile.diagnose()` JSON: the `stats` sub-dict's keys move from `"ic_p"` / `"ic_mean"` / `"caar_p"` / etc. to flat `"p"` / `"mean"` / `"t_nw"`. Filtering / dashboard code that reads keys by their old metric-prefixed string needs the same rename map applied to its own logic. (#187)

### Deprecated

- **`factrix.metrics.multi_horizon_ic` / `multi_horizon_hit_rate`** — sweeping IC / hit-rate across `[1, 5, 10, 20]` forward periods is a dispatcher concern, not a per-cell metric. The in-metric horizon loop conflicted with `FactorProfile.identity` carrying `forward_periods` (#160 anti-shopping defense) and ran a second BHY path inside the metric (`metadata["p_adjusted_bhy"]`) parallel to `multi_factor.bhy(profiles, expand_over=["forward_periods"])`, the FDR SSOT. Both functions remain importable and runnable for one release cycle but emit `DeprecationWarning` on call and are excluded from `list_metrics` output (`_metric_index._DEPRECATED`). `run_metrics` auto-discover already skipped them via `_AUTO_DISCOVER_EXCLUDED` (per #147). Migration: `run_metrics(panel, cfg.replace(forward_periods=h))` per horizon → `compare(bundles)` for descriptive horizon-by-metric view, or `evaluate(...)` per horizon → `multi_factor.bhy(profiles, expand_over=["forward_periods"])` for FDR-controlled inference. Both paths are metric-agnostic — `mfe_mae` / `caar` / `oos` / `monotonicity` inherit horizon-sweep support automatically. Recipes in `docs/api/multi-horizon.md`. Removal version pinned at the next major-bump release-train. (#186)

### Removed

- **`factrix.multi_factor.bhy(p_stat=StatCode)` path** — replaced by `estimator=Estimator` (see Changed above). (#170)
- **`factrix.multi_factor.bhy(gate=...)` deprecation alias** — the v0.4 alias for `p_stat=` is removed alongside its successor; users still on `gate=` should jump directly to `estimator=NeweyWest()`. (#170)
- **`StatCode.NW_LAGS_USED`** — Newey-West auto-bandwidth lag count is no longer surfaced on `profile.stats`. The lag selection logic in `_resolve_nw_lags` is unchanged; the value just stops being externalised. Reinstating it under a dedicated `profile.metadata` (or sibling) channel is tracked as #188 — `_codes.py` was the wrong home (a hyperparameter-selection record, not a stat). (#187)

---

## v0.10.0 (2026-05-09)

Generalises v0.9.0's regime-only dispatch primitive into an axis-agnostic `by_slice` (market / sector / decile / regime / any user-defined column), and demotes `by_regime` to a thin deprecation wrapper. The shape of the partition primitive turned out to be axis-independent — the only thing the regime layer added was an inner-join + time-bisection annotation step — so generalising preserved the v0.9.0 behaviour while opening up cross-section axes that previously required hand-rolled `partition_by` loops.

The deprecation also makes a long-standing semantic distinction explicit: the time-bisection fallback diagnoses a **structural break**, not a regime — a regime is a hypothesised latent state with cross-period recurrence, while structural-break dating is a one-shot change-point detection. `by_slice` requires users to compose their own label so that distinction surfaces at the call site, rather than being silently absorbed by a fallback.

### Added

- **`factrix.by_slice(metric, df, *, slice_col, how="left", ...)`** — axis-agnostic dispatcher for any panel-input metric (market / sector / decile / regime / any user-defined column). (#154)
  - **Label-on-panel convention** — the slice key is an existing column on the metric input rather than a separate `labels` DataFrame. Users typically already carry the partition key on the panel, and the join shape (date-keyed vs asset-keyed) varies by axis; demanding a separate frame would have made the common path harder than the niche one.
  - **Universe-overlap composition is user-side** — superset / multi-membership / hierarchical / sliding-window / cross-product patterns are expected to be composed via `pl.concat` rather than dispatcher kwargs. Baking helpers into the API would multiply parameter surface for niche cases without simplifying the common path; reference idioms are documented in `docs/api/by-slice.md`.
  - **Null-label safety** — null values in `slice_col` raise rather than silently bucket into a `"None"` slice that would collide with literal `"None"` labels. Numeric label columns are stringified by `partition_by` and documented as such.
  - **Layered with v0.9.0 inference fixes** — NW HAC and the v0.8 calendar-time reindex fix (#37) still apply per slice; no double-counting risk introduced by the dispatcher.

### Deprecated

- **`factrix.by_regime`** — emits `DeprecationWarning`; removal scheduled for a future minor (tracked in a separate sub-issue).
  - **Internal shape** — now a thin wrapper that performs the inner-join + time-bisection regime annotation, then delegates partition + dispatch to `by_slice`. One shared `_slice_by_regime` primitive eliminates the silent-divergence risk between two near-identical dispatchers over the deprecation window.
  - **Layer-B unaffected** — `regime_ic` / `regime_caar` consume the private annotator directly and never trip the warning, so per-regime inference workflows do not need migration in this release.

  Migration:

  ```python
  # before
  by_regime(metric, df, regime_col="regime", ...)
  # after — explicit inner-join preserves the old semantics
  by_slice(metric, df.join(regimes, on="date", how="inner"), slice_col="regime", ...)
  ```

  See `docs/api/by-regime.md` for the full migration recipe, including why the time-bisection fallback (kept for behavioural compatibility) is structural-break diagnostics rather than a regime test. (#154)

### Fixed

- **`compute_mfe_mae` Polars schema dict** — dtypes were passed as class references (`pl.Float64`) rather than instances (`pl.Float64()`), which Polars 0.20+ rejects in `schema=` kwargs.
  - Surfaced under the v0.9.0 mypy gate (#114) before reaching users; no behavioural change for callers, but unblocks future Polars upgrades that tightened the runtime check.

### Changed

- **`compute_ic` per-regime / per-horizon return shape typed as `TypedDict`** — internal-only refactor, no signature change.
  - Replaces ad-hoc `dict[str, Any]` return annotations with structured types so downstream Layer-B callers (`regime_ic`, future `regime_fama_macbeth`) get IDE / mypy support without runtime cost.

## v0.9.0 (2026-05-07)

Regime-stratified analysis without re-implementing every metric, plus completion of the introspection symmetry started in v0.8.0. The headline `factrix.by_regime` is a generic dispatcher: hand it any panel-input metric and a regime labelling and it returns per-regime results without baking regime semantics into each metric's signature. Layer-A (`by_regime`) intentionally emits **no** cross-regime test — a generic χ²/Wald would over-claim for non-t-stat metrics like Sharpe, turnover, hit_rate, or monotonicity ρ. Cross-regime inference lives in metric-specific Layer-B wrappers; only `regime_ic` ships in v0.9.0.

### Added

- **`factrix.by_regime(metric, df, *, regime_col="regime", ...)`** — generic regime dispatcher for any panel-input metric. Convention-based: positional `metric` callable + `df` + forwarded kwargs. Inner-join semantics drop unlabeled dates rather than coercing them; explicit `TypeError` on scalar-input metrics (e.g., `breakeven_cost`, `net_spread`) directing users to compose inputs first. Time-bisection fallback when `regime_col` is absent emits a `UserWarning` with `stacklevel=3` so warnings surface at the user's call site. NW HAC and the v0.8 calendar-time reindex fix (#37) still apply per slice; no double-counting risk. (#112)
- **`factrix.metrics.regime_ic`** — Layer-B regime-aware IC with cross-regime Wald test, sharing an internal `_slice_by_regime` primitive with `by_regime` so the time-bisection fallback never diverges between the two layers. (#112)
- **`factrix.compute_forward_return(df, forward_periods=5)`** — promoted to public API. Pipelines that `adapt()` then call non-default preprocess (winsorize, abnormal return) needed a stable public entry point. Contract: panel in, panel with `forward_return` appended out; null-forward rows dropped. No internal kwargs leak. (#91)
- **`factrix.list_metrics` `import_path` + `input_kind` fields** — closes the v0.8 gap where `list_metrics` named the metric but not how to import it. `input_kind ∈ {"panel", "scalar"}` is the discriminator a user needs to decide whether `by_regime(metric, df, ...)` is even legal; `import_path` is `factrix.metrics.<module>`, copy-paste-ready for agent-driven pipeline wiring. (#113)
- **`evaluate(panel, config, *, factor_col=...)`** — signal-column alias for batch screening. Loop `evaluate(panel, cfg, factor_col=name)` over candidate columns without per-iteration `.rename()`. Hard `ValueError` on collision with an existing `"factor"` column — no silent shadowing.
- **`SuggestConfigResult.diagnose()`** — code-resolution helper completing the symmetry with `evaluate()`'s diagnose path. Renders WHY for any code key (warning, info, stat) without grepping `_procedures.py`.
- **`StatCode.description`** — final leg of the description-on-every-code-enum symmetry started in v0.8.0 (`WarningCode.description`, `InfoCode.description`).
- **Per-cell `stats_keys` in `describe_analysis_modes(format="json")`** — drawn from each `FactorProcedure.EMITS_STATS` (always-emitted ∪ conditionally-emitted possible-set). Agents can answer "is this gate reachable?" without running the procedure. Drift guard in tests ensures actually-emitted ⊆ declared.

### Changed

- **`compute_ic` regime slicing extracted to `_slice_by_regime`** — internal refactor consumed by both `compute_ic`'s regime path and `by_regime`. No user-facing signature change; eliminates the silent-divergence risk that two near-identical implementations would invite. (#112)
- **`docs/api/by-regime.md` + `docs/guides/regime-analysis.md`** — API page and analysis guide for the new dispatcher, including the explicit Layer-A / Layer-B contract decision and the reasoning behind the absent generic cross-regime test.
- **Architecture page registry cell table auto-generated from `_registry`** — replaces the hand-curated table that drifted twice in the v0.7 → v0.8 window. The doc renderer now reads the registry directly. (#95)
- **`docs/llms-full.txt` error-coverage expansion** — `WarningCode` / `InfoCode` / `StatCode` and `diagnose()` contracts are now fully covered, so downstream LLM consumers see the same surface as the Python API.
- **`CONTRIBUTING.md` drift-management section** — codifies the CI drift gates added in this release. (#104)
- **Python 3.13 classifier in `pyproject.toml`** — environment is tested on 3.12 and 3.13.
- **README logo** — replaced legacy SVG with a transparent PNG that renders correctly on both light and dark GitHub themes.

### Fixed

- **Public-surface validator extended to all docs pages** (#102) and to `llms-full.txt` (#96) — drift between registered cells and rendered docs is now caught pre-merge instead of in review.
- **Matrix drift / trigger-gap CI gate** (#89) — fails the build when the metric × cell matrix in docs falls behind `_registry` or when a registered procedure has no docs trigger.
- **Stale `preprocess` import path and `forward_periods` kwarg in guides** (#97) — guides referenced the v0.4 module path; updated to the v0.5+ surface.
- **Stale notebooks pruned from `docs/examples` on build** (#103) — removed notebooks no longer reachable from the docs nav were still being shipped.
- **Persona cross-cuts review followups** — event-study + TIMESERIES contracts (#105), equity-researcher input-schema + glossary (#106), final review pass (#110).
- **GitHub Release job** — CHANGELOG regex fix and `--latest` flag so release notes pick up the correct CHANGELOG slice and the GitHub Release "Latest" badge tracks the most recent tag.

## v0.8.0 (2026-05-07)

Magnitude-preserving CAAR semantics, programmatic metric discovery, and a calendar-time NW HAC fix. The `compute_caar` per-row formula now weights by factor magnitude rather than collapsing to sign — resolving the silent discrepancy flagged by `SPARSE_MAGNITUDE_DROPPED` (removed). Inference guardrails extended to the cross-asset axis (`n_assets` two-tier guard) to mirror the existing `n_periods` structure.

### Added

- **`factrix.list_metrics(scope, signal, *, format="text"|"json")`** — programmatic discovery of the standalone metrics applicable to a given analysis cell, closing the gap left by `describe_analysis_modes()` which only surfaces the registered cell procedure. Source of truth is the `Matrix-row:` annotation shared with the docs renderer; `Mode` is excluded because applicability does not vary across PANEL / TIMESERIES. (#79)
- **`SuggestConfigResult.detected: dict[str, Any]`** — structured panel observations (`scope`, `signal`, `mode`, `n_assets`, `n_periods`, `sparsity`), always present with stable types. Pipeline gates and AI agents can branch on shape without parsing `reasoning` strings. (#21)
- **`WarningCode.SMALL_CROSS_SECTION_N` / `BORDERLINE_CROSS_SECTION_N`** — two-tier `n_assets` guard mirroring the `n_periods` guard: `n_assets` 2–9 → SMALL (t-crit inflated 18–548% vs. asymptotic); 10–29 → BORDERLINE (5–15%); ≥30 → clean. Procedure runs at all `n_assets ≥ 2`; warnings surface inference-power decay without blocking execution. (#17)
- **`bmp_test(include_prediction_error_variance=False)`** — opt-in strict BMP (1991) denominator `σ_i · √(1 + 1/T_est)` for the mean-adjusted residual forecast; default `False` preserves the prior simplified denominator. (#48)
- **`WarningCode.SPARSE_MAGNITUDE_WEIGHTED`** — emitted when `compute_caar` or a sparse-panel procedure uses the Sefcik-Thompson (1986) magnitude-weighted estimator (factor is mixed-sign, not a clean ±1 ternary). Surfaces the estimator variant without changing inference for ternary inputs. (#48)
- **`compute_ic` `tie_ratio` column** — IC output schema widened from `(date, ic)` to `(date, ic, tie_ratio)` where `tie_ratio = 1 − n_unique / n` per date. Surfaced at median via `MetricOutput.metadata["tie_ratio"]` for `ic`, `ic_newey_west`, and `ic_ir`. Migration: code asserting the exact IC column list must accept the third column; column-by-name access is unaffected. (#48)
- **Documentation site** — MkDocs-based docs published from this release; `/latest/` and per-version URLs are stable from v0.8.0 onwards.
- **Visual identity** — logo banner and icon added to README and PyPI page. (#86)

### Changed

- **`compute_caar` preserves factor magnitude** — per-row formula changed from `return × sign(factor)` to `return × factor`. `{0, ±1}` callers see no change (sign is identity); non-ternary callers previously flagged by `WarningCode.SPARSE_MAGNITUDE_DROPPED` now receive the magnitude-weighted CAAR. To retain sign-only semantics on a non-ternary input, apply `.sign()` before calling. (#12)
- **Two-tier sample guards on `fama_macbeth`, `caar`, `top_concentration`** — replaces the prior single conservative block with `_HARD` (raise `InsufficientSampleError`) / `_WARN` (stat + `UserWarning`). The pre-v0.8 threshold refused results in the borderline regime where the math was valid. New hard floors: FM `MIN_FM_PERIODS_HARD = 4`; CAAR `MIN_EVENTS_HARD = 4`; top-concentration `MIN_PORTFOLIO_PERIODS_HARD = 3`. (#48)
- **`multi_split_oos_decay` drops `metadata["p_value"]`** (was `1.0`) — the decomposition output (`per_split`, `sign_flipped`, `status`) is the message; the placeholder invited accidental routing into BHY / gate logic that expects a real probability. (#48)
- **`MIN_IC_PERIODS` → `MIN_ASSETS_PER_DATE_IC`** — the old name implied a time-series length; the constant has always guarded per-date asset counts in IC computation. Migration: update the import; no deprecation alias. (#19)
- **`WarningCode.UNRELIABLE_SE_SHORT_SERIES` → `UNRELIABLE_SE_SHORT_PERIODS`** — vocabulary aligned with the `n_periods` parameter canonical since v0.5; both Python identifier and serialised string value change. Migration: update imports and any string-based log / alert filters. (#19)
- **`examples/` reorganised into focused per-recipe notebooks** — `demo.py` / `demo.ipynb` replaced with `multi_factor_screening.ipynb` (BHY family partitioning + cross-family pitfall) and `stock_factor_evaluation.ipynb` (individual_continuous IC, ~80% use case). CI executes every notebook on push to catch API drift before it reaches docs. (#14)

### Fixed

- **NW HAC lag bias in `(INDIVIDUAL, SPARSE, PANEL)`** — event-date-indexed series was fed directly into NW HAC, which assumes consecutive observations are 1 period apart. Sparse events (gap > `forward_periods`) over-corrected an MA overlap that did not exist; clustered events (gap < `forward_periods`) under-corrected the real overlap. Series now reindexed to full calendar and zero-filled before NW HAC (calendar-time portfolio approach: Jaffe 1974, Mandelker 1974, Fama 1998 §2). Applies to all four NW-HAC PANEL procedures (IC / FM / CAAR / common-sparse). `FactorProfile.n_obs` and `StatCode.NW_LAGS_USED` now report dense-series counts; `StatCode.CAAR_MEAN` is unchanged. (#37)
- **`(COMMON, SPARSE, PANEL)` event-count guard** — two-tier guard added on broadcast event count: `n_events < 5` raises `InsufficientSampleError`; `5 ≤ n_events < 20` emits `WarningCode.SPARSE_COMMON_FEW_EVENTS`. Previously only `n_periods` was guarded; a single-event broadcast produced a β silently. (#29)
- **`cross_section_tier` on inference-stage N** — `describe()` was reading pre-filter N for the cross-section-size warning instead of the post-filter N actually seen during inference. (#83)
- **Friendly `ConfigError` on `evaluate()` without config** — previously raised an opaque `AttributeError`; now raises `ConfigError` listing the four factory methods. (#72)

### Removed

- **`WarningCode.SPARSE_MAGNITUDE_DROPPED`** — warned that `compute_caar` would collapse magnitude via `.sign()`; with the magnitude-preserving rewrite, no routing drops magnitude. Migration: remove membership checks for this enum value; `compute_caar` on `{0, R}` inputs now preserves magnitude. (#12)
- **`SuggestConfigResult.detected["magnitude_dropped"]`** key — same root cause as `SPARSE_MAGNITUDE_DROPPED`; `_detect_signal` reduced to a 3-tuple. Migration: delete any branch reading this key. (#12)
- **`factrix[charts]` and `factrix[mlflow]` optional extras** — the corresponding source modules were already stripped in v0.5.0; the extras remained as install-only stubs pulling `plotly` / `mlflow` with no factrix code consuming them. Migration: install `plotly` or `mlflow` directly if your project needs them; integration adapter patterns are tracked in #88.

## v0.7.0 (2026-05-07)

Closes the silent-coercion gap in sparse-procedure dispatch. Until now, a user feeding a sparse-but-continuous signal (SUE z-score, ratings notch delta, event-day return, order-flow imbalance burst, earnings revision delta — anything where magnitude is the research target) was silently routed to `Signal.SPARSE` purely on zero-ratio, then had their magnitude information discarded inside `compute_caar` / `bmp_test` via `pl.col(factor).sign()`. No warning, no info note, no way to know without reading the source. This release makes the coercion *visible* without changing it; the broader axis-design question — whether to add a magnitude-weighted sparse procedure family — is tracked separately (#12) and intentionally **not** bundled here.

### Added

- **`WarningCode.SPARSE_MAGNITUDE_DROPPED`** — emitted by `suggest_config(...)` when `_detect_signal` detects a SPARSE-shaped factor whose non-zero values are not strictly in {-1, +1}. Users see, before running anything, that CAAR / BMP will collapse magnitude to sign, and can rescale to ±1, route to a continuous procedure, or knowingly accept the sign-only semantics (#8).

### Changed (docs)

- **`compute_caar` docstring** now states the `.sign()` coercion in a dedicated `Note:` block and updates the `factor_col` argument description. Behavior is unchanged — the sign-only semantics has always been the contract; the docstring just no longer hides it.

### Migration

No code changes required. If `suggest_config(...).warnings` now contains `WarningCode.SPARSE_MAGNITUDE_DROPPED`, your factor is being treated sign-only by CAAR / BMP — this was already the behavior in prior releases, you just couldn't see it. To preserve sign-only semantics: ignore the warning. To use magnitude: pre-multiply your factor to ±1 by another rule, or wait for the magnitude-weighted sparse procedure tracked in #12.

## v0.6.0 (2026-05-03)

Time-series shape diagnostics + a statistical infrastructure layer that makes them, and future Wald-based metrics, p-value-comparable with the existing `ts_beta` family. Plus a quiet but load-bearing FDR-control fix for batch BHY: `forward_periods` is now part of the family key, so mixing horizons in a single `bhy()` call no longer silently dilutes the step-up threshold.

### Added

- **`ts_quantile_spread` + `ts_asymmetry`** standalone diagnostics for `(COMMON, CONTINUOUS, *)` cells (#5). Both supplement the linear, symmetric OLS β assumed by `ts_beta_t_nw` — the first catches U-shape / inverted-U / extreme-only response via top-bottom bucket Wald, the second catches long-side ≠ short-side via either conditional means (method A) or piecewise slopes (method B). Three applicability gates (`distinct ≥ n_groups×2`, `both_sides_present`, `within_side_variance`) short-circuit with `metadata["reason"]` + redirect hint instead of silent NaN.
- **NW HAC multivariate OLS + Wald helpers** (`factrix/_stats/__init__.py`) — the joint-regression infrastructure under the new diagnostics, with HAC variance and joint Wald χ² so all three (`ts_beta_t_nw`, `ts_quantile_spread`, `ts_asymmetry`) emit p-values from the same framework and stay cross-metric comparable.
- **`docs/metric_applicability.md`** §`ts_quantile_spread / ts_asymmetry` applicability matrix and gate definitions; **README** §Document guide link to the new section.
- **README** use-case → factory reverse-lookup table for users not yet fluent in the three-axis vocabulary, plus a worked Bonferroni-then-BHY recipe for horizon-shopping correction.

### Fixed

- **`multi_factor.bhy()` family partitioning** now splits on `forward_periods` in addition to `(scope, signal, metric)`. Each horizon has its own null distribution and effective sample size; pooling them across horizons silently broke FDR control. Mixing horizons in one `bhy()` call now produces correctly-partitioned families.

### Changed (docs)

- Clarified that `forward_periods` is **rows on the time axis**, not calendar time — factrix is frequency-agnostic and shifts by row count. Aligned wording across README smoke-test callout, `AnalysisConfig` class + attribute docstrings, and `compute_forward_return` so IDE hover and README give the same answer. (Frequent confusion: users defaulted to a daily reading even on weekly / intraday panels.)
- Documented the **metric tier convention** (registry procedure vs standalone diagnostic) and softened user-facing terminology around cells / modes.

## v0.5.0 (2026-05-02)

Three-axis orthogonal API rewrite. Replaces the four `factor_type` strings + four parallel `Profile` dataclasses + `preprocess` / `factor` session / `ProfileSet` triad with a single `AnalysisConfig` (4 factory methods over `FactorScope × Signal × Metric`), a single `FactorProfile` result type, and a registry-SSOT dispatch (`factrix/_registry.py`). PANEL (panel, N≥2) and TIMESERIES (N=1) are now first-class equals — `(COMMON, *, N=1)` and `(INDIVIDUAL, SPARSE, N=1)` produce real `primary_p`, no longer pinned to `1.0`. Single-phase rip-and-replace per `docs/plans/refactor_api.md` §8 — no alias or deprecation cycle.

### BREAKING CHANGE

- **Public surface**: removed `fl.preprocess`, `fl.evaluate_batch`, `fl.factor()`, `fl.adapt`, `fl.validate_factor_data`, `fl.describe_profile`, `fl.describe_profile_values`, `fl.ProfileSet`, `fl.register_rule` / `fl.clear_custom_rules`. The new minimal surface is `fl.AnalysisConfig` + `fl.evaluate(panel, config)` + `fl.multi_factor.bhy(profiles, *, threshold=0.05)`.
- **Config**: `CrossSectionalConfig` / `EventConfig` / `MacroPanelConfig` / `MacroCommonConfig` removed. Construct via `AnalysisConfig.individual_continuous(metric=Metric.IC|Metric.FM)`, `.individual_sparse()`, `.common_continuous()`, `.common_sparse()`. `metric=Metric.FM` replaces `factor_type="macro_panel"` (the old name conflated data shape with research question).
- **New cell**: `(COMMON, SPARSE, None)` (`AnalysisConfig.common_sparse()`) was a coverage hole in v0.4 — now first-class for FOMC / policy / index rebalance broadcast events.
- **Profile**: `CrossSectionalProfile` / `EventProfile` / `MacroPanelProfile` / `MacroCommonProfile` collapsed into a single `FactorProfile` dataclass. Cell-specific scalars now live in `profile.stats: Mapping[StatCode, float]` keyed by enum (not string).
- **Field rename**: `Profile.canonical_p` → `FactorProfile.primary_p`. `Diagnostic` / `DiagnosticSeverity` removed; structured warnings now travel as `frozenset[WarningCode]` on `profile.warnings` (verdict-neutral).
- **Verdict**: `PASS_WITH_WARNINGS` removed. `Verdict` is binary `PASS` / `FAIL`. `warnings` / `info_notes` are surfacing-only — they never auto-rebind `primary_p` or upgrade `verdict()`.
- **TIMESERIES first class**: `(COMMON, *, N=1)` and `(INDIVIDUAL, SPARSE, N=1)` no longer return `primary_p = 1.0`. Real NW HAC t-tests on the underlying time series; `(INDIVIDUAL, SPARSE)` with the same N=1 user config and `(COMMON, SPARSE)` with N=1 collapse to the same procedure via the internal `_SCOPE_COLLAPSED` sentinel and tag the profile with `InfoCode.SCOPE_AXIS_COLLAPSED`.
- **PANEL invalid combos**: `(INDIVIDUAL, CONTINUOUS, *) × N=1` is mathematically undefined and now raises `ModeAxisError` with `suggested_fix=AnalysisConfig.common_continuous(...)` instead of silently degrading. `(INDIVIDUAL, *)` no longer accepts N=1 panels for CONTINUOUS metrics.
- **BHY**: `ProfileSet.multiple_testing_correct(p_source=, fdr=)` → `fl.multi_factor.bhy(profiles, *, threshold=0.05, gate=None)`. Family partitioning is automatic from the config triple — user no longer passes a group key; cross-family p mixing is structurally prevented.
- **Sample guards**: per-metric `MIN_FM_PERIODS = 20` / `MIN_TS_OBS = 20` unified into `MIN_PERIODS_HARD = 20` (raise `InsufficientSampleError`) and `MIN_PERIODS_RELIABLE = 30` (warn `UNRELIABLE_SE_SHORT_SERIES`) in `factrix/_stats/constants.py`. Procedures never silently produce a result on `n_periods < MIN_PERIODS_HARD`.
- **Errors**: `FactrixError` hierarchy — `ConfigError` → `{IncompatibleAxisError, ModeAxisError, InsufficientSampleError}`.
- **Removed v0.4 modules**: `_api.py`, `factor.py`, `config.py`, `validation.py`, `reporting.py`, `evaluation/pipeline.py`, `evaluation/profiles/`, `evaluation/profile_set.py`, `evaluation/diagnostics/`, `preprocess/pipeline.py`, `factors/`, `integrations/`, `charts/`, `metrics/redundancy.py`. `factrix/metrics/*` primitives kept — they back the v0.5 procedures.

### Added

- **API**: `factrix.AnalysisConfig` — three-axis frozen dataclass with 4 type-safe factory methods. `__post_init__` runs every construction path (factory, direct, `from_dict`) through one validation gate.
- **API**: `factrix.evaluate(panel, config) -> FactorProfile` — single dispatch entry point. Panel schema: `(date, asset_id, factor, forward_return)`; Mode is derived from `panel["asset_id"].n_unique()`.
- **API**: `factrix.multi_factor.bhy` — Benjamini-Yekutieli step-up FDR correction with automatic family partitioning. Same-test-family enforced by config triple, not user discipline.
- **Introspection**: `factrix.describe_analysis_modes(format="text"|"json")` reverse-queries the registry to print all legal cells + procedures + references. `factrix.suggest_config(panel)` heuristic-picks a factory call from a raw panel.
- **Codes**: `WarningCode`, `InfoCode`, `StatCode`, `Verdict` StrEnums (`factrix/_codes.py`) — structured replacements for stringly-typed diagnostic / metadata payloads.
- **Registry SSOT**: `_DispatchKey(scope, signal, metric, mode)` → `_RegistryEntry(procedure, use_case, refs)` mapping. Adding a cell touches one `register(...)` call. Bootstrap import at the bottom of `_registry.py` populates the registry before any first query.
- **Procedures**: 7 `FactorProcedure` classes in `factrix/_procedures.py` covering 5 PANEL cells (IC, FM, CAAR, COMMON×CONT, COMMON×SPARSE) + 2 TIMESERIES cells (TS-β CONTINUOUS, TS dummy SPARSE via `_SCOPE_COLLAPSED`).
- **Stats**: Hansen-Hodrick (1980) overlap floor `max(auto_bartlett(T), forward_periods - 1)` applied across all panel and timeseries cells with overlapping forward returns. Newey-West (1994) `auto_bartlett(T) = max(1, int(4 · (T/100)^(2/9)))` lag rule.

### Hardened (post-cut review fixes)

Applied during the v0.5 cut window before the surface was made public:

- `FactorProfile.n_assets: int` — panel cross-section width surfaced alongside `n_obs`. Disambiguates "small effective sample" between short series and thin cross-section. Visible in `diagnose()`.
- `multi_factor.bhy(gate=...)` requires a p-value `StatCode` and raises `ValueError` otherwise. Closes a footgun where `gate=StatCode.IC_T_NW` silently fed t-stats into BHY step-up. New `StatCode.is_p_value` property supports the validation.
- `multi_factor.bhy` emits `RuntimeWarning` when a batch yields ≥2 size-1 families (= no FDR correction power) — surfaces the cross-family no-op anti-pattern.
- `WarningCode` / `InfoCode` gain `.description` glosses, `IncompatibleAxisError` leads with the actionable factory list, registry adds a `_SCOPE_COLLAPSED` metric guard + post-import invariant assert (catches silent registration drift).
- `_route_scope(scope, signal, mode)` SSOT for the §5.4.1 sparse- TIMESERIES scope-collapse rule; `_evaluate`, `_describe`, and `_multi_factor.bhy` all reverse-call it (no parallel implementations).

### Renamed (terminology disambiguation)

Pre-1.0 readability sweep — no behaviour change:

- `MIN_T_HARD` / `MIN_T_RELIABLE` → `MIN_PERIODS_HARD` / `MIN_PERIODS_RELIABLE`. `InsufficientSampleError` kwargs `actual_T` / `required_T` → `actual_periods` / `required_periods`. Disambiguates `T` (time-series length) from `t` (Student's t-statistic) used in `*_T_NW` `StatCode` enums. `auto_bartlett(T)` and `*_T_NW` keep the literal `T` (direct citations of NW1994 and Student's t).
- `describe_analysis_modes(format="json")` row keys `mode_a_panel` / `mode_b_timeseries` → `panel` / `timeseries`, matching the `Mode.PANEL` / `Mode.TIMESERIES` enum values that already drove dispatch.
- README / ARCHITECTURE / docstrings drop the `Mode A` / `Mode B` marketing label in favour of the enum names; procedure code uses `n_periods` / `n_assets` consistently for dimension counts.

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

v0.4 → v0.5 was a single-phase rip-and-replace breaking change with no alias or deprecation cycle. Pin to commit SHA across the boundary.

---

### Note
First standalone release. Extracted from the `awwesomeman/factor-analysis` research workspace via `git filter-repo`; 149 commits of prior development history are preserved in this repository's git log.

Earlier version tags (`3.x`) existed only in the originating workspace and are not reproduced here — version numbering restarts from `0.1.0` to honestly reflect that the API is still iterating. The pre-extraction snapshot is anchored in the source workspace as the tag `pre-extraction-backup`.

### State at extraction
- **555 tests passing** across four factor types (cross-sectional, event signal, macro panel, macro common)
- `fl.evaluate` / `fl.evaluate_batch` / `fl.factor()` public API
- Typed `Profile` dataclasses (`frozen=True, slots=True`) with binary `verdict()` (PASS / FAILED) and `diagnose()` diagnostic list
- `ProfileSet` container with BHY multiple-testing correction (`multiple_testing_correct`)
- Artifacts retention via `return_artifacts=True` and `evaluate_batch(keep_artifacts=, compact=)`
- External rule registration: `register_rule` / `clear_custom_rules`
- `on_result` callback supports `bool | None` early-stopping
- Level 2 helpers integrated into Profile pipeline: `regime_ic`, `multi_horizon_ic`, `spanning_alpha`, orthogonalization
- Shared tradability primitives (`turnover` / `breakeven_cost` / `net_spread`) hoisted to `Factor` base class with per-call override support

### Known caveats
- `examples/demo.ipynb` stored outputs may reflect earlier quantile-spread field names (`q1_q5_spread`, then `long_short_spread` — rename landed 2026-04-20). Current field is `quantile_spread` / `spread_tstat` / `spread_p`; rerun the notebook to regenerate outputs against live code.
- `Factor Signal Analyzer` positioning: `turnover` / `breakeven_cost` / `net_spread` are idealized proxies (equal-weight, zero slippage) and do not represent tradable returns

## v0.4.0 (2026-04-25)

Trading-cost arithmetic overhaul: separates rank-stability turnover from notional position turnover, fixes per-period vs per-rebalance unit mismatch in the bps formulas, and renames `turnover_jaccard` → `notional_turnover` to describe the concept (Novy-Marx & Velikov τ) rather than the implementation (Jaccard set-similarity).

### BREAKING CHANGE

- `breakeven_cost` / `net_spread` now require `forward_periods` (kw-only) so per-period spread and per-rebalance turnover stay on the same time scale. Without it, h ≥ 2 factors were over-charged by N× and breakeven understated by N×. No default — a default of 1 would silently reproduce the buggy answer.
- `breakeven_cost` / `net_spread` numeric values shift on every factor that previously hit the rank-stability-turnover bug (CrossSectionalProfile from prior release; MacroPanelProfile and `Factor.evaluate()` in this release). Direction is optimistic for CS (breakeven rises) and pessimistic for MP (breakeven falls, because the rank-stability turnover overstated churn). Consumers comparing against stored thresholds must re-calibrate.
- `MacroPanelProfile` gains a required `notional_turnover: float` field; direct kwarg-construction breaks until callers add it.
- Identifier rename across the public API: `turnover_jaccard` → `notional_turnover` (primitive, MetricOutput name / cache key, `Factor` session method, Profile dataclass fields on both CS and MP, public export). Migration is mechanical find-and-replace.
- Rule code rename: `cs.high_turnover_jaccard` → `cs.high_notional_turnover`; `macro_panel.high_turnover_jaccard` → `macro_panel.high_notional_turnover`.

### Added

- **tradability**: `notional_turnover` (Novy-Marx & Velikov 2016 τ) separated from rank-stability `turnover` (1 − Spearman ρ). The two measure different things — middle-rank shuffling counts as turnover but not as notional churn — so only `notional_turnover`'s units align with the bps cost arithmetic. (Originally landed as `turnover_jaccard` in 2d005ff; renamed in this release.)
- **tradability**: `Factor.notional_turnover()` session method with the standard `n_groups` override + cache shape; mirrors `quantile_spread`. `n_groups` override on `breakeven_cost` / `net_spread` now also reroutes the turnover bucketing so spread and turnover stay consistent during a sensitivity sweep.
- **tradability**: `notional_turnover` exported from `factrix.metrics.__all__` (the prior `turnover_jaccard` name was never threaded into the public surface).
- **diagnostics**: `cs.high_notional_turnover` and `macro_panel.high_notional_turnover` rules (severity `warn`, threshold `notional_turnover > 0.5`). Sibling to the existing `cs.high_turnover`; both rules can fire independently — a factor with high mid-rank noise but stable Q1/Qn has high `turnover` yet low `notional_turnover` (still implementable).

### Fixed

- **tradability**: per-period vs per-rebalance unit mismatch in the bps formulas. `gross_spread` (per-period) was being subtracted from `2·cost·turnover` (per-N-period rebalance) — different time scales. `breakeven_cost ×= forward_periods`, `net_spread`'s `cost_drag /= forward_periods` to align both sides on the per-period scale.
- **tradability**: `MacroPanelProfile.from_artifacts` and `Factor.breakeven_cost` / `Factor.net_spread` were still feeding rank-stability `turnover` into the bps formulas despite the primitive's docstring forbidding it. Routed through `notional_turnover` instead — completes the wiring that 2d005ff applied only to CrossSectionalProfile.
- **tradability**: `MacroPanelProfile.turnover` is now sampled at `config.forward_periods` stride (was defaulting to lag-1). Mirrors `CrossSectionalProfile`. Diagnostic-only field; bps formulas are unaffected (they consume `notional_turnover`).

### Changed

- **tradability**: rename `turnover_jaccard` → `notional_turnover` throughout the public API. Renamed mechanically; no logic change. See BREAKING above for migration surface.

## v0.3.0 (2026-04-23)

### BREAKING CHANGE

- callers passing adf_check=False must migrate to
adf_threshold=None; callers passing adf_check=True can drop the argument (default unchanged) or pass adf_threshold=0.10.
- metadata n_dates -> n_pairs; short-circuit reason
no_valid_rank_autocorrelation -> insufficient_pairs.

### Added

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

### Fixed

- **trend**: short-circuit nan-only ic series before lstsq

### Changed

- **ic**: drop parallel list in regime_ic bhy step
- **_types**: centralise metric option literals

## v0.2.0 (2026-04-21)

### BREAKING CHANGE

- consumers must update `import factorlib` →
`import factrix` and `pip install factorlib` → `pip install factrix`.
- prepared panels now carry _fl_preprocess_sig (String)
instead of _fl_forward_periods (Int32). Downstream code reading that specific column name must update.
- fl.datasets.make_cs_panel(..., forward_periods=5)
and fl.datasets.make_event_panel(..., forward_periods=5) must be updated to signal_horizon=5. Package is at 0.1.0 with no external users.

### Added

- **profileset**: add diagnose_all, with_canonical, and layered logging
- **profiles**: add PASS_WITH_WARNINGS verdict and alternative p-values
- **preprocess**: widen strict gate to all preprocess-time fields
- **datasets**: add synthetic CS / event panels with calibrated IC

### Fixed

- **metrics**: guard ts_beta_sign_consistency at N<2
- **preprocess,evaluate**: strict-gate safety + fallback visibility

### Changed

- rename package from factorlib to factrix
- drop streamlit; lean pyproject deps; editable install
- **datasets**: rename forward_periods to signal_horizon

## v0.1.0 (2026-04-20)

### BREAKING CHANGE

- describe_profile_values(profile, artifacts, *,
include_detail) -> describe_profile_values(profile). Call-sites in tests / demo / README / docstrings updated to the single-arg form. Demo notebook rerun end-to-end: 56/56 cells green.
- EventProfile field bmp_sar_mean renamed to
bmp_test_mean; metric_outputs cache keys "mfe_mae" -> "mfe_mae_summary" and "bmp_sar" -> "bmp_test".
- Artifacts.metric_outputs cache keys renamed from
"ic_trend" to "caar_trend" (EventProfile) and "beta_trend" (MacroPanel/MacroCommon). CrossSectionalProfile unaffected.
- Profile fields long_short_spread / long_short_spread_vw
renamed to quantile_spread / quantile_spread_vw. Cache keys and MetricOutput.name identifiers renamed in lockstep.
- to test a different forward_periods, rebuild the
Factor session with a new config (fl.preprocess + fl.factor).
- q1_q5_spread → long_short_spread, q1_concentration →
top_concentration, plus intermediate columns q1_return/q5_return → top_return/bottom_return. Old names hard-coded Q1/Q5 but n_groups is configurable (CS default 10) — a quant reading q1_q5_spread under n_groups=10 was literally wrong. Prose / charts / diagnose messages propagated.
- metrics/oos.multi_split_oos_decay returns MetricOutput
(was OOSResult); event_around_return/multi_horizon_hit_rate/mfe_mae_summary return short-circuit MetricOutput instead of None.
- `validate_factor_data` previously required the date
column to be exactly `pl.Datetime("ms")` naive; it now accepts `pl.Date` or any `pl.Datetime(time_unit, time_zone)` variant. Callers relying on the strict-ms rejection must pre-cast themselves.
- `CrossSectionalConfig.q_top` removed. `q1_concentration`
now uses `q_top=1/n_groups` so its Q1 bucket matches the Q1 in `q1_q5_spread`. Previously the default `n_groups=10` + `q_top=0.2` made the two Q1 metrics disagree (concentration used top 20%, spread used top 10%) — a silent inconsistency that users had to notice and tune out.
- `Profile.from_artifacts(artifacts)` now returns
`tuple[Self, dict[str, MetricOutput]]` instead of `Self`. Direct callers (subclass authors, test helpers) must tuple-unpack.
- BaseConfig.multi_horizon_periods removed.
- CrossSectionalConfig.orthogonalize removed;
CrossSectionalProfile.orthogonalize_applied removed; cs.orthogonalize_not_applied diagnose rule removed.

### Added

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

### Fixed

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

### Changed

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
