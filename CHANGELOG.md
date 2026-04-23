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

---

## [0.3.0] - 2026-04-23

### Added
- `factrix/stats/bootstrap.py` — `stationary_bootstrap_resamples`
  (Politis-Romano 1994) + `bootstrap_mean_ci`. Default block length
  uses the Politis-White (2004) practical rule `L = 1.75 · T^(1/3)`
  when caller leaves it `None`; mean statistic fast-paths via axis=1
  reduction.
- `fama_macbeth(shanken_eiv=True, ...)` — Shanken (1992)
  errors-in-variables variance correction; `cluster="two_way"`
  enables Petersen (2009) two-way clustering.
- `bmp_test(adjust_clustering="kolari_pynnonen")` — Kolari-Pynnönen
  (2010) cross-sectional correlation adjustment for the BMP z-stat.
- `compute_mfe_mae` reports `mfe_z` / `mae_z` / `est_sigma` per event
  — estimation-window-normalised excursions (`σ̂ · √window` scale
  matches the order-statistic growth rate; Campbell-Lo-MacKinlay
  Ch 4) for cross-horizon / cross-vol comparison. `mfe_mae_summary`
  surfaces `mfe_z_p50` / `mae_z_p75` / `mfe_mae_ratio_z`.
- `regime_ic` / `multi_horizon_ic` — BHY adjustment per bucket;
  horizon-aware minimum-sample guards (`_scaled_min_periods`).
- `ic_trend(adf_threshold=...)` — pre-checks the input series and
  flags `unit_root_suspected` in metadata when ADF p exceeds the
  threshold (Stock-Watson 1988 spurious-trend protection).
- `top_concentration(weight_by="alpha_contribution")` — alpha-
  realised contribution variant alongside the existing
  `"abs_factor"` HHI weighting.
- `compute_mfe_mae(min_estimation_samples=...)` — exposes the BMP
  daily-σ floor as a kwarg so weekly-frequency callers can lower
  it without monkey-patching the module.
- `hit_rate_test` — exact binomial p-value path for small samples
  (was normal approximation throughout).
- `greedy_forward_selection` — emits one-shot `UserWarning` that
  selected-factor t-stats are inflated by stepwise selection
  (White 2000, Harvey-Liu-Zhu 2016); `suppress_snooping_warning=True`
  silences it (suppression itself logged at INFO).
  `ForwardSelectionResult.t_stats_inference_invalid=True` so callers
  can branch programmatically.
- `ProfileSet` — one-shot `UserWarning` when `filter(Expr)` /
  `rank_by` targets a `P_VALUE_FIELDS` column on a K≥2 set without
  prior `multiple_testing_correct`. Callable-filter path stays
  silent (escape hatch). `_warned_uncorrected` propagates through
  `_derive` so a chain warns exactly once.
- `KPSource` / `ShankenVarSource` / `ConcentrationWeight` Literal
  aliases now exported from `factrix._types` (single source of
  truth for the public option spellings; `from typing import
  Literal` dropped from each metrics module).

### Changed
- `quantile_spread_vw` defaults to `lag_weights=True` — pairs
  `weight[t-1]` with `forward_return[t→t+h]` to remove the
  contemporaneous-cap look-ahead trap. New `_lag_within_asset`
  helper handles the per-asset post-sampling lag pattern.
- `_newey_west_se` / `_newey_west_t_test` accept optional
  `forward_periods` and enforce `lags ≥ h-1` under MA(h-1) overlap
  (Hansen-Hodrick 1980) via shared `_resolve_nw_lags`. Default
  `⌊T^(1/3)⌋` unchanged when `forward_periods is None`.
  `ic_newey_west` / `fama_macbeth` thread `forward_periods`
  through; ad-hoc `max(⌊T^(1/3)⌋, h-1)` at each call site removed.
- `regime_ic` internal: drop the parallel `regime_order` /
  `raw_p_list` accumulators in favour of iterating
  `per_regime.values()` (relies on Python 3.7+ dict insertion
  order). Behaviour-preserving.

### Removed
- **BREAKING**: `ic_trend(adf_check: bool = True)` removed.
  Migrate `adf_check=False` → `adf_threshold=None`; `adf_check=True`
  → drop the kwarg (default `adf_threshold=0.10` matches the prior
  behaviour). Float in `(0, 1)` lets callers tune the unit-root
  cutoff; `None` skips ADF entirely.
- **BREAKING**: `turnover` realigned with the forward-return
  horizon. Sampled at stride `forward_periods` (Hansen-Hodrick
  non-overlap), self-join replaced with `shift().over(asset_id)`,
  optional quantile filter for tail-only AC. `metadata["n_dates"]`
  → `metadata["n_pairs"]`; short-circuit reason
  `no_valid_rank_autocorrelation` → `insufficient_pairs`.
  Minimum dates raised to `2·h+1`.

### Fixed
- `ic_trend` short-circuits on all-NaN input series (e.g. from a
  constant factor whose per-date rank correlation is degenerate)
  instead of flowing into `theilslopes` / `_adf` and tripping
  LAPACK DLASCL.

### Docs
- `docs/statistical_methods.md` — citation / attribution sweep:
  Andrews (1991) vs Newey-West (1994) vs Andrews-Monahan (1992)
  distinctions, Brown-Warner (1985) framing, Hirschman (1945) /
  Herfindahl (1950) ordering, Kolari-Pynnönen scope, Corrado SE
  derivation, Richardson-Stock framework non-adoption, FFJR
  attribution.
- `docs/redundancy_matrix.md` — flag conditional-redundancy blind
  spot of the unconditional metric; point readers to
  `spanning_alpha` and cite Barillas-Shanken (2017).
- `README.md` — refined project positioning, scope, document
  structure.
- `docs/plans/` — migrated project planning and spike
  documentation.
- `CONTRIBUTING.md` — instructions for changing the git
  Signed-off-by signature.

---

## [0.2.0] - 2026-04-21

### Added
- `ic_newey_west()` — HAC t-test on overlapping IC series; first-class
  alternative when `signal_horizon > rebalance_freq`.
- `CrossSectionalProfile.ic_nw_p` (in `P_VALUE_FIELDS`); diagnostic rule
  `cs.overlapping_returns_inflates_ic` with `recommended_p_source`.
- `MacroCommonProfile.factor_adf_p` (metadata — NOT in `P_VALUE_FIELDS`,
  unit-root test ≠ factor significance); rule `macro_common.factor_persistent`.
- Minimal in-house ADF (`_stats._adf`) with MacKinnon asymptotic crits —
  no `statsmodels` dependency.
- `ProfileSet.diagnose_all()` — flatten all diagnostics to a polars DataFrame.
- `ProfileSet.with_canonical(field)` — rebind canonical p-source for
  downstream `multiple_testing_correct` and `canonical_p` alias column.
- `Diagnostic.recommended_p_source` — rules can name a whitelisted
  alternative p-value; validated against `P_VALUE_FIELDS`.
- `Verdict` extended with `"PASS_WITH_WARNINGS"` — emitted when `canonical_p`
  passes but a warn-severity diagnostic names an alternative the user
  has not adopted.
- `factrix._logging` — shared `factrix.evaluation` (orchestration;
  INFO / WARNING) and `factrix.metrics` (correction layer; DEBUG +
  degenerate-sample WARNING) loggers.
- `ARCHITECTURE.md` — "Diagnostics vs Canonical" section describing the
  "framework detects, user decides" philosophy.
- Synthetic CS / event panel datasets with calibrated IC under
  `factrix.datasets`.
- `CONTRIBUTING.md` — factorlib (now factrix) dev workflow guide.

### Changed
- `verdict()` now delegates to `_verdict_with_warnings`; can return
  `"PASS_WITH_WARNINGS"`, and can raise `ValueError` when a registered
  `Rule.recommended_p_source` points outside `P_VALUE_FIELDS` (developer
  error, not user data issue).
- `multiple_testing_correct` records the resolved field name in
  `mt_p_source` when `with_canonical` has been applied, not the literal
  `"canonical_p"` alias.
- Strict-gate safety + fallback visibility hardened in
  `preprocess` / `evaluate`.
- `ts_beta_sign_consistency` guarded at N<2.
- README reframed; statistical methods + metric applicability extracted
  to standalone docs.
- Metrics formula docstrings expanded.
- Dropped `streamlit` dependency; pyproject deps slimmed; editable
  install path documented.

### Removed
- **BREAKING**: package renamed `factorlib` → `factrix`. Update imports.
- **BREAKING**: `PASS_WITH_WARNINGS` introduced as a new `Verdict` value;
  callers exhaustively matching the previous two-valued enum need to
  handle the third state.
- **BREAKING**: preprocess strict gate widened to all preprocess-time
  fields; previously silent fallbacks now raise / warn.
- **BREAKING**: `datasets.forward_periods` renamed to
  `datasets.signal_horizon`.

---

## [0.1.0] - 2026-04-20

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
