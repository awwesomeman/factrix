# Changelog

All notable changes to **factrix** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR**: API-breaking changes (field rename, signature change, removed metric)
- **MINOR**: New metrics, new Profile fields, new optional parameters
- **PATCH**: Bug fixes, docstring/test fixes, internal refactors

While the version is below `1.0.0`, the public API should be considered unstable — breaking changes may occur in **MINOR** bumps. Consumers are expected to pin by commit SHA (e.g., via git submodule) rather than relying on semver range constraints until `1.0.0` is cut.

Entries link to **PR number** (e.g. `(#123)` — the PR that landed the change) from v0.14.0 onwards. Entries dated v0.13.0 and earlier linked to issue numbers (some v0.13.0 entries are mixed because the convention shift landed alongside that release) and are kept as-is.

---

## [Unreleased]

### Added

- **`fx.inspect_panel(panel) -> PanelInspection`** (#443). Typed pre-flight introspection combining axis detection with a per-metric usability verdict against the inspected panel's shape. `PanelInspection` carries `detected: PanelProperties` (enum-fielded `scope` / `signal` / `mode` + `n_assets` / `n_periods` / `sparsity` numerics), `reasoning: PanelReasoning` (per-axis prose), `metrics: list[MetricApplicability]` (one verdict per `visibility=PUBLIC` spec with `usable: bool` / `warnings: list[Warning]` / `blockers: list[str]`), and `warnings: list[Warning]` (panel-level sample-shape diagnostics with `source=None`; per-metric degraded warnings live on each `MetricApplicability`). Two-stage applicability: (1) `Cell.matches(scope, signal, mode)` — mode is integral so a metric whose cell declares `mode=Mode.PANEL` (e.g. IC) is unusable on a single-asset panel, not just degraded; (2) `MetricSpec.sample_floor: SampleFloor | None` — declarative `min_*` / `warn_*` thresholds against panel shape. `min_*` violations make the spec unusable (`blockers`); `warn_*` violations attach a degraded `Warning` while keeping `usable=True`. Caller partitions via list comprehension: `usable = [m for m in info.metrics if m.usable]`. The flat list intentionally does not mirror `MetricResultGroup` (#441) — pre-flight `applicability` and post-execute `role partition` are different concepts; symmetry would be false. `suggest_config` / `SuggestConfigResult` are doomed by #438 unification and stay untouched here per the retirement-lifetime rule; both APIs coexist during the overlap window with `inspect_panel` sharing the detection helpers via direct import. `_repr_html_()` for Jupyter.
- **`Cell.matches(scope, signal, mode=None)`** (#443). Gains optional `mode` argument so callers can enforce mode applicability. Existing `list_metrics` callers stay backward compatible (default `None` skips the mode axis check). Cell-match was previously a 2-axis predicate that ignored mode — `inspect_panel` needs the 3-axis variant so a metric whose cell declares `mode=Mode.PANEL` is correctly rejected on a single-asset panel.
- **`MetricSpec.sample_floor: SampleFloor | None`** (#443). New optional field carrying per-metric pre-flight thresholds (`min_periods` / `warn_periods` / `min_assets` / `warn_assets`). `inspect_panel` evaluates these against `PanelProperties` to surface per-metric usability without running the metric. Populated on `ic_newey_west` / `ic_ir` (representative demonstration that the constants flow from `factrix._stats.constants.MIN_PERIODS_HARD` / `MIN_PERIODS_WARN`); broader population across the metric surface is tracked as a follow-up audit.
- **`DagExecutor` + `CycleError`** (#442). New `factrix._dag.DagExecutor` runs a closed set of `MetricSpec` against `(panel, cfg, factor_cols)`: topologically orders the graph via `MetricSpec.requires` (raising `CycleError` on cycles), dispatches `batchable=True` producers once per batch and `batchable=False` callables once per factor on a thin projection, runs every producer exactly once with downstream consumers reading from a per-`(spec, factor)` cache, and propagates short-circuit `MetricOutput` (`value=NaN` + `metadata["reason"]`) to downstream consumers without invoking them — the downstream `MetricOutput` carries `metadata={"reason": "upstream_unavailable", "upstream": ..., "upstream_reason": ..., "consumer_param": ...}` and the bundle's `warnings` list gains a matching `Warning(code=UPSTREAM_UNAVAILABLE, source=metric_name, message=upstream_reason)`. The executor takes an optional `fn_resolver: Callable[[str], Callable]` so tests can map spec names to locally-defined callables without building a fake module tree. Wiring as the default for `fx.evaluate` / `fx.run_metrics` is the unification follow-up.
- **`EvaluationResult.plan: str` field** (#442). Multi-line numbered topological order emitted by `DagExecutor.execute`, each line annotated with `[batchable]` / `[per-factor]` and a `requires=` upstream list when present. Required (no default) so tests / manual constructions cannot silently pass an empty placeholder that obscures whether the DAG actually ran. Surfaces in `to_dict()` under the top-level `plan` key and as a collapsible block in `_repr_html_()`.
- **`WarningCode.UPSTREAM_UNAVAILABLE`** (#442). Fires when the DAG executor short-circuits a downstream consumer because of upstream short-circuit; carries the original cause in the downstream `MetricOutput.metadata`.
- **Registry-load validation for `MetricSpec.requires`** (#442). `factrix._metric_index._load_module_specs` now calls `_validate_requires` per spec: every key in `requires` must be a parameter of the consumer callable, every value must be callable and must have its own `MetricSpec` in its module's `__metric_specs__` tuple. Replaces the runtime `@batch_primitive` / `@ic_consumer` validation that #440 retired — typos and orphan producers raise `ValueError` at import time.
- **`EvaluationResult` / `MetricResultGroup` / `Warning` result dataclasses** (#441). The unified result-type group that #438 / #442 will surface from the DAG executor lands as types-only in this release; runner wiring (replacing `FactorProfile` / `MetricsBundle` returns) is the follow-up. `EvaluationResult` holds bundle-level identity (`factor` / `axes` / `mode` / `n_obs` / `n_assets`) plus a `MetricResultGroup` and a flat `list[Warning]`. `MetricResultGroup` mirrors the `MetricGroups` shape (`applicable` / `primary` / `diagnostic` as `list[MetricSpec]` partitions) and exposes dict-like access (`__getitem__` / `keys` / `values` / `items` / `__iter__`) to the produced `MetricOutput` instances, all keyed by metric name. `Warning` is a flat `(code, source, message)` dataclass — `source=metric_name` for per-metric warnings, `source=None` for bundle-level. Serialisers: `EvaluationResult.to_df()` for long-form `pl.DataFrame` (stable schema for parquet stacking across factors), `to_dict()` for JSON-friendly nested dicts (NaN / Inf scrubbed to `None`), and `_repr_html_()` for Jupyter. The DoD `plan: str` field is deferred to #442 where the DAG executor that populates it actually exists (no placeholder string).
- **`MetricOutput.spec: MetricSpec | None`** (#441). Back-pointer to the declaring `MetricSpec`. Default `None` so all existing call sites stay valid; runners stamp the field at dispatch time so downstream consumers can recover the spec without a name-keyed lookup.
- **`MetricSpec.requires` / `MetricSpec.batchable` / `MetricSpec.visibility` + `Visibility` enum** (#440). Three new typed fields on `MetricSpec` replace the side-table machinery used by the dispatcher: `visibility=Visibility.INTERNAL` marks stage-1 producers excluded from `list_metrics` / `inspection.metrics.applicable` (was `is_stage1=True`); `batchable=True` marks callables that accept `factor_cols=` and return `dict[factor, output]` (was the `@batch_primitive` decorator); `requires: dict[str, Callable]` declares upstream metric callables to be injected into the named consumer parameter (was the `@ic_consumer` decorator for IC consumers). The dispatcher consumes the new fields via `factrix._metric_index.spec_by_name()`. The dict-callable form for `requires` keeps refactors safe — typo at the producer reference raises `NameError` at import time, and IDE rename tracks both sites at once.

### Changed

- **`is_stage1=True` -> `visibility=Visibility.INTERNAL` on 9 stage-1 producer specs** (#440). `compute_ic`, `compute_caar`, `compute_fm_betas`, `compute_mfe_mae`, `compute_event_returns`, `compute_ts_betas` (×2), `compute_spread_series`, `compute_group_returns` migrate to the new field. The `is_stage1` field is removed outright (no transitional alias).
- **`@batch_primitive` / `@ic_consumer` decorators retired** (#440). `factrix.metrics.ic` / `quantile` / `monotonicity` previously imported these decorators from `factrix.metrics._protocol`; the markers now live on the corresponding `MetricSpec` (`batchable=True` and `requires={"ic_df": compute_ic}` respectively). The decorator module is deleted entirely.

### Removed

- **`MetricSpec.is_stage1` field** (#440). Replaced by `MetricSpec.visibility`. No deprecated alias.
- **`factrix/metrics/_protocol.py`** (#440). Module deleted entirely (`@batch_primitive`, `@ic_consumer`, `_BATCH_ATTR`, `_IC_ATTR`, `is_batch_primitive`, `is_ic_consumer`); their roles fold into `MetricSpec.batchable` / `MetricSpec.requires`.
- **`fx.evaluate(..., factor_col=...)` kwarg + scalar `FactorProfile` return** (#421). Replaced by `factor_cols: Sequence[str]` / `dict[str, FactorProfile]` (see migration table in `### Changed` below). The old `ValueError`s for `factor_col not found` and `'factor' and factor_col both present (ambiguous)` are gone; the new `UserInputError`s on `factor_cols` cover the empty / duplicate / missing-column cases. Sibling factor columns are dropped silently when projecting per factor (the explicit `factor_cols` list is the unambiguous intent).

### Changed

- **`greedy_forward_selection` releases backward-eliminated buffers immediately** (#436). `factrix.metrics.spanning.greedy_forward_selection` now `pop`s selected factors out of the internal `candidate_arrays` dict so `selected_arrays` becomes sole owner of each buffer; when `_backward_eliminate` later drops a factor from `selected_arrays`, the buffer is freed at that point instead of lingering until function return. Pins the invariant `candidate_arrays.keys() == remaining` via a new spy test. No public-API or behavioural change.

- **`fx.evaluate` signature unified with `fx.run_metrics`: `factor_cols: Sequence[str]` in, `dict[str, FactorProfile]` out** (#421). The single-factor `factor_col: str` / `FactorProfile` shape was asymmetric with `run_metrics`'s already-list/dict surface (#402) and made the per-factor Python-loop pattern (`[fx.evaluate(p, cfg, factor_col=c) for c in cols]`) the only multi-factor option. `evaluate` now accepts a list of column names and returns a dict keyed by them; every factor in one call shares the same dispatch cell (locked by `config`) and the same `Mode` (derived once from `panel["asset_id"].n_unique()`), so the returned profiles are comparable by construction. Per-profile `factor_id` is stamped from each column name automatically. **Migration**:

  | Before | After |
  |---|---|
  | `profile = fx.evaluate(panel, cfg)` | `profile = fx.evaluate(panel, cfg)["factor"]` |
  | `profile = fx.evaluate(panel, cfg, factor_col="alpha")` | `profile = fx.evaluate(panel, cfg, factor_cols=["alpha"])["alpha"]` |
  | `profiles = [fx.evaluate(panel, cfg, factor_col=c) for c in cols]` | `profiles = fx.evaluate(panel, cfg, factor_cols=cols)` |
  | `survivors = fx.bhy(profiles)` | `survivors = fx.bhy(profiles.values())` |

  Cross-factor compute sharing (IC stage-1 batch reuse, batch primitives) is **not** done at this layer yet — each factor is dispatched independently via a thin column projection. The protocol-class registry from #418 is the natural seam for that follow-up. `evaluate_chunked` / `evaluate_iter` siblings (paralleling `run_metrics_chunked` / `run_metrics_iter`) will follow on their own issues once the shared-compute path lands.

- **`fx.evaluate` IC cell shares stage-1 across factors** (#426). The post-#421 `evaluate(factor_cols=cols)` shape was a per-factor projection loop; IC stage-1 (`compute_ic`) ran N times even though `compute_ic(factor_cols=cols)` could merge it into a single polars query plan. Adds `FactorProcedure.compute_batch(panel, config, factor_cols)` as the dispatch seam — the six non-IC procedures inherit a default mixin that reproduces the old per-factor loop, the IC procedure overrides it to call `compute_ic(factor_cols=cols)` once and stitch per-factor HAC inference. `_evaluate` becomes a thin dispatcher that picks the procedure and stamps the `SCOPE_AXIS_COLLAPSED` info note on the returned dict; procedures stay blind to dispatch-layer concerns. Observed: `compute_s` drops 0.43× (≈2.3× faster) at 100-factor IC `evaluate` on the `small` preset, at a 1.4× peak RSS cost (the merged polars query holds N factor rank intermediates simultaneously). The chunked / iter siblings on the evaluate path remain follow-ups.

- **Dispatch protocol-class registry replaces marker cascade** (#418). The `run_metrics` dispatcher previously detected batch primitives by signature inspection (`"factor_cols" in inspect.signature(fn).parameters`, introduced in #407) and IC stage-1 consumers by membership in a hand-maintained `_IC_CONSUMERS` frozenset. Both paths drifted silently — renaming `factor_cols` to `factor_columns` demoted a batch primitive to a per-factor loop with no warning; adding a new IC consumer required remembering the central constant. New module `factrix/metrics/_protocol.py` exposes `@batch_primitive` / `@ic_consumer` decorators (both with decoration-time signature-agreement assertions so the marker and the signature stay in sync: `@batch_primitive` requires `factor_cols` in signature, `@ic_consumer` requires `ic_df` as the first positional parameter). Applied to `compute_ic` / `quantile_spread` / `monotonicity` (batch primitives) and `ic` / `ic_newey_west` / `ic_ir` (IC consumers). The dispatcher itself moves from a hard-coded if/elif/else cascade to a `_PROTOCOLS` registry of small `_DispatchProtocol` subclasses (`_IcConsumerProtocol` / `_BatchPrimitiveProtocol` / `_PerFactorProtocol`); adding a future dispatch role (e.g. `compute_fm_betas` stage-1 sharing) means appending a new protocol class to `_PROTOCOLS`, not editing the dispatcher body. Internal-only refactor — no user-facing behaviour change.

### Added

- **`fx.evaluate_iter` — per-factor streaming yield on the evaluate path** (#428). Evaluate-side mirror of `fx.run_metrics_iter` (#419): `factrix.evaluate_iter(panel, cfg, *, factor_cols) -> Iterator[tuple[str, FactorProfile]]` emits `(factor_id, FactorProfile)` in `factor_cols` order so flow-style callers (parquet sink, progress bar, early-stop loop) can act on the first factor before the dispatcher finishes the remaining `N-1`. Cross-factor work the procedure amortises — currently IC stage-1 (`compute_ic` across the batch, from #426) — still runs once before the first yield via the new `FactorProcedure.bind_batch` hook; per-factor HAC inference runs inside the yield loop, so the first `next(gen)` lands after one factor's inference rather than all `N`'s. `fx.evaluate(...)` is now `dict(evaluate_iter(...))` — eager and streaming paths share one dispatcher and the same cell × mode invariants. `UserInputError` / `ModeAxisError` raise eagerly (before the first yield); `InsufficientSampleError` surfaces mid-stream — the standard streaming-iterator trade-off.

- **`fx.evaluate_chunked` — memory-bounded factor batching on the evaluate path** (#427). Evaluate-side mirror of `fx.run_metrics_chunked` (#417): `factrix.evaluate_chunked(panel, cfg, *, factor_cols, chunk_size=None, base_cols=("date", "asset_id", "forward_return")) -> Iterator[dict[str, FactorProfile]]` splits `factor_cols` into chunks, narrows the panel per chunk (with `pl.LazyFrame` projection pushdown on the lazy input path), and yields each chunk's profile dict in turn. Peak RSS is bounded by `chunk_size`, not `len(factor_cols)` — same trade as `run_metrics_chunked`. Within a chunk the cell's procedure runs its normal `compute_batch` path (#426), so any cross-factor sharing on that cell (currently: IC stage-1 reuse) applies inside the chunk and is recomputed across chunks. `chunk_size=None` (default) targets ~25% of `psutil.virtual_memory().available`; the auto-sizer is extracted into the shared `factrix._chunk_size` module so both `*_chunked` APIs evolve together. `fx.evaluate` API is unchanged.

- **`fx.run_metrics_iter` — per-factor streaming yield** (#419). New `factrix.run_metrics_iter(panel, cfg, *, factor_cols, metrics=None) -> Iterator[tuple[str, MetricsBundle]]` emits `(factor_id, bundle)` in `factor_cols` order so flow-style callers (parquet sink, progress bar, early-stop loop) can act on the first factor before the dispatcher finishes the remaining `N-1`. Cross-factor work the dispatcher amortises — IC stage-1 (`compute_ic`) and every `@batch_primitive` metric — still runs once before the first yield via the new `_DispatchProtocol.bind` hook; the per-factor consumer pass then runs inside the yield loop, so the first `next(gen)` lands after one factor's per-factor metrics rather than all `N`'s. `fx.run_metrics(...)` is now `dict(run_metrics_iter(...))` — eager and streaming paths share one dispatcher and identical `factor_cols` / `metrics` semantics. Input validation stays eager (raised before the first yield); per-factor consumer failures surface mid-stream (the standard streaming-iterator trade-off). The `_DispatchProtocol` extension contract gains `bind` as the preferred override point — existing extensions overriding only `dispatch` keep working through the default `bind` wrapper.

- **`fx.run_metrics_chunked` — memory-bounded factor-batching iterator** (#417). New `factrix.run_metrics_chunked(panel, cfg, *, factor_cols, metrics=None, chunk_size=None, base_cols=("date", "asset_id", "forward_return")) -> Iterator[dict[str, MetricsBundle]]` splits `factor_cols` into chunks, narrows the panel per chunk (with `pl.LazyFrame` projection pushdown on the lazy input path), and yields each chunk's `dict[factor_id, MetricsBundle]` in turn. Peak RSS is bounded by `chunk_size`, not `len(factor_cols)`, so 1000-factor screens that would otherwise exceed RAM can stream through a fixed working-set budget. Within a chunk, full batch dispatch is preserved — IC stage-1 is shared across the chunk's factors and batch-native primitives still take one call per metric. `chunk_size=None` (default) targets ~25% of `psutil.virtual_memory().available` as the per-chunk peak budget; pass an explicit value to override. `fx.run_metrics` API is unchanged.

  Naming taxonomy added to `factrix/metrics/_helpers.py` module docstring: `(no suffix)` = batch-native unified, `_batch` = batch variant with single-factor sibling, `_chunked` = split-and-yield K chunks (new), `_iter` = per-factor streaming yield (sibling issue).

- **`bootstrap_mean_ci_batch` — vectorised stationary-bootstrap CI across factors** (#390). New `factrix.stats.bootstrap_mean_ci_batch(values: np.ndarray, ...) -> tuple[np.ndarray, np.ndarray, np.ndarray]` takes a `(n_factors, n_observations)` matrix and returns per-factor `(ci_low, ci_high, point)` 1-D arrays. One `(B, T)` block-index matrix is drawn once and shared across all factors per chunk, replacing the per-factor Python loop pattern. The K=1 case reproduces `bootstrap_mean_ci` bit-for-bit at the same seed (the index draw is identical). K-chunking is auto-sized to a 256 MB peak on the materialised resample tensor so peak RSS stays bounded as `n_factors` grows; pass `chunk_size=` to override. Single-factor `bootstrap_mean_ci` API unchanged.

- **bench harness CI smoke + release-flow baseline rerun hook + first reference baseline** (#383). Closes the long-term-maintenance leg of the multi-factor scale-out harness (#380): a CI `bench-tiny` job runs every scenario at the seconds-level `tiny` preset on PR / push to `main` so harness rot (import breakage, schema drift, scenario crash) fails the build immediately rather than being discovered at release time. A new top-level `Makefile` exposes `bench-bump`, which derives a `<os>-<arch>-<ram>g` machine-id from `platform` + `psutil`, runs the `small` target cold-cache, and writes JSONL into `bench/baselines/v<version>-<machine-id>/`; the target is wired into the release flow via `.github/PULL_REQUEST_TEMPLATE/release.md` (a "Reference baseline rerun" checkbox the release author must tick) and documented in `bench/README.md` under "Release flow". `bench/README.md` also gains "Ratio reading", "`cache_state` operating rules", "Synthetic-data caveat" (the generators approximate computational load, not statistical reality), and "Cross-machine rebaseline" sections — the last codifies the "same `git_sha`, both machines, record per-scenario ratio" protocol for surviving primary-baseline-machine retirement. The first reference baseline is committed under `bench/baselines/v0.14.0-linux-x86_64-125g/` with an index row in `bench/baselines/README.md` (the cloud runner is not the eventual 16 GB-laptop primary; the index notes it as a cross-machine anchor candidate until the laptop baseline lands).



### Added

- **`bhy_hierarchical` two-stage FDR verb + `simes_p` primitive** (#264). `factrix.multi_factor.bhy_hierarchical(profiles, *, group: str, estimator=None, q=0.05) -> Survivors` implements the Yekutieli (2008) procedure for factor sets with natural group structure (momentum / value / quality families, cross-region universes). Outer Benjamini-Yekutieli step-up on Simes group representatives controls *group-level* FDR ≤ `q`; inner BHY within each passing group controls *within-group* FDR ≤ `q`. The cell-level `adj_p` is the max-of-layers fold `max(outer_adj_p[g], inner_adj_p[i])` so the universal `Survivors` duality `survivor[i] iff adj_p[i] <= q` still holds — both layer signals are folded into one number rather than splitting into `q_outer` / `q_inner` kwargs that would break the contract. `factrix.stats.multiple_testing.simes_p(p_values)` lands as a standalone primitive (Simes 1986 global-null combiner, dominates Bonferroni `m * min(p)`, valid under PRDS). The verb closes the third leg of the v0.13 multi-factor-verb surface (alongside `bhy(expand_over=)` and `partial_conjunction`); the three are distinguished by *survivor unit* — pair / identity-joint / identity-group-then-within — and the new `docs/api/bhy-hierarchical.md` opens with the routing table so a factor-zoo researcher picks the right verb without leaving the docs. Three failure modes that would otherwise produce surface-valid output are now blocked at the call site: single-group input raises and points at `bhy()`; every-profile-is-its-own-group at `n >= 3` raises (group axis is near-unique, probably a continuous variable mistakenly passed); majority-singleton-group inputs emit `RuntimeWarning` (inner BHY on n=1 is a raw cutoff, outer Simes on n=1 equals that p — no FDR correction at either layer for those groups). Reuses `_resolve_family` for group-key validation (identity-shadowing rejected, missing-context-key surfaced fail-loud) and the existing `estimator=` selection path so a Newey-West / Hansen-Hodrick p-value can drive both layers without per-call reconfiguration.

  ```python
  import factrix as fx

  # "Which factor families show signal, and within those, which factors?"
  survivors = fx.multi_factor.bhy_hierarchical(
      profiles, group="family", q=0.05,
  )
  survivors.adj_p             # max-of-layers fold per cell
  survivors.n_tests           # {(family,): m_in_family} for every input group
  # per-survivor group label is profile.context["family"]
  ```

- **`MomentEstimator(Estimator)` sub-protocol + `GMM` Hansen J-test** (#191). A symmetric third Estimator layer alongside `HACEstimator`, for over-identifying-restriction tests on a multivariate moment-condition system. `MomentEstimator` adds `min_periods: int` and `compute(moments: np.ndarray, *, forward_periods: int) -> GMMResult`; `GMMResult` is a frozen dataclass parallel to `InferenceResult` carrying `j_stat` / `df` / `overid_p` / `n_moments` / `n_params` / `metadata` / `warnings` (no `stat_name` / `p_name` — the type itself implies the `(StatCode.J_GMM, StatCode.P_GMM)` pair). `factrix.stats.GMM` is the concrete instance: hand-rolled Hansen (1982) two-step efficient GMM in `factrix._stats.gmm` (no statsmodels dependency, matching the lean-dep pattern of `factrix._stats.hac`), pure over-identification (`n_params = 0`) only — parametric GMM is a forward hook. `StatCode.J_GMM` and `WarningCode.SINGULAR_WEIGHT_MATRIX` ship alongside (the latter distinguishes a rank-deficient long-run covariance from a generic short-sample warning, mirroring how `RECT_KERNEL_NEGATIVE_VARIANCE` separates Hansen-Hodrick's kernel-specific failure). `AnalysisConfig.moment_estimator: MomentEstimator | None = None` is wired through the four factory methods + `to_dict` / `from_dict` (backward-compatible with pre-#191 serialized configs that omit the key); `_moment_inference(cfg, moments)` mirrors `_hac_inference(cfg, series)` and stitches `GMMResult` into the procedure-layer `(stats, metadata)` contract keyed by `(J_GMM, P_GMM)`. Applicability gate runs at `__post_init__` time.

  ```python
  import numpy as np
  from factrix.stats import GMM

  # Construct a (T, K) moment matrix yourself — e.g. per-date IC at K
  # forward horizons, factor-sorted decile spreads, cross-asset shared-β
  # residuals, etc. Choice of moment system is a research-design call.
  moments = build_my_moment_system(panel, ...)   # shape (T, K)

  result = GMM().compute(moments, forward_periods=max_horizon)
  result.j_stat        # Hansen J statistic, χ²(K) under H₀
  result.overid_p      # right-tail χ² p-value
  result.df            # K - n_params (n_params = 0 in this release)
  ```

  **Why ship as a standalone primitive rather than a fully integrated cell**: HansenHodrick (#184) plugged into the existing IC PANEL cell because both NW and HH consume the same 1-D mean-IC series — same data shape, different kernel. GMM is fundamentally different: it consumes a multi-dimensional moment matrix, and the *choice of moment-condition system* (multi-horizon panel? multi-bucket spread? cross-sectional shared-β? — see Hansen 1982 §3) is a research-design question without one canonical factrix default. Forcing one would either pre-commit factrix to a single moment system or scaffold a configurable horizon grid that bloats the dispatch path well past #184's simplicity. So #191 ships the primitive + dispatch infrastructure; the integrated multi-horizon panel cell (auto-construction of `(T, K)` IC matrix from a raw forward-return panel + cell-registration / EMITS_STATS extension / horizon-grid spec) lands as a focused follow-up so its design — interacting with #255 (`list_estimators` cell-filter), #257 (provenance schema), and the moment-system choice — gets a clean review pass. Users who already have a moment system in mind can run J-tests today via the standalone API; users who want the integrated multi-horizon convention wait for the follow-up.

- **`HACEstimator(Estimator)` sub-protocol + cell-internal estimator swap** (#163). `Estimator` (the #170 selection protocol) gains a runtime-checkable sub-protocol `HACEstimator` adding `min_periods: int` and `compute(series: np.ndarray, *, forward_periods: int) -> InferenceResult`. `NeweyWest` and `HansenHodrick` implement it; the slice-test instances (`WaldNWCluster` / `WaldTwoWayCluster` / `BlockBootstrap`) stay on the selection-only base (their compute paths are multivariate). `AnalysisConfig` gains a fifth field `estimator: HACEstimator = NeweyWest()`, surfaced through the four factory methods as the new `estimator=` kwarg; `__post_init__` validates `estimator.applicable_to(scope, signal)` at construction time so `AnalysisConfig.common_continuous(estimator=HansenHodrick())` raises immediately rather than deep inside `evaluate`. `to_dict` / `from_dict` round-trip the estimator by name string via the new `factrix.stats.get_estimator(name)` registry helper; missing-`estimator` keys in legacy v0.11/v0.12 dicts fall back to `NeweyWest()`. `UnknownEstimatorError(ConfigError, ValueError)` lists every registered estimator on a name miss. IC PANEL / FM PANEL / CAAR PANEL procedures now route through `cfg.estimator.compute(...)` instead of hardcoded `_newey_west_t_test`; `profile.context["estimator"]` records the cfg-driven choice on every procedure for audit-time provenance (HLZ2016 spec-search defence — see "Why" below). Retires the v0.12.0 `ComputableEstimator` placeholder name from the #170 entry below: the design landed as the `HACEstimator` split (selection vs HAC-on-mean), not as a single computable extension; moment-condition estimators (GMM J-test, #191) and slope-axis HAC (TS β / TS Dummy) live on parallel sub-protocols when they land rather than overloading `HACEstimator.compute`.

  ```python
  # default — bit-equal v0.12 NW path
  cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
  profile = fx.evaluate(panel, cfg)
  profile.primary_stat_name  # StatCode.T_NW
  profile.context["estimator"]  # "NeweyWest"

  # swap to Hansen-Hodrick at study scope
  cfg = AnalysisConfig.individual_continuous(
      metric=Metric.IC, estimator=HansenHodrick(),
  )
  profile = fx.evaluate(panel, cfg)
  profile.primary_stat_name  # StatCode.T_HH
  profile.primary_p == profile.stats[StatCode.P_HH]  # True
  profile.context["estimator"]  # "HansenHodrick"
  ```

  **Why**: HLZ2016's spec-search defence is "don't pick an estimator after seeing results," not "always use a single estimator forever." v0.12 hardcoded NW + auto-side-emitted HH, which let downstream code cherry-pick whichever p was smaller. v0.13's design forces estimator choice to be cfg-scoped (study-level, not per-call) and stamps the choice in `profile.context` so audit-time review can see whether multiple estimators ran on the same study. Per-call `evaluate(panel, cfg, estimator=...)` is deliberately not opened — the cfg object is the spec-search lock. (Related follow-ups: `list_estimators(scope, signal)` cell-filter #255, third-party `register_estimator` #256, provenance asymmetry on slope-axis cells #257.)

- **`Examples:` blocks on every public metric callable** (#322). All 44 callables exported via `factrix.metrics.__all__` across 19 metric modules (`caar`, `clustering`, `concentration`, `corrado`, `event_horizon`, `event_quality`, `fama_macbeth`, `hit_rate`, `ic`, `mfe_mae`, `monotonicity`, `oos`, `quantile`, `spanning`, `tradability`, `trend`, `ts_asymmetry`, `ts_beta`, `ts_quantile`) carry an `Examples:` block placed last in the docstring per the NumPy trailing-section order sealed by #319. Examples follow the call-shape-over-fragile-output convention from #307 — output lines assert structural facts (column-set superset, `MetricOutput.name`, `t_stats_inference_invalid` flag), no concrete floats / DataFrame reprs. Where a metric consumes another callable's output from the same module (`caar` ← `compute_caar`, `fama_macbeth` / `beta_sign_consistency` ← `compute_fm_betas`, `ts_beta` / `mean_r_squared` / `ts_beta_sign_consistency` ← `compute_ts_betas`, `mfe_mae_summary` ← `compute_mfe_mae`), the Example chains from the upstream output rather than re-running setup. `pytest --doctest-modules factrix/metrics` (#314 CI runner) exercises 44 metric-page doctests on every push.

- **Per-module `__all__` as SSOT for the metrics API render surface** (#322). Every metric module under `factrix.metrics/` now declares an `__all__` in teaching order — the same order users encounter on the rendered API page (`compute_caar` → `caar` → `bmp_test`, `compute_fm_betas` → `fama_macbeth` → `pooled_ols` → `beta_sign_consistency`, etc.). Public return-type dataclasses (`spanning.SpanningResult` / `ForwardSelectionResult`, `oos.SplitDetail`) join the `__all__` of their owning module under the same convention used in `factrix.multi_factor.__all__` for `Survivors`. The `members:` list on each `docs/api/metrics/<mod>.md` mirrors `__all__` exactly (mkdocstrings-python does not auto-follow `__all__` when `members:` is omitted — its default filter is surface-by-prefix); `tests/test_metric_api_members_match_all.py` enforces the invariant `docs members: == module.__all__` across 19 modules so a future contributor cannot drift the two. Two callables that previously rendered on the API pages despite being internal coordination helpers — `hit_rate.per_date_series` (slice-test capability protocol implementation, dispatched via `factrix.metrics._metric_capabilities.resolve_per_date_series`) and `ts_beta.ts_beta_single_asset_fallback` (dispatch-registry N=1 fallback flagged in `_metric_index._STAGE1_HELPERS`) — are now correctly excluded from the rendered surface (kept out of `__all__`, kept out of `members:`).

### Changed

- **Internal `verb=` kwarg sweep on error raise sites** (#317). Every source-side `verb="..."` kwarg into `UserInputError` (and the helpers `_resolve_family` / `_expand_over_values` / `_resolve_p_value` / `_build_per_date_panel` / `_resolve_estimator` that thread it through) is renamed to `func_name=`. `UserInputError.__init__` drops the legacy `verb=` kwarg bridge added in #316 — the constructor signature is now `func_name=` only (keyword-only). No user-visible behaviour change: the public attribute is already `e.func_name` (since #316), the rendered error message form is byte-identical, and the rename is purely an internal source-side cleanup so the design-register `verb` token no longer survives at error-raise call sites. The slice-test internal error messages that interpolated the helper's `verb` arg (`f"{verb}: <2 aligned dates ..."`) now interpolate `func_name`; one adjacent prose mention inside a slice-test error string ("this verb currently supports WaldNWCluster ...") is also retargeted to "this function" for register consistency.

- **Contributing guide records the docstring-style boundary** (#313). `docs/development/contributing.md` gains two new policy sections under §8: (1) "Docstring style boundary" makes explicit that code formatting / line length / naming follow PEP 8 + the ruff configuration in `pyproject.toml`, while only the docstring section convention (`Args:` / `Returns:` / `Raises:` / `Warns:` / `Notes:` / `Examples:` / `References:`, plural, in that order) is taken from Google — the Google Python Style Guide as a whole (its 80-character limit, single-quote preference, yapf formatter) is **not** adopted; (2) "Markdown code-block intent layers" records that `pycon` blocks under `docs/api/**` are runnable autodoc-injected examples (copy-button strips `>>>`) while hand-authored `python` blocks with unbound names are illustrative schemas — both layers are intentional, and editors must verify the intent before "fixing" an illustrative block into runnable form. Establishes the contract the rest of #313 (NumPy-underline → Google sweep, plural unification, ruff `D` enablement) executes against. Pre-existing `### Metric docstring style` is retained as the metric-specific extension on top of this baseline.

- **`Survivors.n_total` / `bhy_adjust(..., n_total=)` / `bhy_adjusted_p(..., n_total=)` → `n_tests`** (breaking, #264). The BHY denominator field / kwarg sat alongside `FactorProfile.n_obs` / `n_periods` / `n_pairs` / `n_assets` under the same `n_*` prefix but answered a structurally different question — those four are sample-size axes (observation counts inside one cell), while the BHY denominator is the multiple-testing family size (count of hypotheses in the step-up). A reader scanning `survivors.n_total` would first read "total observations" before noticing the field is a `Mapping[bucket_key, int]`; the name actively misled. `n_tests` names the domain directly. Migration is a single-token find-and-replace at every reading site:

  ```python
  # before
  fx.multi_factor.bhy(profiles, q=0.05).n_total
  fx.stats.bhy_adjusted_p(p, n_total=1000)

  # after
  fx.multi_factor.bhy(profiles, q=0.05).n_tests
  fx.stats.bhy_adjusted_p(p, n_tests=1000)
  ```

  Unrelated `n_total` occurrences in event-study / hit-rate / Corrado / orthogonalize modules are genuine sample counts (events processed, rows retained) and stay untouched.

- **`primary_p` / `primary_stat_name` semantic — cfg-driven, not hardcoded NW** (breaking, #163). On IC PANEL / FM PANEL / CAAR PANEL the canonical pair now reflects whichever `HACEstimator` was wired on `cfg.estimator`, not the hardcoded `(T_NW, P_NW)` v0.12 wrote unconditionally. With default `NeweyWest()` cfg the values are bit-equal to v0.12; with `HansenHodrick()` they shift to `(T_HH, P_HH)`. Downstream callers reading `profile.stats[StatCode.T_NW]` or `profile.stats[StatCode.P_NW]` directly will `KeyError` when a non-NW estimator was used — read `profile.primary_stat` / `profile.primary_stat_name` / `profile.primary_p` instead (these stay populated regardless of estimator) and consult `profile.context["estimator"]` for provenance. TS β / TS Dummy / common-panel procedures are unchanged because they run NW HAC on an OLS slope or an iid cross-asset t (neither fits the HAC-on-mean `compute(series, *, forward_periods)` contract); a slope-axis sub-protocol is tracked separately. `UNRELIABLE_SE_SHORT_PERIODS` is now emitted uniformly on `0 < n_periods < 30` across IC / FM / CAAR — previously only the FM procedure's procedure-side guard fired (4 ≤ n < 30), so a 25-period IC analysis silently lacked the short-sample tag; the estimator-side emission consolidates this and the FM-side guard is removed as redundant.

  ```python
  # before (v0.12.0) — stats[T_NW] always populated
  profile = fx.evaluate(panel, cfg)
  t = profile.stats[StatCode.T_NW]  # safe under any cfg

  # after (v0.13.0) — read primary_stat to stay estimator-agnostic
  profile = fx.evaluate(panel, cfg_with_hh)
  t = profile.primary_stat                  # ← canonical pair, regardless of estimator
  code = profile.primary_stat_name          # StatCode.T_HH under cfg_with_hh
  who = profile.context["estimator"]        # "HansenHodrick" — provenance
  ```

- **`MetricOutput.n_obs` first-class field** (breaking, #248). Promoted from a metadata dict key to a first-class dataclass field — `n_obs: int | None = None` — so user / AI-agent consumers reach the metric primitive's sample size with `output.n_obs` instead of `output.metadata.get("n_obs")`. Builds on the `n_observed → n_obs` metadata rename from #246. `_short_circuit_output(name, reason, n_obs=n, ...)` callers (~38 sites) auto-route via kwarg-only signature change; direct `MetricOutput(...)` constructions in `factrix/metrics/spanning.py` and `factrix/metrics/fama_macbeth.py` are updated to pass `n_obs=` at the top level instead of nesting under `metadata={"n_obs": ...}`. `__repr__` now surfaces `n_obs=` between `value=` and `stat=` when populated. Same family name as `FactorProfile.n_obs` but scoped per metric primitive (single-stage estimator count) rather than per dispatched cell (final-stage test denominator).

  ```python
  # before (v0.12.0 — pre-rename)
  n = out.metadata["n_observed"]

  # intermediate (post-#246 metadata key rename, same v0.13.0 release)
  n = out.metadata["n_obs"]

  # after (#248, v0.13.0 final shape)
  n = out.n_obs
  ```

- **`FactorProfile.diagnose()` schema overhaul** (breaking, #246). Six UX gaps in the structured triage interface land in one schema change. (1) **Four sample axes** replace the previous polymorphic `n_obs` + `n_assets` pair: `n_obs` (cell-canonical final-stage test denominator — semantics unchanged), `n_pairs` (non-null `(period, asset)` pair count, first-stage), `n_periods` (unique periods in raw panel), `n_assets` (unique assets, semantics unchanged). Each axis answers one question and never overlaps with another; a small `n_obs` is now disambiguated by the three companion axes without reverse-engineering the cell. (2) **`primary_stat` / `primary_stat_name` family** added top-level — `primary_stat: float | None` carries the test statistic value paired with `primary_p` (e.g. `t_nw` value for an NW HAC t-test), `primary_stat_name: str` slugs the `stats` key (e.g. `"t_nw"`) so the user can connect `primary_p` to its `stats` entry without consulting the procedure registry. Generic across statistic families: the `None` arm handles future empirical-p primaries (block bootstrap) where there is no test stat. Invariant `stats[primary_stat_name] == primary_stat` (when not `None`) is pinned in docstring. (3) **`cell` group** wraps the dispatch coordinate — `{"scope", "signal", "metric", "mode"}` — so consumers identify which procedure ran without grepping `config`. (4) **Reader-flow key order**: `identity` → `context` → `cell` → sample axes → `primary_p` / `primary_stat` / `primary_stat_name` → `warnings` / `info_notes` → `stats` / `metadata`. (5) **`n_observed` → `n_obs` metadata key rename** across ~38 `factrix/metrics/*.py` short-circuit sites — the two first-class surfaces (`FactorProfile` and `MetricOutput`-metadata) now share the same family name; the `MetricOutput.n_obs` first-class field promotion is tracked separately at #248. (6) **`SuggestConfigResult.diagnose()` "symmetric" docstring** is downgraded to scope-limited — the two surfaces share `warnings` serialisation but answer structurally different questions; no schema change there. Docs (`api/factor-profile.md`, `development/architecture.md`, `llms-full.txt`) are rewritten in lockstep; bare `N` / `T` letter codes in `factor-profile.md` are replaced with full axis names to reduce reader cognitive load.

  ```python
  # before (v0.12.0)
  d = profile.diagnose()
  # → {"identity": {...}, "context": {...},
  #    "mode": "panel",
  #    "n_obs": 30, "n_assets": 500,
  #    "primary_p": 0.0046,
  #    "warnings": [...], "info_notes": [...],
  #    "stats": {...}, "metadata": {...}}

  # after (v0.13.0)
  d = profile.diagnose()
  # → {"identity": {...}, "context": {...},
  #    "cell": {"scope": "individual", "signal": "continuous",
  #             "metric": "ic", "mode": "panel"},
  #    "n_obs": 30, "n_pairs": 12450, "n_periods": 30, "n_assets": 500,
  #    "primary_p": 0.0046, "primary_stat": 2.84,
  #    "primary_stat_name": "t_nw",
  #    "warnings": [...], "info_notes": [...],
  #    "stats": {...}, "metadata": {...}}

  # Migration:
  # - d["mode"]      → d["cell"]["mode"]
  # - new keys n_pairs / n_periods / primary_stat / primary_stat_name carry
  #   additional info; old keys (n_obs / n_assets / primary_p) keep semantics.
  # - MetricOutput.metadata key "n_observed" → "n_obs" across metric primitives.
  ```

- **`Survivors.adj_q` → `Survivors.adj_p`** (breaking, #245). Aligned the adjusted-p-value column name with statistical-software conventions (R `p.adjust`, statsmodels `multipletests`) where adjusted p-values are uniformly named `adj_p` / `p_adj` regardless of whether the underlying procedure controls FWER (Bonferroni / Holm) or FDR (BH / BHY). The previous `adj_q` reflected an internal-consistency goal with the `bhy(q=0.05)` kwarg, but read awkwardly in FWER contexts (where the threshold is α, not q) and required first-time users to ask "what is `adj_q`?". The `q=` kwarg name is **kept** (it remains the API-uniform threshold name across procedure families); only the output column renames. `bhy_adjusted_p()` function name was already `_p` — this change extends the same convention to the survivor container field.

  ```python
  # before (v0.12.0)
  survivors = fx.multi_factor.bhy(profiles, q=0.05)
  for prof, adj in zip(survivors.profiles, survivors.adj_q, strict=True):
      ...

  # after (v0.13.0)
  survivors = fx.multi_factor.bhy(profiles, q=0.05)
  for prof, adj in zip(survivors.profiles, survivors.adj_p, strict=True):
      ...
  ```

### Removed

- **`UserInputError.verb` attribute** (breaking, #316). The user-facing attribute carrying the failing function name renames `.verb` → `.func_name`, aligning the error contract with the published user-facing register (see `contributing.md` §Two-register convention). The rendered error message form is byte-identical (`<func_name>(): unknown <field>=<value>`); only the source-side attribute name changes. User code catching `UserInputError` and reading `e.verb` will now hit `AttributeError` — replace with `e.func_name`. No deprecation cycle: factrix is pre-1.0 with no published user, and v0.x minor bumps are allowed to break compatibility under SemVer. The constructor accepts the legacy `verb=` kwarg without warning as an internal bridge until #317 sweeps the 59 source-side raise sites; this is *not* a user-facing back-compat alias.

- **Auto side-emission of `(T_HH, P_HH)` on default cfg** (breaking, #163). v0.12 IC PANEL / FM PANEL procedures populated `StatCode.P_HH` / `StatCode.T_HH` alongside `P_NW` / `T_NW` whenever `forward_periods > 1`, on every cfg, so callers could read either pair without re-running. Under the new dispatch (#163 Added entry above) only the `HACEstimator` wired on `cfg.estimator` populates its `(stat_name, p_name)` pair — default `NeweyWest()` cfg writes `(T_NW, P_NW)` only; `HansenHodrick()` cfg writes `(T_HH, P_HH)` only. Switch downstream code that hardcoded `profile.stats[StatCode.P_HH]` lookups to either (a) construct a separate `cfg_hh = AnalysisConfig.individual_continuous(metric=..., estimator=HansenHodrick())` and call `evaluate(panel, cfg_hh)` explicitly, or (b) read `profile.primary_p` + `profile.context["estimator"]` for the cfg-driven canonical pair. Family-verb selection — `bhy(profiles, estimator=HansenHodrick())` reading an already-populated `StatCode.P_HH` — still works against profiles produced under an HH cfg.

- **`FactorProfile.verdict()` and `Verdict` enum** (breaking, #243). `verdict(*, threshold=0.05, gate=None)` was a `primary_p < threshold` wrapper. Removed because (a) for N candidate factors, iterating `profile.verdict()` and counting passes is the spec-search anti-pattern factrix explicitly avoids — multi-factor decisions belong to `multi_factor.bhy` survivors, not per-factor threshold gates; (b) the `Verdict` `PASS / FAIL` outcome ignored emitted `WarningCode` (e.g. `UNRELIABLE_SE_SHORT_PERIODS`), letting unreliable inference report `PASS`. The `Verdict` enum is removed alongside the method.

  ```python
  # before (v0.12.0)
  if profile.verdict() is fx.Verdict.PASS:
      ...

  # after — single-factor pre-registered analysis only
  if profile.primary_p < 0.05:
      ...

  # after — N candidate factors: route through BHY, read survivors
  survivors = fx.multi_factor.bhy(profiles, q=0.05)
  for prof, adj_p in zip(survivors.profiles, survivors.adj_p, strict=True):
      ...
  ```

  The `gate=StatCode.X` kwarg (read alternative stat instead of `primary_p`) has no direct replacement; reach `profile.stats[StatCode.X]` and compare to your chosen threshold inline. The `examples/multi_factor_screening.ipynb` per-factor verdict loop is rewritten to demonstrate the BHY path instead.

### Docs

- **Citation accuracy + bibliography role-note layering** (#321, #330, #359). Project-wide sweep across docstrings and `docs/reference/bibliography.md`: every inline paper reference unified to autorefs-linked form (`[Newey-West 1987][newey-west-1987]`), bibliography organised by methodological role rather than chronology, and role-notes audited for attribution accuracy (Andrews vs HH `T^(1/3)`, Shanken EIV framing, factor-zoo / factor-spanning / unit-root / robust-stats / multiple-testing / event-study / cross-section pricing chains). Horizon-shopping multiplicity and per-period forward-return normalisation references added.
- **Per-metric `Examples:` blocks across the public API** (#312, #322). All 44 metric callables and the remaining public-API surface ship runnable `Examples:` blocks under the call-shape-over-fragile-output convention, exercised by `pytest --doctest-modules` in CI (#314).
- **Mkdocs navigation IA cleanup** (#329, #336 series, #341, #343, #347, #350, #351). Top-level nav restructured around Reading flows (User guide / Concepts / Reference / API / Development), new `Reading results` and `Preparing data` pages, `where-factrix-fits` exit pointer + `common_sparse` dispatch arm, glossary + axis-table dedup, routing surfaces demoted to API landing.
- **Docstring style + register conventions** (#313, #331, #332, #357). Google-style section sweep (`Args:` / `Returns:` / `Raises:` / `Warns:` / `Notes:` / `Examples:` / `References:` plural, fixed order) + ruff `D` rule enablement; design-register "verb" prose dropped from user-facing docs (kept as RFC vocabulary); abbreviations expanded on first use per page.
- **Mkdocs UX polish** (#328, #345, #360, #363, #376). Mkdocstrings inventory cross-refs wired for third-party data types; sidebar active item contrast fix scoped to primary nav so the right-side TOC active heading remains legible; internal dense-reindex / event-HHI bin-grid descriptions rephrased from "calendar" to period / time-axis vocabulary (academic methodology names preserved).

---

## v0.12.0 (2026-05-11)

### Changed

- **Docs IA — PyData 5-tab canon + tagline unification + SSOT signposting** (#234). `mkdocs.yml` nav restructures from 6 tabs to the PyData mainstream 5: **Get Started** (landing → where-factrix-fits → install → quickstart, ordered to mirror onboarding `know → buy-in → install → run`), **User Guide** (Concepts / How-to / Examples subsections — replaces flat `Guides` and folds `examples` in), **API Reference** (merges the old separate `Reference` and `API` tabs into Entry points / Results / Tables / Metrics / Other modules), **Development** (contributor docs only), **Release Notes** (CHANGELOG promoted to top-level). The standalone `Home` tab is dropped — `docs/index.md` now doubles as site root **and** Get Started's first entry, matching how every surveyed PyData site (pandas, numpy, scipy, xarray, polars) wires its landing. Material breadcrumbs (`navigation.path`) enabled so deeper paths like `API Reference > Metrics > [per-metric]` retain orientation. Three new canonical pages gain `!!! abstract "Answers"` admonitions (`guides/batch-screening.md`, `guides/panel-timeseries.md`, `guides/slice-analysis.md`), with reverse `!!! tip "Canonical reference"` signposts on the four pages search would otherwise dump readers onto without context (`api/multi-factor.md`, `reference/statistical-methods.md §2`, `reference/ts-mode-conventions.md`, `api/evaluate.md`). `api/evaluate.md` additionally gains a 3-row `profile.mode` (PANEL vs TIMESERIES) crib table — the body's premise of a `mode=` parameter was incorrect (Mode is dispatched from `N`, not user-passed), so the cross-tab semantic split lands on the output field instead. Canonical tagline ("Does this factor possess predictive edge? factrix is the first Polars-native Python toolkit that picks the right statistical test for each factor type. Cross-sectional, event, common factor — each gets the tests that fit its data-generating process.") is now byte-identical across three SSOT entry points: README §Where factrix fits, `docs/index.md`, and `docs/where-factrix-fits.md`. The README L36 subtitle is removed (its core-question phrasing is absorbed into the canonical tagline); the docs/index.md hero is replaced with the same canonical wording; `where-factrix-fits.md` no longer assumes the reader has come from the README (it is now Get Started entry #2 and is reached directly in docs). Factrix-self "framework" is replaced with "toolkit" everywhere it appears in factrix-positioning prose (README / Home / where-factrix-fits / docstring-touching `__init__.py` already on `toolkit` via `pyproject.toml`); the remaining "framework" mentions are about other libraries (linearmodels, multiple-testing framework, SDF framework) and are left alone. Pipeline-stage names re-aligned across README ASCII and `where-factrix-fits.md` mermaid (`strategy construction` / `live trading`), with skfolio / PyPortfolioOpt / riskfolio-lib re-labelled as "Stage 2 strategy construction (portfolio optimisation)" in the ecosystem table. `docs/guides/standalone-metrics.md` is **decided** as User Guide > How-to in full, not split — the page's task-oriented spine (find metric → align input → wire into pipeline → discover dynamically) is tighter than any concept/reference cleavage would yield; tables and code are how-to affordances, not reference content. The orphan `docs/getting-started/index.md` is deleted (its three-line link-list was redundant after Get Started flattened to four direct entries); per the no-redirect policy for the pre-1.0 stage, no migration shim is added. CHANGELOG is exposed under the new `Release Notes` tab via the existing `--8<--` snippet include — no inbound anchored links to CHANGELOG sections exist so slug-stability is moot.
- **Drop `Layer-A` / `Layer-B` stage labels from code and published docs** (#214). Docstrings, doc pages, and test names now describe behaviour functionally (`slice-test verb`, `slice-test Estimator`, `paired-diff slice test`) instead of by planning tier. The contributing guide gains a Terminology subsection so the rewrite does not regress. No public API or behavioural change.
- **Regime analysis guide renamed to slice analysis** (#230). `docs/guides/regime-analysis.md` → `docs/guides/slice-analysis.md`, with intro reframed so regime / universe / sector / ADV-bucket are presented as equivalent applications of the same axis-agnostic surface (`by_slice` + `slice_pairwise_test` / `slice_joint_test`). The old URL is intentionally not redirected — external bookmarks to `guides/regime-analysis/` return 404. Inbound references in `docs/index.md`, `docs/api/metrics/ic.md`, `docs/api/list-metrics.md`, and `docs/api/run-metrics.md` updated to the new path and to drop stale `regime_ic` mentions left behind by #217.
- **Reference nav: metric pages grouped under `Reference > Metrics`** (#231). `Applicability` / `Pipelines` / `Stat keys` move from flat top-level Reference entries to a single sub-section, reducing the flat-Reference page count from 8 to 6 visible entries. `mkdocs.yml` adds `not_in_nav: '**/_generated_*.md'` to silence mkdocs' orphan-page INFO for the four hook-generated tables (`_generated_metric_matrix.md`, `_generated_metric_name_index.md`, `_generated_evaluate_metric_table.md`, `_generated_registry_cells.md`) — all four are already embedded into their host pages via `pymdownx.snippets`, so the orphan listing was a false positive. `reference/metric-applicability.md` gains a back-link to `guides/choosing-metric.md` so the task-oriented decision guide and the lookup table are bidirectionally discoverable.
- **Docs Home and API landing page now show a clickable verb map** (#233). Both `docs/index.md` (Home) and `docs/api/index.md` host a top-of-page mermaid flowchart of the seven shipped verbs (`evaluate` / `run_metrics` / `by_slice` / `slice_pairwise_test` / `slice_joint_test` / `multi_factor.bhy` / `list_metrics`) — coloured by category (compute / decision / view / introspection), with hand-written `click` directives so each node navigates to its API page. The Home page replaces its previous "Where to start" card table with a slightly richer link grid that adds entries for **Where factrix fits** (positioning vs peers) and the full **API reference**, so the visual mental model is the first thing a new visitor sees. A new "Typical patterns" table summarises the canonical pipelines. The 17-node dispatch mermaid in `docs/development/architecture.md` is also collapsed to a single `FactorProcedure` node — the seven concrete procedure-class names that previously crammed the diagram into illegibility on mobile widths are listed in prose below the graph with a link to the cell-keyed mapping in `metric-applicability.md`. Sibling original-design verbs that are not yet shipped (`compare`, `robustness`, `bhy_hierarchical`, `partial_conjunction` per #148) are documented in prose under the verb map rather than drawn, so the diagram never claims an API that does not exist. The proposed mkdocs hook for auto-injecting `click` rows (#232) is shelved as over-engineering for a single-graph consumer.

### Added

- **CI link checker for docs** (#238). New `link-check.yml` workflow runs `lychee-action@v2` against the built `site/` plus `README.md` to catch broken links and dead fragments that `mkdocs build --strict` misses. PR runs fail and block merge on broken links; a daily scheduled run on `main` (06:00 UTC) auto-opens or updates a tracking issue when external URL rot is detected, instead of red-building the main branch. False positives are routed through a top-level `.lycheeignore` file.
- **`factrix.SliceResult`** (#212). Container returned by
  `by_slice` — a `Mapping[str, MetricOutput]` subclass, so every
  existing `dict`-shaped consumer (`for k, v in result.items()`,
  `result["bull"]`, `len(result)`) keeps working unchanged. Adds
  `.to_frame()`, a fixed-schema long-form `pl.DataFrame` renderer
  (`slice`, `name`, `value`, `stat`, `p_value`) for plotting,
  leaderboards, and Notebook rendering, plus a `_repr_html_` that
  delegates to the same frame so `SliceResult` shows as a table in
  Jupyter. `slice_col=` renames the label column. See
  `docs/api/by-slice.md` for the schema rationale (why a fixed
  projection instead of a configurable `cols=` argument).
- **`factrix.slice_pairwise_test` / `factrix.slice_joint_test`** (#176, #215). Cross-slice statistical-test verb pair, hosted in the new `factrix.slicing` subpackage and re-exported at top level (`from factrix import by_slice, slice_pairwise_test, slice_joint_test`). `slice_pairwise_test` reports K(K−1)/2 pairwise Wald contrasts with Holm / Romano-Wolf / Bonferroni adjusted p; `slice_joint_test` reports the single omnibus Wald χ² that all slice means are equal. Default estimator `WaldNWCluster` (joint NW HAC over the per-date K-vector panel) covers analytic inference; `BlockBootstrap` triggers the joint bootstrap path with Romano-Wolf as the default multiple-testing adjustment. Both verbs require the metric's module to declare a `per_date_series` capability — `ic` / `fama_macbeth` / `hit_rate` ship with it; metrics without it raise `TypeError`. See `docs/api/slice-test.md`.
- **`factrix.slicing` subpackage** (#215). Hosts `by_slice` (axis-agnostic dispatcher) plus the verb pair above. The move makes `factrix.metrics` a pure cell-metric registry — every public `*.py` under `factrix/metrics/` is a per-(scope, signal) metric, and `run_metrics` auto-discovery is now structural rather than maintained via a denylist.
- **`factrix.metrics._metric_capabilities`** (#176). Resolver helpers (`resolve_per_date_series`, `resolve_min_assets_per_group`) plus a `PerDateSeries` Protocol and a `per_date_series_rename` factory. Centralises capability lookup so inference verbs reuse one resolver instead of grovelling `sys.modules` directly.

### Removed

- **`factrix.multi_factor.bhy(threshold=)`** kwarg (breaking, #217). Deprecated since v0.4.0 in favour of `q=`. The deprecation collector path (`**deprecated` kwarg, `_apply_deprecated_kwargs`, runtime `DeprecationWarning`) is gone; passing `threshold=` now raises `TypeError` via the standard signature check.

  ```python
  # before (v0.4.0 – v0.11.x, deprecated)
  bhy(profiles, threshold=0.05)

  # after (v0.12.0)
  bhy(profiles, q=0.05)
  ```
- **`factrix.metrics.by_regime`** (breaking, #217). Deprecated since v0.10.0 in favour of the axis-agnostic `factrix.by_slice`. The `factrix/metrics/regime.py` module is removed entirely (including the private `_slice_by_regime` helper, which was only used by `by_regime` and the already-deleted `regime_ic`).

  ```python
  # before (v0.10.0 – v0.11.x, deprecated)
  from factrix.metrics import by_regime
  per_regime = by_regime(ic, ic_df, regime_labels=labels)

  # after (v0.12.0) — labels-join semantics are now explicit;
  # project the labels frame down to (date, regime) so extra columns
  # do not bleed into the metric input
  from factrix import by_slice
  per_regime = by_slice(
      ic,
      ic_df.join(labels.select("date", "regime"), on="date", how="inner"),
      label="regime",
  )
  ```

  The labels frame must carry a column literally named `regime` to line up with `label="regime"`. The previous `by_regime` raised a friendly `ValueError("no rows survived the regime-label join — likely a date-range or dtype mismatch")` on an empty inner-join; the recipe no longer wraps that guard, so callers wanting it should add an `assert merged.height > 0, "..."` after the join.

  The time-bisection fallback (`regime_labels=None` → `first_half` / `second_half`) has no `by_slice` equivalent — that path was a structural-break sanity check, not a regime test, and the removal is the right moment to make that explicit. The old behaviour split by **row index** (sorted by date, first half / second half), not by median date; reproduce it explicitly:

  ```python
  import polars as pl

  df = df.sort("date").with_row_index("_i").with_columns(
      pl.when(pl.col("_i") < pl.len() // 2)
        .then(pl.lit("first_half"))
        .otherwise(pl.lit("second_half"))
        .alias("regime"),
  ).drop("_i")
  by_slice(ic, df, label="regime")
  ```
- **`factrix.metrics.regime_ic`** (breaking, #176, #215). Replaced by `factrix.slice_pairwise_test`. The #176 deprecation initially announced a one-minor frozen-shape window; #215 supersedes that plan and removes the function in the same minor, because v0.12 is a breaking train and there is no external user base to migrate. `regime_labels` is expected to carry a column literally named `regime` so the resulting `label="regime"` argument lines up:

  ```python
  # before (v0.11.0)
  from factrix.metrics import regime_ic
  result = regime_ic(ic_df, regime_labels=regime_df)   # regime_df has columns date, regime

  # after (v0.12.0)
  from factrix import slice_pairwise_test
  from factrix.metrics import ic
  pairs = slice_pairwise_test(
      ic,
      ic_df.join(regime_df, on="date", how="inner"),
      label="regime",
  )
  ```

  **Statistical-decision change** (read before reproducing v0.11 numbers): `regime_ic` reported `min |t|` across regimes with **BHY**-adjusted p across regimes — a regime-as-family **FDR** rule answering "does *any* regime show signal?". `slice_pairwise_test` instead returns K(K−1)/2 **pairwise** contrasts with **Holm**-adjusted p by default — a pair-as-family **FWER** rule answering "does *any pair* of regimes differ?". Different null hypothesis, different error rate, and the directions of strictness are not monotone — cross-version paper reproductions need an explicit method footnote. Switch to `estimator=BlockBootstrap()` for the joint-bootstrap path with Romano-Wolf adjustment if your slices share dates.
- **`factrix.metrics.by_slice` / `factrix.metrics.slice_pairwise_test` / `factrix.metrics.slice_joint_test` deep imports** (breaking, #215). Import paths moved to `factrix.slicing`; top-level re-exports keep the calling shape stable.

  ```python
  # before (v0.11.0)
  from factrix.metrics import by_slice, slice_pairwise_test, slice_joint_test

  # after (v0.12.0) — preferred
  from factrix import by_slice, slice_pairwise_test, slice_joint_test
  # or equivalently
  from factrix.slicing import by_slice, slice_pairwise_test, slice_joint_test
  ```
- **`factrix._validators.validate_n_assets`** (breaking, #218). Dead since pre-v1 — function was never wired (no internal caller, no test coverage, no doc reference). Validation responsibilities live in `AnalysisConfig` construction and metric-side short-circuit guards. The host module `factrix/_validators.py` is removed entirely (no other public symbols).
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

### BREAKING CHANGE

- ``StatCode.P`` (value ``"p"``) renamed to
``StatCode.P_NW`` (value ``"p_nw"``). Replace every ``StatCode.P``
lookup with ``StatCode.P_NW``; ``profile.diagnose()`` JSON keys move
from ``"p"`` to ``"p_nw"``.
- bhy() and the shared _resolve_family layer no longer
accept p_stat=StatCode. The kwarg becomes estimator=Estimator | None
and dispatch routes through Estimator.applicable_to / .emits_for to
look up the relevant entry in profile.stats. The gate=StatCode v0.4
deprecation is removed alongside its successor; threshold= → q=
remains live.
- bhy() return type changed from list[FactorProfile]
to multi_factor.Survivors. Migration: replace 'survivors' with
'survivors.profiles' for downstream list/iteration use; new adj_q /
q / expand_over / n_total fields available for richer downstream
diagnostics.

### Feat

- **api**: standalone-metric runner for a cell (#147)
- **stats**: emit T_HH alongside P_HH + reserve J_GMM (#184)
- **stats**: IC + FM PANEL emit HH p-value (#184)
- **stats**: HansenHodrick Estimator instance + registry (#184)
- **stats**: HH primitives + clamp warning code (#184)
- **profile**: metadata channel for hyperparameter records (#188)
- **stats**: add list_estimators introspection (#170)
- **stats**: dispatch family verbs via Estimator (#170)
- **stats**: add NeweyWest reference Estimator (#170)
- **stats**: add Estimator protocol (#170)
- **multi_factor**: Survivors container for bhy (#171) (#182)
- **multi_factor**: bhy on _resolve_family + explicit families (#161)
- **family**: _resolve_family + FamilyEntry (#161)
- **profile**: identity / context split (#160) (#172)
- **errors**: user-facing error UX contract (#165) (#169)

### Refactor

- **metrics**: deprecate multi_horizon_* (#186) (#199)
- **stats**: rename StatCode.P to P_NW for symmetry with T_NW (#192)
- **stats**: tighten T_HH metadata + defer J_GMM to #191 (#184)
- **stats**: tighten review feedback (#187)
- **stats**: flatten StatCode naming (#187)
- **stats**: tighten review feedback (#170)

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

  # before (v0.4.0 – v0.11.x, deprecated)
  bhy(profiles, threshold=0.05)

  # after (v0.12.0)
  bhy(profiles, q=0.05)
  ```
- **`factrix.metrics.by_regime`** (breaking, #217). Deprecated since v0.10.0 in favour of the axis-agnostic `factrix.by_slice`. The `factrix/metrics/regime.py` module is removed entirely (including the private `_slice_by_regime` helper, which was only used by `by_regime` and the already-deleted `regime_ic`).

  ```python
  # before (v0.10.0 – v0.11.x, deprecated)
  from factrix.metrics import by_regime
  per_regime = by_regime(ic, ic_df, regime_labels=labels)

  # after (v0.12.0) — labels-join semantics are now explicit;
  # project the labels frame down to (date, regime) so extra columns
  # do not bleed into the metric input
  from factrix import by_slice
  per_regime = by_slice(
      ic,
      ic_df.join(labels.select("date", "regime"), on="date", how="inner"),
      label="regime",
  )
  ```

  The labels frame must carry a column literally named `regime` to line up with `label="regime"`. The previous `by_regime` raised a friendly `ValueError("no rows survived the regime-label join — likely a date-range or dtype mismatch")` on an empty inner-join; the recipe no longer wraps that guard, so callers wanting it should add an `assert merged.height > 0, "..."` after the join.

  The time-bisection fallback (`regime_labels=None` → `first_half` / `second_half`) has no `by_slice` equivalent — that path was a structural-break sanity check, not a regime test, and the removal is the right moment to make that explicit. The old behaviour split by **row index** (sorted by date, first half / second half), not by median date; reproduce it explicitly:

  ```python
  import polars as pl

  df = df.sort("date").with_row_index("_i").with_columns(
      pl.when(pl.col("_i") < pl.len() // 2)
        .then(pl.lit("first_half"))
        .otherwise(pl.lit("second_half"))
        .alias("regime"),
  ).drop("_i")
  by_slice(ic, df, label="regime")
  ```
- **`factrix.metrics.regime_ic`** (breaking, #176, #215). Replaced by `factrix.slice_pairwise_test`. The #176 deprecation initially announced a one-minor frozen-shape window; #215 supersedes that plan and removes the function in the same minor, because v0.12 is a breaking train and there is no external user base to migrate. `regime_labels` is expected to carry a column literally named `regime` so the resulting `label="regime"` argument lines up:

  ```python
  # before (v0.11.0)
  from factrix.metrics import regime_ic
  result = regime_ic(ic_df, regime_labels=regime_df)   # regime_df has columns date, regime

  # after (v0.12.0)
  from factrix import slice_pairwise_test
  from factrix.metrics import ic
  pairs = slice_pairwise_test(
      ic,
      ic_df.join(regime_df, on="date", how="inner"),
      label="regime",
  )
  ```

  **Statistical-decision change** (read before reproducing v0.11 numbers): `regime_ic` reported `min |t|` across regimes with **BHY**-adjusted p across regimes — a regime-as-family **FDR** rule answering "does *any* regime show signal?". `slice_pairwise_test` instead returns K(K−1)/2 **pairwise** contrasts with **Holm**-adjusted p by default — a pair-as-family **FWER** rule answering "does *any pair* of regimes differ?". Different null hypothesis, different error rate, and the directions of strictness are not monotone — cross-version paper reproductions need an explicit method footnote. Switch to `estimator=BlockBootstrap()` for the joint-bootstrap path with Romano-Wolf adjustment if your slices share dates.
- **`factrix.metrics.by_slice` / `factrix.metrics.slice_pairwise_test` / `factrix.metrics.slice_joint_test` deep imports** (breaking, #215). Import paths moved to `factrix.slicing`; top-level re-exports keep the calling shape stable.

  ```python
  # before (v0.11.0)
  from factrix.metrics import by_slice, slice_pairwise_test, slice_joint_test

  # after (v0.12.0) — preferred
  from factrix import by_slice, slice_pairwise_test, slice_joint_test
  # or equivalently
  from factrix.slicing import by_slice, slice_pairwise_test, slice_joint_test
  ```
- **`factrix._validators.validate_n_assets`** (breaking, #218). Dead since pre-v1 — function was never wired (no internal caller, no test coverage, no doc reference). Validation responsibilities live in `AnalysisConfig` construction and metric-side short-circuit guards. The host module `factrix/_validators.py` is removed entirely (no other public symbols).
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
