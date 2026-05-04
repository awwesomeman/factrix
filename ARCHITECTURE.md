# factrix Architecture

Current-state snapshot — describes the v0.5 library as it stands.

---

## Positioning

**factrix is a Factor Signal Validator, not a backtest engine.**

The library produces a single `primary_p` per factor cell from a NW HAC-corrected
canonical procedure (IC / FM-λ / CAAR / TS-β). Realistic execution simulation,
tradability proxies, and portfolio construction are out of scope — feed
screened factors into Zipline / Backtrader / `vectorbt` downstream.

---

## Public API surface

Three entry points, all in `factrix.__init__`:

| Symbol | Purpose |
|--------|---------|
| `fl.AnalysisConfig` | Three-axis frozen dataclass; construct via 4 factory methods |
| `fl.evaluate(panel, config)` | Dispatch to the registered procedure → `FactorProfile` |
| `fl.multi_factor.bhy(profiles, *, threshold=0.05)` | Benjamini-Yekutieli FDR correction across a profile batch |

Plus introspection / error / enum re-exports:

- `fl.FactorScope`, `fl.Signal`, `fl.Metric`, `fl.Mode` — three user-facing axes + the evaluate-time-derived fourth
- `fl.WarningCode`, `fl.InfoCode`, `fl.StatCode`, `fl.Verdict` — structured codes carried on `FactorProfile`
- `fl.FactorProfile` — single unified result type
- `fl.describe_analysis_modes(format="text"|"json")` — registry-reflected cell catalogue
- `fl.suggest_config(panel)` — heuristic factory call from a raw panel
- `fl.ConfigError`, `fl.IncompatibleAxisError`, `fl.ModeAxisError`, `fl.InsufficientSampleError` — exception hierarchy

`__version__ = "0.5.0"`.

---

## Three orthogonal axes + Mode

The user-facing axis triple is `(FactorScope, Signal, Metric)`. A fourth
axis `Mode` is **derived at evaluate-time** from `panel["asset_id"].n_unique()`:

| Axis        | Values                                      | User-facing? |
|-------------|---------------------------------------------|--------------|
| `FactorScope` | `INDIVIDUAL` / `COMMON`                   | yes          |
| `Signal`    | `CONTINUOUS` / `SPARSE`                     | yes          |
| `Metric`    | `IC` / `FM` / `None`                        | yes (only `(INDIVIDUAL, CONTINUOUS)` accepts a non-None metric) |
| `Mode`      | `PANEL` (N≥2) / `TIMESERIES` (N=1)          | no — derived  |

Five legal `(scope, signal, metric)` triples × two modes give seven legal
`(scope, signal, metric, mode)` cells (TIMESERIES narrows to three triples; the
remaining tuples are routed via the `_SCOPE_COLLAPSED` sentinel — see §5.4.1
of the refactor plan).

---

## Registry SSOT dispatch

`factrix/_registry.py` holds **the** source of truth:

- `_DispatchKey(scope, signal, metric, mode)` — the cell coordinate
- `_RegistryEntry(key, procedure, canonical_use_case, references)` — procedure + docs metadata
- `_DISPATCH_REGISTRY: dict[_DispatchKey, _RegistryEntry]`
- `register(key, procedure, *, use_case, refs)` — append-only; duplicate keys raise
- `matches_user_axis(scope, signal, metric)` — reverse query for `AnalysisConfig` validation
- `_SCOPE_COLLAPSED: _ScopeCollapsedSentinel` — internal routing token for `(*, SPARSE, N=1)`; not exposed as a `FactorScope` enum value to keep the user-facing axis narrow

Bootstrap order: `_registry` defines `register` / `_DispatchKey`, then imports
`_procedures` at the bottom of the module so all 7 cells `register(...)` at
import time. Every introspection/validation path reverse-queries this dict —
no parallel rule table.

Adding a cell touches one `register(...)` call.

---

## FactorProcedure protocol

`factrix/_procedures.py` defines the seven procedure classes. Each implements:

```python
class FactorProcedure(Protocol):
    INPUT_SCHEMA: ClassVar[InputSchema]
    def compute(self, raw: pl.DataFrame, config: AnalysisConfig) -> FactorProfile: ...
```

`InputSchema` lists `required_columns` — currently `("date", "asset_id", "factor", "forward_return")` for all 7 cells.

The seven cells (cell tuple ↔ procedure class). For the user-facing factory
mapping and per-cell canonical statistic / references, see
[README §5 種支援的分析情境](README.md#5-種支援的分析情境--對應檢定方法) — that
table is the SSOT for what each procedure computes.

| `(scope, signal, metric, mode)`                         | Procedure class                                  |
|---------------------------------------------------------|--------------------------------------------------|
| `(INDIVIDUAL, CONTINUOUS, IC, PANEL)`                    | `_ICPanelProcedure`                              |
| `(INDIVIDUAL, CONTINUOUS, FM, PANEL)`                    | `_FMPanelProcedure`                              |
| `(INDIVIDUAL, SPARSE, None, PANEL)`                      | `_CAARPanelProcedure`                            |
| `(COMMON, CONTINUOUS, None, PANEL)`                      | `_CommonContPanelProcedure`                      |
| `(COMMON, SPARSE, None, PANEL)`                          | `_CommonSparsePanelProcedure`                    |
| `(COMMON, CONTINUOUS, None, TIMESERIES)`                 | `_TSBetaContTimeseriesProcedure`                 |
| `(_SCOPE_COLLAPSED, SPARSE, None, TIMESERIES)`           | `_TSDummySparseTimeseriesProcedure`              |

The two timeseries cells share the NW HAC + auto-Bartlett-with-Hansen-Hodrick-floor
lag rule from `factrix/_stats/constants.py`.

---

## FactorProfile dataclass contract

`factrix/_profile.py`:

```python
@dataclass(frozen=True, slots=True)
class FactorProfile:
    config: AnalysisConfig
    mode: Mode
    primary_p: float
    n_obs: int          # cell-canonical effective N (T / events / assets)
    n_assets: int       # raw panel cross-section width (always available)
    warnings: frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode] = frozenset()
    stats: Mapping[StatCode, float] = field(default_factory=dict)
```

`n_obs` semantics vary by cell — T for IC/FM/TS-β, event count for
CAAR, asset count for COMMON×* PANEL. `n_assets` is always
`raw["asset_id"].n_unique()`; reading both side by side disambiguates
"small effective sample" between short series vs thin cross-section.

- `verdict(*, threshold=0.05, gate=None) -> Verdict` — `gate=None` uses `primary_p`; supplying a `StatCode` swaps the gate (raises `KeyError` if not populated)
- `diagnose() -> dict[str, Any]` — flatten `mode / n_obs / primary_p / warnings / info_notes / stats` for human or AI agent triage

Single dataclass, no per-cell subclass proliferation. Cell-specific scalars live
in `stats: Mapping[StatCode, float]` keyed by enum, not by string.

---

## PANEL / TIMESERIES equivalence

Both modes produce real `primary_p` values — neither is degraded.

`(INDIVIDUAL, CONTINUOUS, *) × N=1` is mathematically undefined (no
cross-sectional dispersion → IC and per-date OLS undefined). `_evaluate`
raises `ModeAxisError` with `suggested_fix=AnalysisConfig.common_continuous(...)`
drawn from `_FALLBACK_MAP` in `factrix/_analysis_config.py`. Explicit
user-correctable, never silent rewrite.

`(*, SPARSE, *) × N=1` is well-defined but the `INDIVIDUAL` / `COMMON`
distinction collapses (one asset → no scope axis). Both user-facing factory
calls route to the same `_TSDummySparseTimeseriesProcedure` via the
`_SCOPE_COLLAPSED` sentinel, with `InfoCode.SCOPE_AXIS_COLLAPSED` attached
to the resulting profile so the routing is auditable.

---

## Sample guards

`factrix/_stats/constants.py`:

- `MIN_PERIODS_HARD = 20` — `n_periods < MIN_PERIODS_HARD` raises `InsufficientSampleError`
- `MIN_PERIODS_RELIABLE = 30` — `n_periods < MIN_PERIODS_RELIABLE` adds `WarningCode.UNRELIABLE_SE_SHORT_PERIODS`
- `auto_bartlett(T) = max(1, int(4 * (T/100)**(2/9)))` — Newey-West (1994) auto lag rule
- Hansen-Hodrick (1980) overlap floor: `max(auto_bartlett(T), forward_periods - 1)` — ensures NW lag covers MA(h-1) structure from overlapping forward returns

`factrix/_types.py` keeps the older per-metric thresholds (`MIN_ASSETS_PER_DATE_IC = 10`,
`MIN_EVENTS = 10`, etc.) used internally by the metric primitives that
procedures wrap.

### Cross-sectional guards (`n_assets`)

`factrix/_stats/constants.py`:

- `MIN_ASSETS = 10` — `n_assets < MIN_ASSETS` emits `WarningCode.SMALL_CROSS_SECTION_N`
  from the `common_continuous` PANEL procedure and from `suggest_config`.
  df = `n_assets` − 1 → t_crit at `n_assets` = 3 ≈ 4.30 (+119% vs asymptotic 1.96),
  at `n_assets` = 5 ≈ 2.78 (+42%). Test still runs; warning surfaces the
  inflation so caller can collect more cross-section before trusting reject
  decisions.
- `MIN_ASSETS_RELIABLE = 30` — `MIN_ASSETS ≤ n_assets < MIN_ASSETS_RELIABLE`
  emits `WarningCode.BORDERLINE_CROSS_SECTION_N`. df → t_crit at
  `n_assets` = 10 ≈ 2.26 (+15%), at `n_assets` = 20 ≈ 2.09 (+7%). The gross
  failure tier is cleared, but residual t-stat inflation matters for borderline
  p-values (e.g. p ≈ 0.04 should be read as "borderline at this `n_assets`",
  not "rejected").

Symmetric with the `n_periods` two-tier (`MIN_PERIODS_HARD = 20` raises
`InsufficientSampleError`; `MIN_PERIODS_RELIABLE = 30` emits
`UNRELIABLE_SE_SHORT_PERIODS`). The `n_assets` axis never raises because
the cross-asset t-test on E[β] is mathematically well-defined for
`n_assets ≥ 2` — only its statistical power degrades. Constant naming
deliberately drops the `_HARD` suffix on `MIN_ASSETS` to avoid implying a
raise; `_RELIABLE` mirrors the `n_periods` semantics.

`MIN_ASSETS_PER_DATE_IC = 10` (in `factrix/_types.py`) drops dates with
fewer than 10 assets from `compute_ic`. At `n_assets` < 10 the IC procedure
short-circuits to NaN because every date is dropped. `compute_fm_betas`
carries an inline `if len(y) < 3: continue` guard but no per-date min
above 3.

---

## Procedure pipelines

The 7 registered procedures differ in **aggregation order** — which axis is
collapsed first determines small-sample failure modes and the N=1 collapse
behavior. The user-facing factory chosen determines which pipeline runs.

### Terminology — aggregation regime

Two regimes, each with concrete sub-forms. Pipeline pseudocode tags each
step with `(cross-section step)` or `(time-series step)` inline:

- **cross-section step** — aggregate over assets at a fixed date
  - `per-date` — applied to every date (continuous panel)
  - `per-event-date` — restricted to dates where `factor != 0` (sparse cells)
- **time-series step** — aggregate over the time axis
  - `per-asset` — fix one asset, aggregate its full date sequence
    (`filter(asset_id == X)`)
  - on a previously-built time-indexed series — e.g. NW HAC t-test on
    `IC[t]` or `β[i]` after the upstream step has produced the series

Unqualified `per-event` is **not** used — always written as `per-event-date`
to keep the regime unambiguous.

### `individual_continuous(IC)` — cross-section first

```
per-date Spearman across n_assets         (cross-section step)
                                       →  n_periods-length IC time series
                                       →  NW HAC t-test on mean(IC)        (time-series step)
```

Failure modes:

- `n_assets` < 10 → `MIN_ASSETS_PER_DATE_IC` drops every date → output is NaN.
- `n_periods < MIN_PERIODS_HARD` → `InsufficientSampleError`.
- `MIN_PERIODS_HARD ≤ n_periods < MIN_PERIODS_RELIABLE` → `UNRELIABLE_SE_SHORT_PERIODS`.

### `individual_continuous(FM)` — cross-section first

```
per-date OLS R = α + β·Signal across n_assets   (cross-section step)
                                              →  n_periods-length λ time series
                                              →  NW HAC t-test on mean(λ)   (time-series step)
```

Failure modes:

- per-date `n_assets` < 3 → date dropped (`if len(y) < 3: continue`).
- per-date `n_assets` small but ≥ 3 → df = `n_assets` − 2 minimal, β unstable.
- `n_periods < MIN_FM_PERIODS = 20` → short-circuit to insufficient.

### `individual_sparse` (CAAR PANEL) — cross-section first (events)

```
per-event-date mean of signed_car = return × factor      (cross-section step)
                                                       →  event-date-indexed CAAR
reindex to dense calendar, zero-fill non-event dates   →  n_periods-length CAAR series
                                                       →  NW HAC t-test on mean(CAAR)   (time-series step)
```

The CAAR series is **calendar-indexed**: `compute_caar` produces an
event-date-indexed primitive (filter `factor != 0`), which the procedure
then reindexes against the full date set with zero-fill. This is the
calendar-time portfolio approach (Jaffe 1974, Mandelker 1974; Fama 1998
§2) — restores the lag rule's "consecutive observations are 1 calendar
period apart" assumption that an event-only series would otherwise
break. With it, sparse events let zero-padding zero out spurious
autocovariance terms and clustered events get the real MA(h-1) overlap
weighted correctly. Pipeline parity with IC / FM / common-sparse PANEL.

Magnitude is preserved as a weight in `signed_car` (no `.sign()` coercion
at this layer — `compute_caar`'s docstring carries the input-form
behaviour table). User-facing `CAAR_MEAN` reports the per-event-date
mean (the average effect on event days); `n_obs` and `NW_LAGS_USED`
reflect the dense series.

Failure modes:

- `n_events < MIN_EVENTS` → event series too short → primary_p reverts to insufficient.
- `n_periods < MIN_PERIODS_HARD` (overall panel length) → `InsufficientSampleError`.
- `MIN_PERIODS_HARD ≤ n_periods < MIN_PERIODS_RELIABLE` → `UNRELIABLE_SE_SHORT_PERIODS`.

### `common_continuous` — time-series first

```
per-asset OLS R_i = α_i + β_i·F over all n_periods dates   (time-series step)
                                                         →  n_assets-length β vector
                                                         →  cross-asset t-test on E[β]   (cross-section step)
```

Failure modes:

- per-asset `n_periods < MIN_TS_OBS = 20` → asset dropped.
- `n_assets < MIN_ASSETS = 10` → `WarningCode.SMALL_CROSS_SECTION_N` (still runs).
- `MIN_ASSETS ≤ n_assets < MIN_ASSETS_RELIABLE = 30` → `WarningCode.BORDERLINE_CROSS_SECTION_N`.
- `n_assets = 1` → degenerate cross-asset test → mode auto-routed to
  TIMESERIES single-series β test (null: β = 0, **not** E[β] = 0). The
  `StatCode.TS_BETA` identifier is shared across the two modes, so the
  same field on `FactorProfile` carries different statistical meaning
  depending on `profile.mode`; see §PANEL/TIMESERIES equivalence.

### `common_sparse` (PANEL) — time-series first

```
per-asset OLS R_i = α_i + β_i·D over all n_periods dates   (time-series step)
                                                         →  n_assets-length β vector
                                                         →  cross-asset t-test on E[β]   (cross-section step)
```

Same shape as `common_continuous`; the broadcast `D ∈ {-1, 0, +1}` dummy
replaces the continuous regressor. Factor magnitudes are **preserved** in
the OLS (no `.sign()` coercion at this layer — distinct from the
`individual_sparse` PANEL pipeline). ADF persistence diagnostic is skipped
per I6 (sparse regressors are not unit-root candidates).

Failure modes:

- per-asset `n_periods < MIN_TS_OBS = 20` → asset dropped.
- `n_assets` two-tier guard same as `common_continuous` (`SMALL_CROSS_SECTION_N` /
  `BORDERLINE_CROSS_SECTION_N`).
- The procedure does not currently impose a `n_events` floor on the
  broadcast dummy — very-few-event factors can produce point estimates
  driven by a single observation.
- Cross-asset SE assumes asset-level independence (plan §4.3 spec); under
  contemporaneous return correlation the standard t over-states
  significance — Petersen (2009) clustered SE deferred per plan §11.

### `common_continuous` (TIMESERIES, N=1) — time-series only

```
single-asset OLS y_t = α + β·F_t + ε   (time-series step)
                                     →  NW HAC t-test on β
                                     +  ADF persistence diagnostic on F
```

The N=1 collapse of `common_continuous`. Null is `β = 0` for the single
series, not `E[β] = 0` across assets — semantically distinct from the
PANEL form.

Failure modes:

- `n_periods < MIN_PERIODS_HARD` → `InsufficientSampleError`.
- `MIN_PERIODS_HARD ≤ n_periods < MIN_PERIODS_RELIABLE` → `UNRELIABLE_SE_SHORT_PERIODS`.
- ADF p > 0.10 → `WarningCode.PERSISTENT_REGRESSOR`.

### `(*, SPARSE, *) × N=1` (TS dummy) — time-series only

```
single-asset OLS y_t = α + β·D_t + ε on calendar-dense series   (time-series step)
                                                              →  NW HAC t-test on β
                                                              +  Ljung-Box on residual
                                                              +  event_temporal_hhi
                                                              +  event-window-overlap check
```

Reached from both `individual_sparse` and `common_sparse` at N=1 via the
`_SCOPE_COLLAPSED` sentinel — at N=1 the two scopes are statistically
equivalent (plan §5.4.1). The series is the **full calendar grid** with
zero-padding on non-event dates (distinct from the PANEL CAAR pipeline,
which works on the event-date-only series). Factor magnitudes are
preserved (no `.sign()` coercion at this layer).

Failure modes:

- `n_periods < MIN_PERIODS_HARD` → `InsufficientSampleError`.
- `MIN_PERIODS_HARD ≤ n_periods < MIN_PERIODS_RELIABLE` → `UNRELIABLE_SE_SHORT_PERIODS`.
- Ljung-Box p < 0.05 on residuals → `WarningCode.SERIAL_CORRELATION_DETECTED`.
- Consecutive event gap < 2·`forward_periods` → `WarningCode.EVENT_WINDOW_OVERLAP`.

---

## BHY family partitioning

`factrix/_multi_factor.py::bhy(profiles, *, threshold=0.05, gate=None)`:

1. partitions `profiles` by `_family_key(profile) -> _FamilyKey(dispatch, forward_periods)`:
   - `dispatch: _DispatchKey` derived from `(scope, signal, metric, mode)` with the
     `_SCOPE_COLLAPSED` collapse rule applied so PANEL and TIMESERIES sparse profiles
     aggregate consistently
   - `forward_periods` from `profile.config.forward_periods` — split into separate families
     because each horizon carries its own null distribution and effective sample size;
     pooling horizons dilutes the step-up threshold `q × k / N` and silently inflates FDR
2. within each family, runs Benjamini-Yekutieli step-up correction at the
   given FDR threshold and returns the survivors.

`_FamilyKey` is kept distinct from `_DispatchKey` (the registry SSOT must remain
horizon-agnostic — one procedure per cell). User does **not** pass a group key —
same-test-family is enforced mechanically.

Cross-family aggregation (e.g. horizon-shopping correction) is the user's
responsibility — see README §批次評估與 BHY for the FWER-then-BHY recipe.

---

## Registry procedure vs standalone metric

Two-tier metric organisation. Choosing the right tier when adding a new metric:

| Tier | Lives in | Count today | Definition | Surfaces |
|------|----------|-------------|------------|----------|
| **Registry procedure** | `factrix/_procedures.py` (`register(...)` at module bottom) | exactly 7 (one per legal cell) | The **canonical PASS/FAIL test** for one `(scope, signal, metric, mode)` cell | `evaluate()` dispatch, `FactorProfile.verdict()`, BHY family key |
| **Standalone metric** | `factrix/metrics/*.py` | ~19 modules | **Diagnostic / second-look / multi-statistic** decomposition. User imports and calls directly. | `from factrix.metrics import X` returning `MetricOutput` |

### When to register

Add a registry procedure **only** when introducing a new legal cell on the axis
(`FactorScope × Signal × Metric × Mode`). The 7-cell invariant (`_registry.py::_EXPECTED_REGISTRY_SIZE`)
is a load-bearing assert — adding to the registry without adding a new cell would
mean two canonical procedures compete for the same dispatch, breaking the SSOT
contract.

### When to add a standalone metric

Everything else. Specifically:

- **Same cell already has a canonical procedure** but you want to surface a different angle
  (non-linearity, asymmetry, decomposition, regime split). Example precedent:
  `event_quality.py` (hit_rate / profit_factor / event_skewness / signal_density) all
  supplement the registered CAAR procedure for `(INDIVIDUAL, SPARSE, None, PANEL)`.
- **Descriptive diagnostic without a formal H₀** (concentration HHI, tradability, OOS decay).
- **Multi-factor relationship** outside the single-factor verdict frame (`spanning.py`).

### Standalone metric contract

- Take `pl.DataFrame` with the cell's standard schema (`date, asset_id, factor, forward_return`)
  plus any optional columns
- Return `MetricOutput` (`factrix/_types.py`) — `name`, `value`, optional `stat`, `significance`,
  and a `metadata` dict for cell-specific scalars
- Use `_short_circuit_output(...)` for sample-floor failures rather than raising
- Reuse `_stats/` primitives (`_p_value_from_t`, `_calc_t_stat`, NW HAC helpers) so the
  statistical treatment matches the registered procedures — most notably **NW HAC SE
  for any inference on overlapping forward returns**, never iid Welch / OLS SE

A standalone metric never enters BHY automatically (no `FactorProfile`, no canonical
`primary_p`); the user is responsible for collecting comparable p-values into a family
themselves if FDR control is needed across a batch of standalone runs.

---

## Module layout

```
factrix/
├── __init__.py              # public surface
├── _axis.py                 # FactorScope / Signal / Metric / Mode StrEnums
├── _codes.py                # WarningCode / InfoCode / StatCode / Verdict StrEnums
├── _errors.py               # FactrixError → ConfigError → {IncompatibleAxisError, ModeAxisError, InsufficientSampleError}
├── _analysis_config.py      # AnalysisConfig + 4 factories + _FALLBACK_MAP
├── _registry.py             # _DispatchKey, _RegistryEntry, _SCOPE_COLLAPSED, register()
├── _procedures.py           # 7 FactorProcedure classes; bootstrap-registered at import
├── _profile.py              # FactorProfile dataclass + verdict / diagnose
├── _evaluate.py             # _derive_mode + _evaluate dispatch wrapper
├── _describe.py             # describe_analysis_modes + suggest_config + SuggestConfigResult
├── _multi_factor.py         # bhy with family partitioning
├── multi_factor.py          # public namespace (re-exports bhy)
├── _stats/
│   ├── __init__.py          # _ols_nw_slope_t, _ljung_box_p, _adf, _newey_west_t_test, _resolve_nw_lags
│   └── constants.py         # MIN_PERIODS_HARD / MIN_PERIODS_RELIABLE / auto_bartlett
├── _types.py                # MetricOutput, EPSILON, DDOF, MIN_*_PERIODS
├── metrics/                 # primitives: ic, fama_macbeth, ts_beta, caar, ...
└── datasets.py              # synthetic CS / event panels
```

---

## Invariants

Hard constraints — violating these breaks the API contract:

1. `AnalysisConfig` is `frozen=True, slots=True`; every construction path goes through `__post_init__ → _validate_axis_compat` (factory, direct, `from_dict` all hit the same gate).
2. `FactorProfile` is `frozen=True, slots=True`. One unified type — no per-cell subclass.
3. The registry is the SSOT for "which cells exist". `_validate_axis_compat`, `describe_analysis_modes`, `suggest_config`, BHY family partitioning all reverse-query it; no parallel rule table.
4. `_SCOPE_COLLAPSED` is an internal sentinel. It never appears in a user-facing `AnalysisConfig` — `evaluate()` rewrites the routed scope at dispatch time and reports the collapse via `InfoCode.SCOPE_AXIS_COLLAPSED`.
5. `FactorProfile.primary_p` is a real probability for every legal cell × mode. TIMESERIES never returns a degenerate `primary_p = 1.0`.
6. `verdict()` reads `primary_p` (or a user-supplied `StatCode` gate); `warnings` and `info_notes` never auto-rebind it.
7. BHY family key = `_FamilyKey(_DispatchKey, forward_periods)`, derived mechanically from the profile, not user-supplied. Horizons split into separate families to preserve nominal FDR control.
8. `register(...)` is append-only at import time. Duplicate keys raise `ValueError`.
9. NW HAC lag selection in panel-aggregation cells uses `max(auto_bartlett(T), forward_periods - 1)` — the Hansen-Hodrick floor must not be skipped under overlapping forward returns.
10. `T < MIN_PERIODS_HARD` raises `InsufficientSampleError`; procedures never silently produce a result on under-sample data.

---

## Testing

`tests/` covers the v0.5 surface only — v0.4 tests were removed in the §8.2
deletion sweep. Fixtures are fully synthetic (`tests/conftest.py` +
`factrix.datasets`); no test reads real market data from disk.

Run: `uv run pytest`
