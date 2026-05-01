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
`(scope, signal, metric, mode)` cells (Mode B narrows to three triples; the
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

The seven cells:

| `(scope, signal, metric, mode)`                         | Procedure class                                  | Canonical statistic             |
|---------------------------------------------------------|--------------------------------------------------|---------------------------------|
| `(INDIVIDUAL, CONTINUOUS, IC, PANEL)`                    | `_ICPanelProcedure`                              | NW HAC t on `E[Spearman ρ_t]`   |
| `(INDIVIDUAL, CONTINUOUS, FM, PANEL)`                    | `_FMPanelProcedure`                              | NW HAC t on `E[λ_t]`            |
| `(INDIVIDUAL, SPARSE, None, PANEL)`                      | `_CAARPanelProcedure`                            | Cross-event t on CAAR           |
| `(COMMON, CONTINUOUS, None, PANEL)`                      | `_CommonContPanelProcedure`                      | Cross-asset t on `E[β_i]`       |
| `(COMMON, SPARSE, None, PANEL)`                          | `_CommonSparsePanelProcedure`                    | Cross-asset t on `E[β_i]` on dummy |
| `(COMMON, CONTINUOUS, None, TIMESERIES)`                 | `_TSBetaContTimeseriesProcedure`                 | NW HAC t on β; ADF on factor    |
| `(_SCOPE_COLLAPSED, SPARSE, None, TIMESERIES)`           | `_TSDummySparseTimeseriesProcedure`              | NW HAC t on β; Ljung-Box on ε   |

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
    n_obs: int
    warnings: frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode] = frozenset()
    stats: Mapping[StatCode, float] = field(default_factory=dict)
```

- `verdict(*, threshold=0.05, gate=None) -> Verdict` — `gate=None` uses `primary_p`; supplying a `StatCode` swaps the gate (raises `KeyError` if not populated)
- `diagnose() -> dict[str, Any]` — flatten `mode / n_obs / primary_p / warnings / info_notes / stats` for human or AI agent triage

Single dataclass, no per-cell subclass proliferation. Cell-specific scalars live
in `stats: Mapping[StatCode, float]` keyed by enum, not by string.

---

## Mode A / Mode B equivalence

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

- `MIN_T_HARD = 20` — `T < MIN_T_HARD` raises `InsufficientSampleError`
- `MIN_T_RELIABLE = 30` — `T < MIN_T_RELIABLE` adds `WarningCode.UNRELIABLE_SE_SHORT_SERIES`
- `auto_bartlett(T) = max(1, int(4 * (T/100)**(2/9)))` — Newey-West (1994) auto lag rule
- Hansen-Hodrick (1980) overlap floor: `max(auto_bartlett(T), forward_periods - 1)` — ensures NW lag covers MA(h-1) structure from overlapping forward returns

`factrix/_types.py` keeps the older per-metric thresholds (`MIN_IC_PERIODS = 10`,
`MIN_EVENTS = 10`, etc.) used internally by the metric primitives that
procedures wrap.

---

## BHY family partitioning

`factrix/_multi_factor.py::bhy(profiles, *, threshold=0.05, gate=None)`:

1. partitions `profiles` by `_family_key(profile)` — derived from
   `(profile.config.scope, profile.config.signal, profile.config.metric)`
   with the same `_SCOPE_COLLAPSED` collapse rule applied at evaluate-time so
   Mode A and Mode B sparse profiles aggregate to one family.
2. within each family, runs Benjamini-Yekutieli step-up correction at the
   given FDR threshold and returns the survivors.

User does **not** pass a group key — same-test-family is enforced by the
config triple, not by user discipline.

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
│   └── constants.py         # MIN_T_HARD / MIN_T_RELIABLE / auto_bartlett
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
5. `FactorProfile.primary_p` is a real probability for every legal cell × mode. Mode B never returns a degenerate `primary_p = 1.0`.
6. `verdict()` reads `primary_p` (or a user-supplied `StatCode` gate); `warnings` and `info_notes` never auto-rebind it.
7. BHY family key is derived from the config triple, not user-supplied. Same-test-family is mechanical, not by discipline.
8. `register(...)` is append-only at import time. Duplicate keys raise `ValueError`.
9. NW HAC lag selection in panel-aggregation cells uses `max(auto_bartlett(T), forward_periods - 1)` — the Hansen-Hodrick floor must not be skipped under overlapping forward returns.
10. `T < MIN_T_HARD` raises `InsufficientSampleError`; procedures never silently produce a result on under-sample data.

---

## Testing

`tests/` covers the v0.5 surface only — v0.4 tests were removed in the §8.2
deletion sweep. Fixtures are fully synthetic (`tests/conftest.py` +
`factrix.datasets`); no test reads real market data from disk.

Run: `uv run pytest`
