# factrix Architecture

Current-state snapshot — describes the library as it stands, not how it got
here. For design-process history see the originating workspace's `docs/`
(spike docs, refactor plans) in the `awwesomeman/factor-analysis` repo.

---

## Positioning

**factrix is a Factor Signal Analyzer, not a backtest engine.**

Metrics like `turnover`, `breakeven_cost`, and `net_spread` are idealized
proxies (equal-weight, zero slippage) used to screen signal quality; they
are not tradable P&L. For realistic execution simulation, feed screened
factors into Zipline / Backtrader / a proprietary engine.

---

## Public API surface

Four entry points, all in `factrix.__init__`:

| Function | Purpose |
|----------|---------|
| `fl.preprocess(df, config=...)` | Attach `forward_return` column, attach `_fl_forward_periods` marker, dispatch to per-type orchestrator |
| `fl.evaluate(prepared_df, factor_name, config=...)` | Single-factor evaluation → typed `Profile` dataclass |
| `fl.evaluate_batch(prepared_df, factor_names, config=..., compact=, keep_artifacts=)` | Multi-factor evaluation → `ProfileSet` |
| `fl.factor(factor_type, ...)` | Factory for `Factor` session (caches primitives, supports per-call overrides) |

Both `evaluate` and `evaluate_batch` accept `return_artifacts=True` to expose
the underlying `Artifacts` object for drill-down into intermediate
`MetricOutput` dictionaries.

---

## Four factor types

Each type has its own `Config`, `Profile` dataclass, `Factor` subclass, and
preprocess orchestrator. Mixing types in a single `ProfileSet` raises
`TypeError`.

| Type | Config | Profile | Factor subclass | Canonical p-value source |
|------|--------|---------|-----------------|--------------------------|
| cross_sectional | `CrossSectionalConfig` | `CrossSectionalProfile` | `CrossSectionalFactor` | IC t-test p |
| event_signal | `EventConfig` | `EventProfile` | `EventFactor` | CAAR p on main event window |
| macro_panel | `MacroPanelConfig` | `MacroPanelProfile` | `MacroPanelFactor` | Fama-MacBeth λ p |
| macro_common | `MacroCommonConfig` | `MacroCommonProfile` | `MacroCommonFactor` | time-series β p |

The `Factor` base class itself is deliberately **not** re-exported in
`fl.__all__` — factory `fl.factor()` is the canonical entry. Subclasses
are exported so `isinstance(x, CrossSectionalFactor)` works.

---

## Profile dataclass contract

- `frozen=True, slots=True` — Profiles are immutable value objects
- `verdict()` returns `"PASS" | "PASS_WITH_WARNINGS" | "FAILED"`. `PASS_WITH_WARNINGS` fires when `canonical_p` passes but a warn-severity diagnostic names a whitelisted alternative `recommended_p_source` the user has not adopted. No `CAUTION` bucket — that shape was deliberately removed.
- `canonical_p` is per-type hard-coded (see table above); `P_VALUE_FIELDS`
  is the cross-factor whitelist used when BHY-correcting a `ProfileSet`
- `diagnose()` returns `list[Diagnostic]` — diagnostic rules are evaluated
  lazily per call, allowing user-registered rules via `register_rule`
- Canonical p field **must** appear in `P_VALUE_FIELDS`

---

## Artifacts and compact mode

`Artifacts` holds the raw computation output (IC series, quantile returns,
event matrices, intermediates). It's hidden by default. Opt-in paths:

- `evaluate(..., return_artifacts=True)` → `(profile, artifacts)` tuple
- `evaluate_batch(..., keep_artifacts=True)` — keeps all artifacts in
  `ProfileSet.artifacts` (memory-expensive for large batches)
- `evaluate_batch(..., compact=True)` — keeps only a minimal subset
  (`MetricOutput.compact()`), suitable for 1000+ factor sweeps

Level 2 helpers (`regime_ic`, `multi_horizon_ic`, `spanning_alpha`,
orthogonalization) write their 1-row DataFrame results into
`artifacts.intermediates` with stable keys.

---

## ProfileSet and multiple testing

`ProfileSet` is a homogeneous container (single Profile type). Key methods:

- `multiple_testing_correct(p_source=..., fdr=0.05, method="bhy")` —
  adds a `<method>_significant` column and `mt_method` marker column
- `to_polars()` — flatten to a wide DataFrame for filtering / ranking
- `rank_by(field)`, `top(n)`, `filter(...)` — chainable selectors
- `describe_profile_values(...)` — inspect field completeness
- `diagnose_all()` — flatten every profile's diagnostics into one polars
  DataFrame (`factor_name, severity, code, message, recommended_p_source`)
  for zoo-scale triage
- `with_canonical(field)` — rebind the canonical p-source for downstream
  `multiple_testing_correct` and the `canonical_p` alias column; does
  not mutate individual profile dataclasses

Invariant: `multiple_testing_correct`'s `n_total` must be `>= len(self)`
(correction can be applied over a larger universe than what was kept).

---

## Diagnostics vs Canonical — design philosophy

factrix deliberately keeps **two decision surfaces distinct**:

1. `canonical_p` (bound to `CANONICAL_P_FIELD`) — the single authoritative
   p-value fed to BHY. Stable per Profile class; never auto-switches.
2. `diagnose()` — a list of `Diagnostic(severity, code, message,
   recommended_p_source)` surfacing risks (clustering, factor persistence,
   overlap-induced IC inflation). Rules may recommend an alternative
   p-value via `recommended_p_source`, but the framework **does not**
   silently rebind canonical based on diagnostics.

Why not auto-switch? Three reasons:
- **Reproducibility**: canonical tied to a rule threshold (e.g. HHI>0.2)
  means the BHY input set depends on sample-dependent diagnostics. Two
  runs on slightly different data could silently pick different p-values.
- **Hidden assumptions**: a user inspecting `canonical_p=0.01` should be
  able to name the test without cross-referencing dynamic state.
- **Threshold fragility**: rule cutoffs (HHI 0.2 vs 0.25, ADF p 0.05 vs
  0.10) are themselves judgement calls that should not drive first-order
  statistical inference.

Instead:
- `diagnose()` surfaces risk flags as data.
- `verdict()` exposes `PASS_WITH_WARNINGS` when a warn-severity diagnostic
  names a defensible alternative the user has not adopted — a UX hint, not
  a severity grade.
- `ProfileSet.with_canonical(field)` lets the user **explicitly** rebind
  for zoo-scale BHY.
- `factrix.evaluation` logger emits INFO on each `multiple_testing_correct`
  call and WARNING when `PASS_WITH_WARNINGS` fires. `factrix.metrics`
  logger emits DEBUG per correction (sample shrink, NW lags) and WARNING
  for degenerate regimes (sample < 1.5×min, lags×5 > T).

Slogan: **framework detects risk, user decides the correction**.

Caveat: some diagnostics have **config-level** remediations rather than
an alternative p-value (e.g. `event.clustering_high` should be fixed by
setting `EventConfig.adjust_clustering='kolari_pynnonen'`, not by swapping
p-source). These rules carry no `recommended_p_source` and therefore do
not upgrade `verdict()` to `PASS_WITH_WARNINGS`. The message field names
the config lever instead. Users who want a uniform "any diagnose warning
implies verdict caveat" view can filter `diagnose_all()` directly.

---

## Factor session (caching + override)

`fl.factor(factor_type, ...)` returns a `Factor` session bound to a
(df, config) pair. Primitives are cached by key — the cache key **must**
match `MetricOutput.name` (asserted on the override path to catch
name/key drift).

Per-call overrides (e.g., sensitivity sweeps):
```python
f = fl.factor("cross_sectional", df=prepared, name="Mom_20D")
f.breakeven_cost(n_groups=5)         # override n_groups
f.net_spread(estimated_cost_bps=25)  # override cost assumption
```

Overrides emit a `UserWarning` (advisory — the user is deviating from the
bound config). See `factrix/factor.py`.

---

## Module layout

```
factrix/
├── _api.py                   # fl.evaluate / evaluate_batch / factor / redundancy_matrix
├── _types.py                 # Diagnostic, FactorType, MetricOutput, PValue, Verdict
├── _ols.py, _stats.py        # numeric primitives
├── config.py                 # {CrossSectional,Event,MacroPanel,MacroCommon}Config
├── factor.py                 # Factor base + 4 subclasses, caching, override path
├── validation.py             # validate_factor_data
├── reporting.py              # describe_profile_values
├── adapt.py                  # fl.adapt (deprecated path, retained for compat)
├── preprocess/               # per-type preprocess orchestrators
├── evaluation/
│   ├── pipeline.py           # build_artifacts, compute_spread_series
│   ├── profiles/             # {CrossSectional,Event,MacroPanel,MacroCommon}Profile
│   ├── profile_set.py        # ProfileSet
│   └── diagnostics.py        # Rule, register_rule, clear_custom_rules
├── metrics/                  # ic, caar, hit_rate, monotonicity, concentration,
│                             # quantile, fama_macbeth, ts_beta, tradability,
│                             # trend, spanning, clustering, corrado, event_quality
├── stats/                    # bhy_adjust, bhy_adjusted_p, small-sample p-value
├── charts/                   # quantile, monotonicity (plotly)
├── factors/                  # reference factor library (Mom, Rev, Vol, …)
└── integrations/
    └── mlflow.py             # run.log_factrix_artifacts helper
```

---

## Invariants

Hard constraints — violating these breaks the API contract:

1. `Profile.verdict()` returns `PASS | PASS_WITH_WARNINGS | FAILED`; never `CAUTION`.
2. `Profile` dataclass is `frozen=True, slots=True`.
3. `CANONICAL_P_FIELD` for each Profile type must appear in its
   `P_VALUE_FIELDS`.
4. `ProfileSet` holds a single Profile type; mixing raises `TypeError`.
5. `P_VALUE_FIELDS` is a cross-factor whitelist (same test applied to
   the whole BHY batch).
6. `config.ortho` is `OrthoConfig | pl.DataFrame | None`.
7. `ortho_stats` / `regime_stats` / `multi_horizon_stats` / `spanning_stats`
   are 1-row DataFrames living in `artifacts.intermediates`.
8. `ProfileSet.mt_method` column means "correction applied" (marker).
9. `multiple_testing_correct(n_total=N)` requires `N >= len(self)`.
10. `fl.factor()` supports all four `factor_type` values; the `Factor`
    base class is not in `fl.__all__` — factory is the canonical entry.
11. `Factor` cache key **must** equal the primitive's `MetricOutput.name`
    (asserted on override path).

---

## Testing

`tests/` mirrors `factrix/` layout. Fixtures are fully synthetic — no
test reads real market data from disk. 555 tests total as of v0.1.0.

Run: `uv run pytest`
