# factorlib Architecture

Current-state snapshot — describes the library as it stands, not how it got
here. For design-process history see the originating workspace's `docs/`
(spike docs, refactor plans) in the `awwesomeman/factor-analysis` repo.

---

## Positioning

**factorlib is a Factor Signal Analyzer, not a backtest engine.**

Metrics like `turnover`, `breakeven_cost`, and `net_spread` are idealized
proxies (equal-weight, zero slippage) used to screen signal quality; they
are not tradable P&L. For realistic execution simulation, feed screened
factors into Zipline / Backtrader / a proprietary engine.

---

## Public API surface

Four entry points, all in `factorlib.__init__`:

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
- `verdict()` returns a **binary** `"PASS" | "FAILED"` (no `CAUTION` — deliberately removed)
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

Invariant: `multiple_testing_correct`'s `n_total` must be `>= len(self)`
(correction can be applied over a larger universe than what was kept).

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
bound config). See `factorlib/factor.py`.

---

## Module layout

```
factorlib/
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
    ├── mlflow.py             # run.log_factorlib_artifacts helper
    └── streamlit/            # factor-lab dashboard (`factor-lab` CLI entry)
```

---

## Invariants

Hard constraints — violating these breaks the API contract:

1. `Profile.verdict()` is binary `PASS | FAILED`; never `CAUTION`.
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

`tests/` mirrors `factorlib/` layout. Fixtures are fully synthetic — no
test reads real market data from disk. 555 tests total as of v0.1.0.

Run: `uv run pytest`
