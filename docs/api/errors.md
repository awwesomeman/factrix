---
title: Errors
---

How to read factrix errors and which exception class to catch.

## TL;DR

```python
import factrix as fx

try:
    profile = fx.evaluate(panel, cfg)
except fx.UserInputError as exc:
    # User typed the wrong thing — typo, unknown name, wrong column.
    # The message carries a fuzzy suggestion + a docs link.
    print(exc)
except fx.ConfigError as exc:
    # AnalysisConfig validation / dispatch failure.
    # exc.suggested_fix may carry a nearest-legal AnalysisConfig.
    ...
except fx.FactrixError as exc:
    # Catch-all for anything else factrix raises.
    ...
```

All factrix-raised exceptions inherit from `FactrixError`, so a single
`except fx.FactrixError` blocks every library-raised failure.

## Exception hierarchy

```
FactrixError                       # base
├── ConfigError                    # AnalysisConfig validation / dispatch
│   ├── MissingConfigError         # evaluate(panel) called without a config
│   ├── IncompatibleAxisError      # (scope, signal, metric) is not a legal cell
│   ├── ModeAxisError              # legal cell, no procedure at runtime mode
│   └── InsufficientSampleError    # T below MIN_PERIODS_HARD on a TIMESERIES procedure
└── UserInputError                 # named-set typo / type mismatch
```

| Exception | When you see it | What it carries |
|---|---|---|
| `MissingConfigError` | `evaluate(panel)` called without an `AnalysisConfig` | — |
| `IncompatibleAxisError` | `(scope, signal, metric)` is not a legal cell | optional `.suggested_fix` |
| `ModeAxisError` | Legal cell has no procedure at the runtime `Mode` | typically `.suggested_fix: AnalysisConfig` |
| `InsufficientSampleError` | `T` below the procedure floor | `.actual_periods`, `.required_periods` |
| `UserInputError` | Unknown metric / `p_stat` / context key, column not in panel, wrong type | structured `.field`, `.value`, `.candidates`, `.suggestions`, `.expected`, `.docs_url` |

---

## Error → fix mapping

Concrete messages, what triggers them, and where to look for the fix.
Use the table to skim; jump to the linked page for the why.

### Panel-schema failures

| Message hint | Trigger | Fix |
|---|---|---|
| `factor_col 'X' not in panel columns` | Typo or wrong column name | Check `panel.columns`; pass the actual name to `factor_col=`. See [Panel schema § `factor_col=`](panel-schema.md#factor_col--non-default-signal-column-name). |
| `Both 'factor' and 'X' present` | Wide panel still has stale `"factor"` column alongside the renamed one | `panel.drop("factor")` before calling. |
| `forward_return column missing` | Forgot the preprocess step | `compute_forward_return(raw, forward_periods=h)` before `evaluate`. See [Panel schema § Preprocess pipeline](panel-schema.md#preprocess-pipeline). |

### Config failures (`ConfigError` family)

| Exception / message | Trigger | Fix |
|---|---|---|
| `MissingConfigError: evaluate(panel) needs AnalysisConfig` | `evaluate(panel)` called with no `cfg` | Pass an explicit `AnalysisConfig` (one of the four factory methods on `AnalysisConfig`). |
| `IncompatibleAxisError: (scope, signal, metric) is not a legal cell` | Combination like `(INDIVIDUAL, SPARSE, IC)` that the dispatch table never registers | Use one of the four factories (`individual_continuous`, `individual_sparse`, `common_continuous`, `common_sparse`) — illegal combos are unreachable via factories. See [Concepts](../getting-started/concepts.md). |
| `ModeAxisError: no procedure at runtime Mode=TIMESERIES` | Legal cell, but `N = panel["asset_id"].n_unique()` triggers a Mode the cell does not implement (typical case: `individual_continuous` at `N=1`) | Read `.suggested_fix` — it carries the nearest-legal `AnalysisConfig` for the actual `Mode`. See [Quickstart § N = 1](../getting-started/quickstart.md). |
| `InsufficientSampleError: T below required` | `n_periods` below the procedure's `MIN_PERIODS_HARD` floor | Read `.actual_periods` and `.required_periods`. The fix is either more data, or — if the procedure is not the right one for short series — switching to a TIMESERIES-friendly cell. `evaluate()` raises; standalone metric callables (e.g. [`quantile_spread`](metrics/quantile.md#factrix.metrics.quantile.quantile_spread)) and `run_metrics()` surface the same condition differently — see [Panel vs timeseries § Sample-deficiency surfacing by entry point](../guides/panel-timeseries.md#sample-deficiency-surfacing-by-entry-point). |

### User-input failures (`UserInputError`)

Every `UserInputError` carries structured attributes (see
[Reading a `UserInputError`](#reading-a-userinputerror)). Common
triggers and fix paths:

| Message hint | Trigger | Fix |
|---|---|---|
| `unknown metric='...'` | Typo or metric not applicable to the cell | `list_metrics(scope, signal)` enumerates the applicable set; `exc.suggestions` carries the top-3 fuzzy candidates. See [`list_metrics`](list-metrics.md). |
| `unknown estimator='...'` | Typo or estimator not applicable to the cell | `list_estimators(scope, signal)` enumerates the applicable set. See [`list_estimators`](list-estimators.md). |
| `unknown expand_over='...'` | Context key not present on every profile in the family | All profiles in the family must carry the key in `.context`; check that the caller is populating it consistently at `evaluate` time. See [Cross-function reference § `expand_over`](decision-tree.md#expand_over-is-not-one-concept) for the three different `expand_over` semantics across functions. |
| `expand_over=[...] requires every profile to carry key 'X'` | One or more profiles missing the key | Confirm the upstream `evaluate` call records the key in `context=...`. |
| `Expected: list[FactorProfile], got list[MetricsBundle]` | Passing the wrong artefact family to a screening function | Screening (`bhy`, `partial_conjunction`, `bhy_hierarchical`) consumes `list[FactorProfile]` (primary-p carriers). For descriptive cross-factor views use `compare(bundles)`. See [API reference § Typical patterns](index.md#typical-patterns). |

---

## Reading a `UserInputError`

Every user-facing raise that takes a named input renders the same
three-part message:

```
bhy(): unknown expand_over='univere_id'
  Did you mean: "universe_id"?
  Available: ['regime_id', 'sector', 'universe_id']
  Docs: https://awwesomeman.github.io/factrix/api/bhy#expand_over
```

| Line | What to look at |
|---|---|
| `<func_name>(): unknown <field>=<value>` | Which kwarg / column triggered the raise, and what value was received. |
| `Did you mean: "..."` | Top-3 fuzzy candidates (omitted when nothing matches above the cutoff). |
| `Available: [...]` | The full legal set — sorted, so the same set always renders identically. |
| `Docs: https://...` | The function's deployed-docs anchor. |

For type / shape mismatches the second line reads `Expected: <shape>`
instead of `Did you mean: ...` — same three-part structure, different
diagnostic.

## Programmatic recovery

The structured attributes are the contract — read them, do not parse
the rendered message:

```python
import factrix as fx

bad: dict[str, object] = {}
for cfg in candidates:
    try:
        profiles.append(fx.evaluate(panel, cfg))
    except fx.UserInputError as exc:
        bad[exc.field] = exc.value
        # exc.suggestions carries top-3 fuzzy matches when applicable
```

| Attribute | Meaning |
|---|---|
| `func_name` | The calling function (e.g. `"bhy"`, `"evaluate"`). |
| `field` | The kwarg / column name that failed validation. |
| `value` | The value the caller passed in. |
| `candidates` | Sorted tuple of legal names (named-set branch); `()` otherwise. |
| `suggestions` | `difflib` top-3 matches against `candidates`; `()` when none. |
| `expected` | Human-readable shape (mismatch branch); `None` otherwise. |
| `docs_url` | Resolved deployed-docs URL for the function. |

## Raising your own `UserInputError`

If you build functions on top of factrix and want the same canonical
format, construct a `UserInputError` directly — it is keyword-only and
renders its own message:

```python
import factrix as fx

if metric_name not in fx.list_metrics(cfg):
    raise fx.UserInputError(
        func_name="run_metrics",
        field="metrics",
        value=metric_name,
        candidates=fx.list_metrics(cfg),
        docs_path="api/run_metrics#metrics",
    )
```

Pass exactly one of `candidates` / `expected`. The rendered message is
human-readable output; downstream code should rely on the attributes
above, not on substring matches against `str(exc)`.

---

## Class reference

Autodoc anchors for cross-references of the form
`` [`<Error>`][factrix.<Error>] `` from any docs page.

### Base

::: factrix.FactrixError
    options:
      show_root_toc_entry: false
      heading_level: 4

### User-input failures

::: factrix.UserInputError
    options:
      show_root_toc_entry: false
      heading_level: 4

### Config failures

::: factrix.ConfigError
    options:
      show_root_toc_entry: false
      heading_level: 4

::: factrix.IncompatibleAxisError
    options:
      show_root_toc_entry: false
      heading_level: 4

::: factrix.InsufficientSampleError
    options:
      show_root_toc_entry: false
      heading_level: 4

::: factrix.UnknownEstimatorError
    options:
      show_root_toc_entry: false
      heading_level: 4
