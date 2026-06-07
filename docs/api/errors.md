---
title: Errors
---

How to read factrix errors and which exception class to catch.

## TL;DR

```python
import factrix as fx

try:
    results = fx.evaluate(data, metrics={"ic": ic()}, factor_cols=["factor"])
except fx.UserInputError as exc:
    # User typed the wrong thing — typo, unknown name, wrong column.
    # The message carries a fuzzy suggestion + a docs link.
    print(exc)
except fx.IncompatibleAxisError as exc:
    # Axis miswire.
    ...
except fx.InsufficientSampleError as exc:
    # n_periods or other axis is below the required hard floor.
    # exc.actual_periods and exc.required_periods carry details.
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
├── IncompatibleAxisError          # (scope, density, metric) is not a legal cell
├── InsufficientSampleError        # T below MIN_PERIODS_HARD on a TIMESERIES procedure
├── UnknownEstimatorError          # lookup miss in get_estimator
└── UserInputError                 # named-set typo / type mismatch / dataset schema error
```

| Exception | When you see it | What it carries |
|---|---|---|
| `IncompatibleAxisError` | `(scope, density, metric)` is not a legal cell | — |
| `InsufficientSampleError` | `T` below the procedure floor | `.actual_periods`, `.required_periods` |
| `UnknownEstimatorError` | `get_estimator(name)` lookup miss | — |
| `UserInputError` | Unknown metric / estimator / `primary` label, column not in data, wrong type | structured `.field`, `.value`, `.candidates`, `.suggestions`, `.expected`, `.docs_url` |

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

### Structural and sample failures

| Exception / message | Trigger | Fix |
|---|---|---|
| `IncompatibleAxisError: (scope, density, metric) is not a legal cell` | Combination like `(INDIVIDUAL, SPARSE, IC)` that the dispatch table never registers | Use compatible axes. Check [`list_metrics`](list-metrics.md) or [`inspect_data`](../guides/reading-results.md) to find applicable metrics. |
| `InsufficientSampleError: T below required` | `n_periods` below the procedure's hard floor | Read `.actual_periods` and `.required_periods`. The fix is either more data, or switching to a TIMESERIES-friendly metric. See [Panel vs timeseries](../guides/panel-timeseries.md). |

### User-input failures (`UserInputError`)

Every `UserInputError` carries structured attributes (see
[Reading a `UserInputError`](#reading-a-userinputerror)). Common
triggers and fix paths:

| Message hint | Trigger | Fix |
|---|---|---|
| `unknown metric='...'` | Typo or metric not applicable to the cell | `inspect_data(panel).usable` enumerates the metrics applicable to a panel; `exc.suggestions` carries the top-3 fuzzy candidates. See [`list_metrics`](list-metrics.md) for the full catalog. |
| `unknown estimator='...'` | Typo or estimator not applicable to the cell | `list_estimators()` enumerates every registered estimator. See [`list_estimators`](list-estimators.md). |
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

known = {spec.name for specs in fx.list_metrics().values() for spec in specs}
if metric_name not in known:
    raise fx.UserInputError(
        func_name="run_metrics",
        field="metrics",
        value=metric_name,
        candidates=sorted(known),
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

### Structural and sample failures

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
