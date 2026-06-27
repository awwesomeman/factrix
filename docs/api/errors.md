---
title: Errors
---

How to read factrix errors and which exception class to catch.

## TL;DR

```python
import factrix as fx
from factrix.metrics import ic

try:
    results = fx.evaluate(
        data, 
        metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)}, 
        factor_cols=["factor"]
    )
except fx.UserInputError as exc:
    # User typed the wrong thing — typo, unknown name, wrong column.
    # The message carries a fuzzy suggestion + a docs link.
    print(exc)
except fx.IncompatibleAxisError as exc:
    # Axis miswire.
    ...
except fx.InsufficientSampleError as exc:
    # Sample threshold is below the required hard floor.
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
├── IncompatibleInferenceError     # inference= outside the metric's applicable-inference allowlist
├── InsufficientSampleError        # T below SampleThreshold on a TIMESERIES/PANEL procedure
├── UserInputError                 # named-set typo / type mismatch / dataset schema error
└── CycleError                     # MetricSpec.requires declares a dependency cycle
```

| Exception | When you see it | What it carries |
|---|---|---|
| `IncompatibleAxisError` | `(scope, density, metric)` is not a legal cell | — |
| `IncompatibleInferenceError` | `inference=` outside the metric's `applicable_inference` allowlist | `.func_name`, `.value`, `.applicable` |
| `InsufficientSampleError` | `T` below the procedure floor | `.actual_periods`, `.required_periods` |
| `UserInputError` | Unknown metric, column not in data, wrong type | structured `.field`, `.value`, `.candidates`, `.suggestions`, `.expected`, `.docs_url` |
| `CycleError` | A custom metric's `MetricSpec.requires` forms a dependency cycle | — |

---

## Error → fix mapping

Concrete messages, what triggers them, and where to look for the fix.

### Data-schema failures

| Message hint | Trigger | Fix |
|---|---|---|
| `factor_cols 'X' not in data columns` | Typo or wrong column name | Check `data.columns`; pass the actual name to `factor_cols=`. See [Data schema](data-schema.md). |
| `forward_return column missing` | Forgot the preprocess step | `compute_forward_return(raw, forward_periods=h)` before `evaluate`. See [Preparing data](../guides/preparing-data.md). |

### Structural and sample failures

| Exception / message | Trigger | Fix |
|---|---|---|
| `IncompatibleAxisError: (scope, density, metric) is not a legal cell` | Combination that the dispatch table never registers | Use compatible axes. Check [`list_metrics`](metrics/index.md#factrix.list_metrics) or [`inspect_data`](inspect-data.md) to find applicable metrics. |
| `InsufficientSampleError: T below required` | Sample size below the procedure's hard floor | Read `.actual_periods` and `.required_periods`. The fix is either more data, or switching to a less restrictive metric. |

### User-input failures (`UserInputError`)

Every `UserInputError` carries structured attributes (see [Reading a `UserInputError`](#reading-a-userinputerror)). Common triggers and fix paths:

| Message hint | Trigger | Fix |
|---|---|---|
| `unknown metrics='...'` | Typo or metric not applicable to the data | `inspect_data(data).usable` enumerates the metrics applicable to the data shape. See [`list_metrics`](metrics/index.md#factrix.list_metrics) for the full catalog. |
| `invalid expand_over=[...]` | One or more `expand_over` context keys missing on some results | The message lists every `(factor, missing_key)` pair in one pass. All results in the family must carry the key in `.context`; populate it consistently, or drop the key from `expand_over`. |
| `Expected: list[EvaluationResult], got ...` | Passing the wrong artifact type to a screening function | Screening (`bhy`, `partial_conjunction`, `bhy_hierarchical`) consumes `list[EvaluationResult]`. |

---

## Reading a `UserInputError`

Every user-facing raise that takes a named input renders the same three-part message:

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
| `Available: [...]` | The full legal set. |
| `Docs: https://...` | The function's deployed-docs anchor. |

For type / shape mismatches, the second line reads `Expected: <shape>` instead of `Did you mean: ...`.

## Programmatic recovery

The structured attributes are the contract — read them, do not parse the rendered message:

```python
import factrix as fx

bad: dict[str, object] = {}
for factor_col in candidates:
    try:
        results = fx.evaluate(data, metrics=metrics, factor_cols=[factor_col])
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

---

## Class reference

Autodoc anchors for cross-references of the form `[`FactrixError`][factrix.FactrixError]` from any docs page.

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

::: factrix.IncompatibleInferenceError
    options:
      show_root_toc_entry: false
      heading_level: 4

::: factrix.InsufficientSampleError
    options:
      show_root_toc_entry: false
      heading_level: 4

### Custom-metric wiring failures

::: factrix.CycleError
    options:
      show_root_toc_entry: false
      heading_level: 4
