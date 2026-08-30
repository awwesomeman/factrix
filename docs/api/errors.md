---
title: Errors
---

How to read factrix errors and which exception class to catch.

## TL;DR

```python title="Illustrative"
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
    # A requested metric's effective sample is below its own hard floor.
    # exc.axis names the BINDING axis ("periods" / "assets" / "events" / ...),
    # exc.actual / exc.required carry the counts, and exc.shortfalls lists
    # one (label, reason, axis, actual, required) tuple per failing metric.
    print(f"{exc.axis}: had {exc.actual}, needed {exc.required}")
except fx.FactrixError as exc:
    # Catch-all for anything else factrix raises.
    ...
```

Every input mistake factrix rejects at an entry point — `evaluate`, a
direct metric call, `by_slice` and the slice tests, the `datasets`
builders — raises a `FactrixError` subclass, and the same mistake raises
the same class whichever entry point you used: `ic(ic_df,
overlap_periods=0)` and `evaluate(..., overlap_periods=0)` both raise
`UserInputError`, `n_groups=1` raises it from every bucketing metric, and
an unvetted `inference=` raises `IncompatibleInferenceError` whether the
sample-floor pre-flight or the metric body sees it first.

`except fx.FactrixError` does not yet block *every* library-raised
failure. What still surfaces as a builtin:

| Still builtin | Where | Why |
|---|---|---|
| `TypeError` | An argument of the wrong Python type (`by_slice(rows, ...)` on a non-frame, a metric class passed where an instance is required, a positional metric knob) | Wrong *type* is a `TypeError` by Python convention; wrong *value* is a `UserInputError`. |
| `ValueError` | Numeric knob guards on some metrics (`k_spread(k=0)`, `notional_turnover(rebalance_lag=0)`, `holding_periods`, `neutral_epsilon`) | Not yet on the structured contract; they carry a plain message and no `.field`. |
| `ValueError` | The low-level helpers in [`factrix.stats`](stats.md) (`bhy_adjust`, `romano_wolf_adjusted_p`, `stationary_bootstrap`, …) and the result exporters (`to_frame` / `to_dict` collisions) | Array-shape and finiteness guards on numeric primitives, documented per function. |

`UserInputError` multi-inherits from `ValueError`, so
`except ValueError` catches both it and everything in the table.

## Exception hierarchy

```
FactrixError                       # base
├── IncompatibleAxisError          # (scope, density, metric) is not a legal cell
├── IncompatibleInferenceError     # inference= outside the metric's applicable-inference allowlist
├── InsufficientSampleError        # a metric's effective sample is below its own SampleThreshold floor
├── UserInputError                 # named-set typo / type mismatch / dataset schema error
└── CycleError                     # MetricSpec.requires declares a dependency cycle
```

| Exception | When you see it | What it carries |
|---|---|---|
| `IncompatibleAxisError` | `(scope, density, metric)` is not a legal cell | — |
| `IncompatibleInferenceError` | `inference=` outside the metric's `applicable_inference` allowlist | `.func_name`, `.value`, `.applicable` |
| `InsufficientSampleError` | Under `strict=True`, a requested metric short-circuited on a data shortage (an `insufficient_*` reason) | `.axis`, `.actual`, `.required`, `.shortfalls` |
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
| `<metric>(): unknown factor_col='X'` | Typo in a column named on a direct metric call (`factor_col` / `return_col` / `weight_col` / `factor_cols`) | Read `.candidates` / `.suggestions` — they are the frame's own columns. A mis-named column is a caller error, not a thin sample: it never returns a NaN "insufficient data" result. |

### Structural and sample failures

| Exception / message | Trigger | Fix |
|---|---|---|
| `IncompatibleAxisError: (scope, density, metric) is not a legal cell` | Combination that the dispatch table never registers | Use compatible axes. Check [`list_metrics`](metrics/index.md#factrix.list_metrics) or [`inspect_data`](inspect-data.md) to find applicable metrics. |
| `InsufficientSampleError: N metric(s) below their sample floor` | A requested metric's effective sample is below its own hard floor | Read `.axis` first — the binding axis is not always `periods`. On `periods`, extend the window. On `assets`, either widen the universe or reconfigure the metric (`monotonicity(n_groups=3)`, `k_spread(k=2)` for a small universe). `strict=False` returns NaN placeholders instead. |

### User-input failures (`UserInputError`)

Every `UserInputError` carries structured attributes (see [Reading a `UserInputError`](#reading-a-userinputerror)). Common triggers and fix paths:

| Message hint | Trigger | Fix |
|---|---|---|
| `unknown metrics='...'` | Typo or metric not applicable to the data | `inspect_data(data).usable` enumerates the metrics applicable to the data shape. See [`list_metrics`](metrics/index.md#factrix.list_metrics) for the full catalog. |
| `invalid expand_over=[...]` | One or more `expand_over` keys missing on some results' `params` | The message lists every `(factor, missing_key)` pair in one pass. All results in the family must carry the key in `.params`; populate it consistently, or drop the key from `expand_over`. A key found on `.metadata` instead is called out separately — bookkeeping never partitions a family. |
| `Expected: list[EvaluationResult], got ...` | Passing the wrong artifact type to a screening function | Screening (`bhy`, `partial_conjunction`, `bhy_hierarchical`) consumes `list[EvaluationResult]`. |
| `unknown weight_by='...'` / `unknown tie_policy='...'` | Typo in a closed-set metric knob | Every closed-set knob (`weight_by`, `tie_policy`, `direction`, `center`, `method`) is validated before any computation and the message lists its legal values. A typo raises rather than falling through to the default branch. |
| `invalid q_top=...`, `Expected: a fraction strictly inside (0, 1)` | A fraction knob outside its bounds | `top_concentration(q_top=)` needs a genuine sub-fraction of the cross-section: 0 leaves no top bucket, 1 selects all of it. |

---

## Reading a `UserInputError`

Every user-facing raise that takes a named input renders the same three-part message:

```
bhy(): unknown expand_over='univere_id'
  Did you mean: "universe_id"?
  Available: ['regime_id', 'sector', 'universe_id']
  Docs: https://awwesomeman.github.io/factrix/api/bhy#factrix.multi_factor.bhy
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

```python title="Illustrative"
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
