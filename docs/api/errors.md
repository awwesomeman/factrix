# Errors

How to read factrix errors and which exception class to catch.

## TL;DR

```python
import factrix as fl

try:
    profile = fl.evaluate(panel, cfg)
except fl.UserInputError as exc:
    # User typed the wrong thing — typo, unknown name, wrong column.
    # The message carries a fuzzy suggestion + a docs link.
    print(exc)
except fl.ConfigError as exc:
    # AnalysisConfig validation / dispatch failure.
    # exc.suggested_fix may carry a nearest-legal AnalysisConfig.
    ...
except fl.FactrixError as exc:
    # Catch-all for anything else factrix raises.
    ...
```

All factrix-raised exceptions inherit from `FactrixError`, so a single
`except fl.FactrixError` blocks every library-raised failure.

## Exception hierarchy

```
FactrixError                       # base
├── ConfigError                    # AnalysisConfig validation / dispatch
│   ├── MissingConfigError         # evaluate(raw) called without a config
│   ├── IncompatibleAxisError      # (scope, signal, metric) is not a legal cell
│   ├── ModeAxisError              # legal cell, no procedure at runtime mode
│   └── InsufficientSampleError    # T below MIN_PERIODS_HARD on a TIMESERIES procedure
└── UserInputError                 # named-set typo / type mismatch
```

| Exception | When you see it | What it carries |
|---|---|---|
| `MissingConfigError` | `evaluate(raw)` called without an `AnalysisConfig` | — (call `fl.suggest_config(raw)` to recover) |
| `IncompatibleAxisError` | `(scope, signal, metric)` is not a legal cell | optional `.suggested_fix` |
| `ModeAxisError` | Legal cell has no procedure at the runtime `Mode` | typically `.suggested_fix: AnalysisConfig` |
| `InsufficientSampleError` | `T` below the procedure floor | `.actual_periods`, `.required_periods` |
| `UserInputError` | Unknown metric / `p_stat` / context key, column not in panel, wrong type | structured `.field`, `.value`, `.candidates`, `.suggestions`, `.expected`, `.docs_url` |

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
| `verb(): unknown <field>=<value>` | Which kwarg / column triggered the raise, and what value was received. |
| `Did you mean: "..."` | Top-3 fuzzy candidates (omitted when nothing matches above the cutoff). |
| `Available: [...]` | The full legal set — sorted, so the same set always renders identically. |
| `Docs: https://...` | The verb's deployed-docs anchor. |

For type / shape mismatches the second line reads `Expected: <shape>`
instead of `Did you mean: ...` — same three-part structure, different
diagnostic.

## Programmatic recovery

The structured attributes are the contract — read them, do not parse
the rendered message:

```python
import factrix as fl

bad: dict[str, object] = {}
for cfg in candidates:
    try:
        profiles.append(fl.evaluate(panel, cfg))
    except fl.UserInputError as exc:
        bad[exc.field] = exc.value
        # exc.suggestions carries top-3 fuzzy matches when applicable
```

| Attribute | Meaning |
|---|---|
| `verb` | The calling verb (e.g. `"bhy"`, `"evaluate"`). |
| `field` | The kwarg / column name that failed validation. |
| `value` | The value the caller passed in. |
| `candidates` | Sorted tuple of legal names (named-set branch); `()` otherwise. |
| `suggestions` | `difflib` top-3 matches against `candidates`; `()` when none. |
| `expected` | Human-readable shape (mismatch branch); `None` otherwise. |
| `docs_url` | Resolved deployed-docs URL for the verb. |

## Raising your own `UserInputError`

If you build verbs on top of factrix and want the same canonical
format, construct a `UserInputError` directly — it is keyword-only and
renders its own message:

```python
import factrix as fl

if metric_name not in fl.list_metrics(cfg):
    raise fl.UserInputError(
        verb="run_metrics",
        field="metrics",
        value=metric_name,
        candidates=fl.list_metrics(cfg),
        docs_path="api/run_metrics#metrics",
    )
```

Pass exactly one of `candidates` / `expected`. The rendered message is
human-readable output; downstream code should rely on the attributes
above, not on substring matches against `str(exc)`.
