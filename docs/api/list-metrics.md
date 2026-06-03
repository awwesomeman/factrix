---
title: factrix.list_metrics
---

::: factrix.list_metrics

Programmatic discovery of standalone metrics applicable to a given
analysis cell. Complements [`describe_analysis_modes`](
../reference/metric-applicability.md) — that helper enumerates the
*primary procedure* per cell (IC / FM / CAAR / TS-β); `list_metrics`
returns the full set of standalone callables under
[`factrix.metrics`](metrics/index.md) that the user can additionally
invoke once `evaluate()` has produced a `FactorProfile`.

## DataStructure axis is not an input

`DataStructure` is intentionally not a parameter — applicability does not
change across PANEL / TIMESERIES (see
[Metric applicability](../reference/metric-applicability.md) for the
underlying matrix). See the docstring Examples block above for the
canonical text-list and JSON-form calls.

## Discover-then-import workflow

Pass `with_import=True` to render a copy-paste-ready two-column view
that pairs each metric with its submodule path:

```python
print("\n".join(fx.list_metrics(
    fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE, with_import=True,
)))
# ic                       → factrix.metrics.ic
# ic_ir                    → factrix.metrics.ic
# fm_beta             → factrix.metrics.fm_beta
# breakeven_cost           → factrix.metrics.tradability
# ...
```

Every name on the right is also re-exported from `factrix.metrics`,
so the canonical wire-up is always:

```python
from factrix.metrics import ic, fm_beta, breakeven_cost
```

The submodule path is shown so you know *where the implementation
lives* — useful when reading source, jumping in an IDE, or checking
the module-level `Matrix-row:` tag.

## Structured output for tooling

```python
fx.list_metrics(
    fx.FactorScope.INDIVIDUAL,
    fx.FactorDensity.DENSE,
    format="json",
)
# -> [
#     {
#         "name": "ic",
#         "module": "ic",
#         "cell": "(INDIVIDUAL, DENSE, IC, PANEL)",
#         "agg_order": "cs-first",
#         "inference_se": "NW HAC / cross-asset t",
#         "import_path": "factrix.metrics.ic",
#         "input_kind": "panel",
#     },
#     ...
# ]
```

The JSON form is the single-source-of-truth row produced by parsing
the `Matrix-row:` tag in each metric module's docstring (the same
parser MkDocs uses to render
[the cross-metric matrix](../reference/metric-pipelines.md)). Keys:

| Key | Meaning |
|---|---|
| `name` | Function name as exported under `factrix.metrics` |
| `module` | Submodule stem (e.g. `ic`, `fm_beta`) |
| `cell` | Raw cell string from the `Matrix-row:` tag |
| `agg_order` | Aggregation order (`cs-first`, `ts-first`, `ts-only`, `static-cs`, `per-event`) |
| `inference_se` | Inference / SE method or `no formal H₀` for descriptive metrics |
| `import_path` | Fully-qualified submodule (`factrix.metrics.<module>`) — also re-exported from `factrix.metrics` |
| `input_kind` | `"panel"` for the standard `(date-keyed DataFrame, **kwargs) -> MetricResult` contract; `"scalar"` for pre-aggregated-scalar utilities |

### `panel` vs `scalar` — and why it matters

Most metrics take a date-keyed DataFrame as their first positional
argument; a few (`breakeven_cost`, `net_spread` in
`factrix.metrics.tradability`) consume pre-aggregated scalars
(`gross_spread: float`, `turnover: float`, …) and return a
`MetricResult` directly. The date-slicing dispatcher
[`by_slice`](by-slice.md) only accepts the `panel` shape — there is
no date column in a scalar to slice on.

Filter the JSON output to enumerate `by_slice`-eligible metrics:

```python
panel_metrics = [
    r for r in fx.list_metrics(
        fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE, format="json",
    )
    if r["input_kind"] == "panel"
]
```

## Errors

[`IncompatibleAxisError`][factrix.IncompatibleAxisError] is raised when `(scope, signal)` matches no
registered metric. All four real combinations are populated, so this
is defensive — surfaced for symmetry with the rest of the API.

## Source of truth

`list_metrics` and the docs matrix share one parser
(`factrix._metric_index`). Adding a new metric only requires adding a
`Matrix-row:` tag to the new module's docstring; both surfaces pick it
up automatically.
