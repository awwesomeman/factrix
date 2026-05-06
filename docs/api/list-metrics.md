# list_metrics

Programmatic discovery of standalone metrics applicable to a given
analysis cell. Complements [`describe_analysis_modes`](
../reference/metric-applicability.md) — that helper enumerates the
*primary procedure* per cell (IC / FM / CAAR / TS-β); `list_metrics`
returns the full set of standalone callables under
[`factrix.metrics`](metrics/index.md) that the user can additionally
invoke once `evaluate()` has produced a `FactorProfile`.

## Call shape

```python
import factrix as fl

fl.list_metrics(fl.FactorScope.INDIVIDUAL, fl.Signal.CONTINUOUS)
# -> ['top_concentration', 'beta_sign_consistency', 'fama_macbeth',
#     'pooled_ols', 'hit_rate', 'ic', 'ic_ir', 'ic_newey_west',
#     'multi_horizon_ic', 'regime_ic', 'monotonicity',
#     'multi_split_oos_decay', 'quantile_spread', 'quantile_spread_vw',
#     'greedy_forward_selection', 'spanning_alpha', 'breakeven_cost',
#     'net_spread', 'notional_turnover', 'turnover', 'ic_trend']
```

`Mode` is intentionally not an input — applicability does not change
across PANEL / TIMESERIES (see
[Metric applicability](../reference/metric-applicability.md) for the
underlying matrix).

## Structured output for tooling

```python
fl.list_metrics(
    fl.FactorScope.INDIVIDUAL,
    fl.Signal.CONTINUOUS,
    format="json",
)
# -> [
#     {
#         "name": "ic",
#         "module": "ic",
#         "cell": "(INDIVIDUAL, CONTINUOUS, IC, PANEL)",
#         "agg_order": "cs-first",
#         "inference_se": "NW HAC / cross-asset t",
#     },
#     ...
# ]
```

The JSON form is the single-source-of-truth row produced by parsing
the `Matrix-row:` tag in each metric module's docstring (the same
parser MkDocs uses to render
[the cross-metric matrix](../reference/standalone-metrics.md)). Keys:

| Key | Meaning |
|---|---|
| `name` | Function name as exported under `factrix.metrics` |
| `module` | Submodule stem (e.g. `ic`, `fama_macbeth`) |
| `cell` | Raw cell string from the `Matrix-row:` tag |
| `agg_order` | Aggregation order (`cs-first`, `ts-first`, `ts-only`, `static-cs`, `per-event`) |
| `inference_se` | Inference / SE method or `no formal H₀` for descriptive metrics |

## Errors

`IncompatibleAxisError` is raised when `(scope, signal)` matches no
registered metric. All four real combinations are populated, so this
is defensive — surfaced for symmetry with the rest of the API.

## Source of truth

`list_metrics` and the docs matrix share one parser
(`factrix._metric_index`). Adding a new metric only requires adding a
`Matrix-row:` tag to the new module's docstring; both surfaces pick it
up automatically.

::: factrix.list_metrics
