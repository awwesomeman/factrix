---
title: factrix.evaluate
---

> **Input contracts:** `evaluate` consumes an evaluation panel with the
> four-column floor documented in [Data schema](data-schema.md).
> `evaluate_horizons` consumes a raw canonical panel with `price` and factor
> columns, then computes `forward_return` internally for each horizon.
> Scalar post-processing helpers such as `breakeven_cost` and `net_spread`
> are direct-call utilities, not `evaluate()` metrics.

::: factrix.evaluate

<hr>

::: factrix.evaluate_horizons

<hr>

## `forward_periods=` and `overlap_periods=`

Both are properties of the data, normally read from the stamps
[`compute_forward_return`](preprocess.md) leaves on the panel, and never
per-metric knobs. They answer different questions:

| Argument | Question | Read from | Declare it when |
|---|---|---|---|
| `forward_periods` | Over how many periods of the price grid was the return measured? The hypothesis; joins the `(factor, forward_periods, *params)` identity. | `_forward_periods` stamp | The panel carries a self-attached `forward_return` with no stamp. |
| `overlap_periods` | How many adjacent evaluation observations share future periods? What inference consumes. | `_overlap_periods` stamp (equal to the horizon on the full grid; derived from `dates=` on a coarser one) | The unstamped panel sits on a coarser evaluation grid than its horizon. Otherwise it defaults to `forward_periods`. |

A value that disagrees with its stamp raises [`UserInputError`](errors.md)
rather than silently overriding the data. Both surface on the result —
`forward_periods` as identity, `overlap_periods` as bookkeeping — and
`overlap_periods` is injected into every metric as
`metadata["overlap_periods"]`. See
[Evaluating on a coarser grid](preprocess.md#evaluating-on-a-coarser-grid)
for why the two are kept apart.

### Two more period counts the tradability metrics declare

On a full evaluation grid all four counts below are the same number, which is
why one name used to carry all of them. They separate as soon as the
evaluation grid is coarser than the return horizon.

| Quantity | Question | Unit | Declared on |
|---|---|---|---|
| `forward_periods` | Economic horizon: over how many periods was the return measured? | Underlying period grid | `compute_forward_return`; stamped |
| `overlap_periods` | Inference: how many adjacent evaluation observations share future periods? | Evaluation-grid observations | Derived and stamped; injected, never a metric knob |
| `rebalance_lag` | Turnover: how far apart are the rankings / memberships compared? | Evaluation-grid observations | `rank_turnover()`, `notional_turnover()` |
| `holding_periods` | Cost amortisation: how many return periods pass between paying trading cost? | Underlying period grid | `breakeven_cost()`, `net_spread()` |

`rebalance_lag` defaults to the injected `overlap_periods` — horizon-aligned
turnover, the historical behaviour. Pass `rebalance_lag=1` when the evaluation
grid *is* the rebalance schedule. `holding_periods` has no default tie to the
stamp: `gross_spread` is per underlying return period (`compute_forward_return`
divides by `forward_periods`), so the cost must be amortised over underlying
periods too. Substituting the derived evaluation-grid overlap there is a unit
error, not a conservative choice — see the
[tradability migration note](metrics/tradability.md#migration--the-holding_periods-rename).

## Use cases

<div class="grid cards" markdown>

-   __Single-factor significance__

    ---

    One panel of factor data → one result carrying the mainstream metric's
    `p_value` and the cell-specific statistics.

-   __Batch screening with false discovery rate (FDR)__

    ---

    Loop `evaluate` over candidate signal columns and feed the
    resulting `EvaluationResult` list to [`bhy`][factrix.multi_factor.bhy]
    for false-discovery-rate control. See
    [Multi-factor FDR](multi-factor.md).

-   __Cross-cell apples-to-apples__

    ---

    Compare information coefficient (IC) rank-ordering against
    Fama-MacBeth λ on the same panel, or individual-asset factors
    against broadcast macro factors. Return shape is identical across
    cells.

-   __TIMESERIES dispatch__

    ---

    At `n_assets == 1` there is no cross-section, so any `DENSE` metric whose
    cell is `PANEL` — `Individual × Continuous` (`ic`, `fm_beta`) **and**
    `Common × Continuous` (`common_beta`, `common_quantile`, `common_asymmetry`) —
    raises `IncompatibleAxisError` (or NaN + `structure_mismatch` under
    `strict=False`). Single-asset data runs through the same entry point
    with `predictive_beta` for dense predictive-regression slopes, sparse
    metrics whose cell wildcard allows `TIMESERIES`, and panel-input wildcard
    metrics such as `directional_hit_rate`. Two-column diagnostics
    (`positive_rate`, `oos_decay`, `ic_trend`) are standalone `(date, value)`
    tools; in `evaluate()` they layer on panel IC series, not raw
    single-asset dense panels.

</div>

## Worked example — single-factor smoke test

!!! example "Synthetic panel → `evaluate` → read `value` + `p_value`"

    Full runnable example complementing the doctest snippets in **Examples**
    above with realistic console output.

    ```python
    import factrix as fx
    from factrix.metrics import ic, quantile_spread

    # 1. Create dummy panel data
    raw = fx.datasets.make_cs_panel(n_assets=15, n_dates=80)
    data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

    # 2. Run evaluation
    results = fx.evaluate(
        data,
        metrics={"ic": ic(), "spread": quantile_spread(n_groups=5)},
        factor_cols=["factor"],
        forward_periods=5,
    )

    # 3. Retrieve and inspect results
    res = results["factor"]
    print(f"Factor: {res.factor}")
    print(f"Cell: {res.cell}")
    print(f"Plan: \n{res.plan}")

    # Access metrics result group
    ic_res = res.metrics["ic"]
    print(f"IC Value: {ic_res.value:.4f}")
    print(f"IC p-value: {ic_res.p_value:.4f}")
    ```

## Sensitivity grids

For exploratory grids across asset counts, horizons, or factor families, run
with `strict=False` and stack each result's long-form table. The table carries
`is_applicable` and `reason`, so the grid can keep running while still making
failed metric/input combinations visible.

```python
import polars as pl

results = fx.evaluate(
    data,
    metrics={"ic": ic(), "spread": quantile_spread(n_groups=2)},
    factor_cols=["factor"],
    forward_periods=5,
    strict=False,
)

status = pl.concat([r.to_frame() for r in results.values()])
failed = status.filter(~pl.col("is_applicable"))
```

## Evaluating under different cell contexts

Metric behaviors are defined by instantiating metric classes directly. The DAG executor handles dispatch automatically depending on the cell registered by the metric.

```python
import factrix as fx
from factrix.metrics import ic, caar, common_beta

# 1. Individual × Continuous (e.g. Information Coefficient)
results_ic = fx.evaluate(
    data,
    metrics={"ic": ic()},
    factor_cols=["factor"],
    forward_periods=5
)

# 2. Individual × Sparse (e.g. Event Study CAAR, requires a 'price' column)
results_caar = fx.evaluate(
    data_with_price,
    metrics={"caar": caar()},
    factor_cols=["event_factor"],
    forward_periods=5
)

# 3. Common × Continuous (e.g. Time-Series Beta)
results_common_beta = fx.evaluate(
    data,
    metrics={"common_beta": common_beta()},
    factor_cols=["macro_factor"],
    forward_periods=5
)
```

Per-cell required / optional columns and the DataStructure (PANEL vs TIMESERIES)
derivation are automatically resolved at dispatch time.

## Next steps

<div class="grid cards" markdown>

-   **Multi-factor FDR**

    ---

    Wires `evaluate` into the multi-factor FDR pipeline: pass candidate
    results to BHY; choose between
    [`bhy`][factrix.multi_factor.bhy] /
    [`partial_conjunction`][factrix.multi_factor.partial_conjunction] /
    [`bhy_hierarchical`][factrix.multi_factor.bhy_hierarchical];
    mixed-cell batches.

    [Read the guide →](multi-factor.md)

-   __Data schema__

    ---

    New to the fixed-horizon input contract? Start here for the evaluation
    panel floor (`date`, `asset_id`, `factor`, `forward_return`), dtype
    semantics, and optional columns that activate extra metrics.

    [Read the schema →](data-schema.md)

</div>

## See also

<div class="grid cards" markdown>

-   __Timeseries-mode conventions__

    ---

    The `n_assets == 1` dispatch rules and SE conventions for the per-asset
    time-series stage.

    [reference/ts-mode-conventions →](../reference/ts-mode-conventions.md)

-   __Panel vs timeseries sample guard__

    ---

    Per-metric, per-axis sample floors on the effective sample, and the
    `InsufficientSampleError` recovery path (`.axis` / `.actual` / `.required`).

    [guides/panel-timeseries →](../guides/panel-timeseries.md)

</div>
