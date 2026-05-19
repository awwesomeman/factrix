---
title: factrix.evaluate
---

> **Input contract** — the panel must satisfy the four-column floor
> documented in [Panel schema](panel-schema.md).

::: factrix.evaluate

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Single-factor significance__

    ---

    One panel + one config → one result carrying `primary_p` and the
    cell-specific statistics.

-   __Batch screening with false discovery rate (FDR)__

    ---

    Loop `evaluate` over candidate signal columns and feed the
    resulting list of profiles to [`bhy`][factrix.multi_factor.bhy]
    for false-discovery-rate control. See
    [Batch screening](../guides/batch-screening.md).

-   __Cross-cell apples-to-apples__

    ---

    Compare information coefficient (IC) rank-ordering against
    Fama-MacBeth λ on the same panel, or individual-asset factors
    against broadcast macro factors. Return shape is identical across
    cells.

-   __TIMESERIES auto-routing__

    ---

    `Common × Continuous` with `N == 1` falls back to single-series
    ordinary least squares (OLS) with Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) SE, so single-asset macro factors flow
    through the same entry point without a parallel code path.

</div>

## Worked example — single-factor smoke test

!!! example "Synthetic panel → `evaluate` → read `primary_p` + `diagnose()`"

    Full runnable example complementing the doctest snippets in **Examples**
    above with realistic console output and a `diagnose()` dump.

    ```python
    # example pending v0.14.0 docs rewrite
    ```

## Config recipes — one per dispatch cell

```python
# example pending v0.14.0 docs rewrite
```

Per-cell required / optional columns and the PANEL ↔ TIMESERIES PanelMode
derivation are documented in the **Dispatch lore** admonition above.

## Next steps

<div class="grid cards" markdown>

-   __Batch screening guide__

    ---

    Wires `evaluate` into the multi-factor FDR pipeline: loop over
    candidates while preserving `identity` / `context`; choose between
    [`bhy`][factrix.multi_factor.bhy] /
    [`partial_conjunction`][factrix.multi_factor.partial_conjunction] /
    [`bhy_hierarchical`][factrix.multi_factor.bhy_hierarchical];
    mixed-cell batches; `primary_p` vs `stats` at the FDR stage.

    [Read the guide →](../guides/batch-screening.md)

-   __Panel schema__

    ---

    New to the input contract? Start here for the four-column floor
    (`date`, `asset_id`, `factor`, `forward_return`), dtype semantics,
    and optional columns that activate extra metrics.

    [Read the schema →](panel-schema.md)

</div>

## See also

<div class="grid cards" markdown>

-   __Timeseries-mode conventions__

    ---

    The `N == 1` auto-routing rules and SE conventions for single-series
    paths.

    [reference/ts-mode-conventions →](../reference/ts-mode-conventions.md)

-   __Panel vs timeseries sample guard__

    ---

    Sample-size floors and the `InsufficientSampleError` recovery path.

    [guides/panel-timeseries →](../guides/panel-timeseries.md)

</div>
