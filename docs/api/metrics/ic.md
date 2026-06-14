---
title: factrix.metrics.ic
---

::: factrix.metrics.ic
    options:
      show_root_members_full_path: true
      members:
        - ic
        - ic_ir

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Compute per-date information coefficient (IC)__

    ---

    Build the per-date Spearman IC series (with `tie_ratio` diagnostics)
    from a long-format panel before any inferential test. Pre-step for
    `ic` / `ic_ir`.

-   __Mean-IC significance, non-overlapping__

    ---

    Test $H_0: \mathbb{E}[\mathrm{IC}] = 0$ on the every-`forward_periods`
    subsample to avoid the autocorrelation induced by overlapping forward
    returns. Default for the IC cell.

-   __Mean-IC significance, heteroskedasticity-and-autocorrelation-consistent (HAC)__

    ---

    Same null, but keep every overlapping observation and absorb the
    induced MA dependence through a Newey-West HAC standard error.
    Invoked via `ic(inference=fx.inference.NEWEY_WEST)`.

-   __IC stability (signed IR)__

    ---

    `mean(IC) / std(IC)` over the per-date series — a Sharpe-style
    descriptive statistic for signal time-stability. No inference attached.

</div>

## Choosing a function

| Goal                                                                | Function                                            |
|---------------------------------------------------------------------|-----------------------------------------------------|
| Per-date IC table for downstream inspection / slicing               | `compute_ic`                                        |
| Mean-IC significance (non-overlapping or Newey-West HAC)            | `ic`                                                |
| Time-stability ratio (no inference)                                 | `ic_ir`                                             |

All three are invoked indirectly via `evaluate(data, metrics={"ic": ic()})`
— they're documented here for callers who want the standalone numerical
output without the evaluation framing.

## Worked example — per-date IC then mean significance

!!! example "compute_ic → ic on a synthetic cross-sectional panel"

    ```python
    import factrix as fx
    from factrix.metrics.ic import compute_ic, ic
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=500, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    ic_df = compute_ic(panel)["factor"]
    print(ic_df.head())
    # ┌────────────┬───────────┬───────────┐
    # │ date       ┆ ic        ┆ tie_ratio │
    # ├────────────┼───────────┼───────────┤
    # │ 2024-01-01 ┆ 0.083     ┆ 0.000     │
    # │ 2024-01-02 ┆ 0.071     ┆ 0.000     │
    # │ ...        ┆ ...       ┆ ...       │
    # └────────────┴───────────┴───────────┘

    out = ic(ic_df, forward_periods=5, inference=fx.inference.NEWEY_WEST)
    print(out.value, out.stat, out.metadata["p_value"])
    # 0.0722  14.60  2.13e-40
    ```

!!! note "Cross-slice IC analysis"

    For per-slice IC summaries (regime / universe / sector / ...), use
    [`by_slice`](../by-slice.md) on an IC frame joined with slice labels.
    For inferential contrasts (pairwise Wald χ² + Holm / Romano-Wolf
    adjusted p), use [`slice_pairwise_test`](../slice-test.md). The
    metric-specific `regime_ic` callable and `by_regime` dispatcher were
    removed in v0.12.0; see the
    [Slice analysis guide](../../guides/slice-analysis.md).

## See also

<div class="grid cards" markdown>

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice IC summaries.

    [api/by-slice →](../by-slice.md)

-   __`slice_pairwise_test` / `slice_joint_test`__

    ---

    Cross-slice inference (Wald χ² + Holm / Romano-Wolf adjusted p).

    [api/slice-test →](../slice-test.md)

-   __Slice analysis guide__

    ---

    Slicing and cross-slice inference end-to-end.

    [guides/slice-analysis →](../../guides/slice-analysis.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guards that gate it.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Statistical methods__

    ---

    HAC SE, false discovery rate (FDR), robust-scale, unit-root disciplines that govern the inference.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
