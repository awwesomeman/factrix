---
title: factrix.metrics.spanning
---

::: factrix.metrics.spanning
    options:
      show_root_members_full_path: true
      members:
        - spanning_alpha
        - greedy_forward_selection
        - SpanningResult
        - ForwardSelectionResult

<hr>

!!! warning "Inputs are factor-return series, not raw panels"
    `spanning` is a post-PANEL consumer: both callables operate on
    spread-series DataFrames (`date, spread`) produced by
    `compute_spread_series` (or any equivalent factor-mimicking-portfolio
    return series), not the raw `(date, asset_id, factor, forward_return)`
    panel consumed by the other metrics in this cell.

## Use cases

<div class="grid cards" markdown>

-   __Single-factor incremental alpha__

    ---

    `spanning_alpha` regresses the candidate spread series on a set of
    base-factor spread series; tests $H_0: \alpha = 0$ via OLS
    $t$-stat. Standard tool for "does this factor add anything beyond
    the existing model?" (Barillas-Shanken 2017).

-   __Mean-return test when no base factors__

    ---

    With `base_spreads=None` (or empty), `spanning_alpha` collapses to
    a plain mean-return $t$-test on the candidate's spread series —
    convenient when the question is "is there *any* alpha here" before
    pulling in controls.

-   __Greedy model construction over a pool__

    ---

    `greedy_forward_selection` iteratively adds the candidate with
    largest $|\alpha|$ above a $|t|$ threshold, then backward-eliminates
    any selected factor that loses significance. **Use as a
    model-construction helper only** — the returned $t$-stats are
    selection-conditioned and not valid for inference.

</div>

!!! warning "Stepwise selection inflates t-stats"
    `greedy_forward_selection` searches the candidate pool and picks
    by $|\alpha|$; the per-selected-factor $t$-stat is order-statistic
    inflated (typically 2-4x on pools of 10-100 candidates) and is
    *not* a draw from the $t$-null (White 2000; Harvey-Liu-Zhu 2016).
    The returned `t_stats_inference_invalid=True` flag encodes this
    contract. For post-selection significance, re-evaluate survivors
    on a held-out window, or use a Hansen (2005) SPA / White (2000)
    Reality Check on the pre-selection stage.

## Choosing a function

| Goal                                                                          | Function                    |
|-------------------------------------------------------------------------------|-----------------------------|
| Single-factor spanning regression — incremental alpha vs base factors         | `spanning_alpha`            |
| Greedy multi-factor selection over a candidate pool (model-construction only) | `greedy_forward_selection`  |

## Worked example — single-factor spanning then greedy selection

!!! example "compute_spread_series → spanning_alpha → greedy_forward_selection"

    ```python
    import factrix as fx
    from factrix.metrics.quantile import compute_spread_series
    from factrix.metrics.spanning import (
        spanning_alpha, greedy_forward_selection,
    )
    from factrix.preprocess import compute_forward_return

    # Build a spread series for each factor on the same panel dates.
    panels = {
        name: compute_forward_return(
            fx.datasets.make_cs_panel(
                n_assets=200, n_dates=500, ic_target=ic, seed=seed,
            ),
            forward_periods=5,
        )
        for name, ic, seed in [
            ("size",     0.05, 1),
            ("value",    0.06, 2),
            ("momentum", 0.08, 3),
            ("candidate",0.04, 4),
        ]
    }
    spreads = {
        name: compute_spread_series(p, forward_periods=5, n_groups=5)
        for name, p in panels.items()
    }

    # Single-factor: candidate vs the base set
    out = spanning_alpha(
        factor_spread = spreads["candidate"],
        base_spreads  = {k: spreads[k] for k in ("size", "value", "momentum")},
    )
    print(out.value, out.stat, out.metadata["p_value"], out.metadata["r_squared"])
    # 0.0011  1.83  0.068  0.21   (approximate)

    # Multi-factor: greedy build a parsimonious set
    pool = {k: spreads[k] for k in ("size", "value", "momentum", "candidate")}
    sel = greedy_forward_selection(
        factor_spreads          = pool,
        significance_threshold  = 2.0,
        max_factors             = 4,
        suppress_snooping_warning = True,  # acknowledged: construction-only
    )
    for s in sel.selected_factors:
        print(s.factor_name, s.alpha, s.t_stat)
    # momentum  0.0028  4.10
    # value     0.0019  2.51   (approximate; t_stats inflated, do not infer)
    ```

## See also

<div class="grid cards" markdown>

-   __`compute_spread_series` / `quantile_spread`__

    ---

    Produces the per-date spread series consumed here.

    [api/metrics/quantile →](quantile.md)

-   __`compute_factor_returns` (preprocess)__

    ---

    Any factor-return / spread time series with `(date, spread)` shape
    is a valid input; this is the upstream pipeline for the
    post-PANEL cell.

    [api/preprocess →](../preprocess.md)

-   __`slice_pairwise_test` / `slice_joint_test`__

    ---

    Cross-slice inference on spanning alphas (Wald $\chi^2$ + Holm /
    Romano-Wolf adjusted $p$).

    [api/slice-test →](../slice-test.md)

-   __Statistical methods__

    ---

    OLS $t$ on the alpha; when overlap is added, swap to NW HAC SE
    via the same kernel discipline used elsewhere in factrix.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the post-selection-inference contracts
    that govern `greedy_forward_selection`.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Continuous landing__

    ---

    Adjacent metrics in the same cell.

    [api/metrics/individual-continuous →](individual-continuous.md)

</div>
