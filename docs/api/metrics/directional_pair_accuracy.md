---
title: factrix.metrics.directional_pair_accuracy
---

::: factrix.metrics.directional_pair_accuracy
    options:
      show_root_members_full_path: true
      members:
        - directional_pair_accuracy

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Small-N rank-order diagnostic__

    ---

    `directional_pair_accuracy` asks whether the higher-scored asset
    outperformed the lower-scored asset on the same date. It is useful when
    the universe has roughly 5-20 names and quantile buckets or fixed-K
    spreads are too lumpy to describe the ordering signal cleanly.

-   __Ties and nulls are explicit__

    ---

    Factor ties and return ties are excluded from the comparable-pair
    denominator and counted in metadata. Null factor/return rows are dropped
    before pair construction. Read `metadata["n_pairs"]`,
    `factor_tie_pairs`, `return_tie_pairs`, and `dropped_rows_null` before
    treating the headline accuracy as stable.

-   __Descriptive by design__

    ---

    The metric returns no `p_value`: same-date asset pairs share shocks, so a
    naive binomial test over all pairs would overstate precision. Use it as a
    targeted allocation diagnostic alongside the first-pass IC / FM evidence.

</div>

## Choosing a function

| Goal | Function |
|---|---|
| Rank relation with formal inference on per-date IC series | `ic` |
| Fixed-count long-short spread in a small universe | `k_spread` |
| Sign prediction against realised direction | `directional_hit_rate` |
| Within-date pairwise ordering accuracy | `directional_pair_accuracy` |

## Worked example

!!! example "Pairwise ordering on a small allocation panel"

    ```python
    import factrix as fx
    from factrix.metrics.directional_pair_accuracy import directional_pair_accuracy
    from factrix.preprocess import compute_forward_return

    raw = fx.datasets.make_cs_panel(n_assets=12, n_dates=160, seed=2024)
    panel = compute_forward_return(raw, forward_periods=5)

    out = directional_pair_accuracy(panel, forward_periods=5)
    print(out.value, out.p_value, out.metadata["n_pairs"])
    # 0.56  None  198   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __Choosing a metric__

    ---

    Research-question-first guidance for first-pass metrics vs diagnostics.

    [guides/choosing-metric](../../guides/choosing-metric.md)

-   __Allocation-style experiments__

    ---

    How to use small-N diagnostics without treating them as portfolio
    optimizers or backtests.

    [guides/allocation-experiment](../../guides/allocation-experiment.md)

-   __Metric applicability reference__

    ---

    The pairs-axis sample floors behind this diagnostic.

    [reference/metric-applicability](../../reference/metric-applicability.md)

</div>
