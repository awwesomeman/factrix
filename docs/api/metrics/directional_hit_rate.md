---
title: factrix.metrics.directional_hit_rate
---

::: factrix.metrics.directional_hit_rate
    options:
      show_root_members_full_path: true
      members:
        - directional_hit_rate

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Small-N robust sibling of `hit_rate`__

    ---

    `directional_hit_rate` is the `(*, DENSE, *)` directional counterpart
    of `hit_rate`. Where `hit_rate` runs a naive binomial against
    $p = 0.5$ on a single `(date, value)` series, this metric consumes
    the raw panel `(date, asset_id, factor, forward_return)` and tests
    whether `sign(factor)` predicts `sign(forward_return)`. The headline
    `value` is itself a hit rate — the fraction of correctly-signed
    calls.

-   __Pesaran-Timmermann conditions on the marginals__

    ---

    The Pesaran-Timmermann (1992) statistic compares the realised hit
    rate to the rate *expected under directional independence*, derived
    from the marginal up/down frequencies of both the prediction and the
    realisation. A factor that is simply long a persistently-rising
    market is therefore not credited with skill — the correct framing for
    small, sign-imbalanced samples (the `N < 30` allocation regime).

-   __One-sided, non-overlap sampled__

    ---

    The test is one-sided: a large positive $S_n$ signals genuine
    directional skill, so a sign-inverted predictor scores poorly (flip
    its sign before testing). Observations are sub-sampled at stride
    `forward_periods` so overlapping forward-return windows do not
    inflate the statistic; degenerate samples (one-signed predictions or
    realisations) short-circuit to `NaN`.

</div>

## Choosing a function

| Goal                                                                  | Function               |
|-----------------------------------------------------------------------|------------------------|
| Sign-significance of a `(date, value)` series vs $p = 0.5$            | `hit_rate`             |
| Directional skill of a factor in small, sign-imbalanced samples      | `directional_hit_rate` |

## Worked example — directional skill on a panel

!!! example "factor → directional_hit_rate"

    ```python
    import factrix as fx
    from factrix.metrics.directional_hit_rate import directional_hit_rate
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=25, n_dates=120, ic_target=0.06, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    out = directional_hit_rate(panel, forward_periods=5)
    print(out.value, out.stat, out.metadata["p_value"], out.metadata["method"])
    # 0.57  2.74  0.0031  Pesaran-Timmermann (1992)   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`hit_rate`__

    ---

    The series-diagnostic sibling: binomial significance of a single
    `(date, value)` series against $p = 0.5$.

    [api/metrics/hit_rate →](hit_rate.md)

-   __Metric applicability reference__

    ---

    When this metric applies and the sample-size guard that gates it
    (`MIN_IC_ASSETS` floor on the non-overlapping pooled signs).

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Statistical methods__

    ---

    Non-overlap stride discipline and the Hansen-Hodrick autocorrelation
    floor that motivates it.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

</div>
