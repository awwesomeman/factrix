---
title: factrix.metrics.event_quality
---

::: factrix.metrics.event_quality
    options:
      show_root_heading: true
      show_root_full_path: true
      show_root_toc_entry: true
      show_root_members_full_path: true
      heading_level: 1
      members:
        - event_hit_rate
        - event_ic
        - profit_factor
        - event_skewness
        - signal_density

<hr>

!!! info "Event-study contracts"
    These metrics use the **sign-only** form
    $\text{signed\_car} = \text{forward\_return} \times \text{sign}(\text{factor})$
    — distinct from `caar`'s magnitude-weighted
    $\text{forward\_return} \times \text{factor}$. See the
    [abnormal-return table](../../reference/metric-applicability.md#abnormal-return-definition-per-metric)
    for the full per-metric contract and the
    [confounded-event note](../../reference/metric-applicability.md#confounded-event-handling)
    for how the binomial / Spearman nulls behave under within-asset
    event clustering.

## Use cases

<div class="grid cards" markdown>

-   __Directional accuracy__

    ---

    Fraction of events whose `signed_car` is positive, with a two-sided
    binomial test against $H_0: p = 0.5$. Exact branch below the
    normal-approximation cutoff; $z$ branch above. Headline statistic
    for "is the sign right more often than chance".

-   __Magnitude → magnitude__

    ---

    Among triggered events, does the signal's `|factor|` co-move with
    the realised `signed_car`? Spearman rank correlation with Fisher-$z$
    inference. Auto-skips on $\{0, \pm 1\}$ inputs where `|factor|` has
    no variance.

-   __Gain / loss ratio and shape__

    ---

    `profit_factor` reports $\sum\text{gains} / |\sum\text{losses}|$ as
    a descriptive gross ratio; `event_skewness` reports the Fisher-
    corrected skewness of the `signed_car` distribution with a
    D'Agostino test when $N \geq 20$. Useful for screening
    fat-right-tail vs symmetric event payoffs.

-   __Firing frequency__

    ---

    `signal_density` reports mean bars-per-event per asset (inverse
    frequency). Pair with `clustering_diagnostic` when independence
    assumptions matter — bars-per-event ignores temporal clustering.

</div>

## Choosing a function

| Goal                                                       | Function             |
|------------------------------------------------------------|----------------------|
| Directional-accuracy binomial test                         | `event_hit_rate`     |
| Magnitude-of-signal → magnitude-of-return rank correlation | `event_ic`           |
| Gross gain / loss ratio (descriptive only)                 | `profit_factor`      |
| Tail asymmetry of `signed_car` with D'Agostino skew test   | `event_skewness`     |
| Inverse firing frequency (bars per event)                  | `signal_density`     |

## Worked example — directional accuracy + tail shape

!!! example "event_hit_rate + event_skewness on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.event_quality import (
        event_hit_rate, event_skewness, profit_factor,
    )
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        post_event_drift=0.004, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    hit = event_hit_rate(panel)
    print(hit.value, hit.stat, hit.metadata["p_value"])
    # 0.564  3.81  1.4e-04   (approximate)

    sk = event_skewness(panel)
    print(sk.value, sk.stat)
    # 0.42  4.10   (approximate; stat is D'Agostino z when N >= 20)

    pf = profit_factor(panel)
    print(pf.value, pf.metadata["n_wins"], pf.metadata["n_losses"])
    # 1.34  1131  864
    ```

## See also

<div class="grid cards" markdown>

-   __`caar` / `bmp_test`__

    ---

    Mean-CAAR significance and BMP variance-robust $z$ on the same
    event sample.

    [api/metrics/caar →](caar.md)

-   __`clustering_diagnostic`__

    ---

    Event-date concentration index — read alongside `signal_density`
    when independence matters.

    [api/metrics/clustering →](clustering.md)

-   __`by_slice`__

    ---

    Per-slice event-quality summaries (regime / universe / sector).

    [api/by-slice →](../by-slice.md)

-   __Statistical methods__

    ---

    Binomial test branches, Fisher-$z$ Spearman, D'Agostino skew test.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    Sample-size guards and `signed_car` contracts for the sign-only
    family.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Sparse landing__

    ---

    Adjacent event-study metrics in the same cell.

    [api/metrics/individual-sparse →](individual-sparse.md)

</div>
