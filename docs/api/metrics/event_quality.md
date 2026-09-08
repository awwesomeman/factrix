---
title: factrix.metrics.event_quality
---

::: factrix.metrics.event_quality
    options:
      show_root_members_full_path: true
      members:
        - event_hit_rate
        - event_ic
        - profit_factor
        - event_skewness
        - signal_density

<hr>

!!! info "Event-study contracts"
    These metrics use the **sign-only** form
    $\text{signed\_car} = \text{abnormal\_return} \times \text{sign}(\text{factor})$
    — distinct from `caar`'s magnitude-weighted
    $\text{abnormal\_return} \times \text{factor}$. See the
    [abnormal-return table](../../reference/metric-applicability.md#abnormal-return-definition-per-metric)
    for the full per-metric contract and the
    [confounded-event note](../../reference/metric-applicability.md#confounded-event-handling)
    for the event-axis spacing and same-period dependence rules.

## Use cases

<div class="grid cards" markdown>

-   __Directional accuracy__

    ---

    Fraction of events whose `signed_car` is positive. The Cowan (1992)
    generalised sign null is estimated from non-event abnormal returns.
    `stat` is the hit count for the exact test and a z statistic when a
    material same-period clustering adjustment applies. A zero-variance
    boundary null keeps the hit rate but withholds the test fields.

-   __Magnitude → magnitude__

    ---

    Among triggered events, does the signal's `|factor|` co-move with
    the realised `signed_car`? Spearman rank correlation with Fisher-$z$
    inference, using the Fieller-Hartley-Pearson Spearman standard error
    $1.06/\sqrt{n-3}$ rather than the Pearson $1/\sqrt{n-3}$, and deflated
    for same-period clustering of the per-event rank score (see
    `EVENT_CLUSTERING_ADJUSTED`). Auto-skips on $\{0, \pm 1\}$ inputs where
    `|factor|` has no variance.

-   __Gain / loss ratio and shape__

    ---

    `profit_factor` reports $\sum\text{gains} / |\sum\text{losses}|$ as
    a descriptive gross ratio. If gains are positive and losses are zero,
    the ratio is unbounded (`value = inf`, `profit_factor_status =
    "unbounded_no_losses"`); if both are zero, the ratio is undefined
    (`value = NaN`). `event_skewness` reports the Fisher-
    corrected skewness of the `signed_car` distribution, descriptively:
    `p_value` and `stat` are always `None` because no pooled test of that
    third moment is calibrated here (see
    [Inference calibration and limitations](../../reference/inference-calibration.md#event-skewness-has-no-calibrated-test)).
    Useful for screening fat-right-tail vs symmetric event payoffs.

-   __Firing frequency__

    ---

    `signal_density` reports mean bars-per-event per asset (inverse
    frequency). Pair with `clustering_hhi` when independence
    assumptions matter — bars-per-event ignores temporal clustering.

</div>

## Choosing a function

| Goal                                                       | Function             |
|------------------------------------------------------------|----------------------|
| Directional-accuracy binomial test                         | `event_hit_rate`     |
| Magnitude-of-signal → magnitude-of-return rank correlation | `event_ic`           |
| Gross gain / loss ratio (descriptive only)                 | `profit_factor`      |
| Tail asymmetry of `signed_car` (descriptive only)          | `event_skewness`     |
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
        event_magnitude_jitter=0.5, post_event_drift_bps=40.0, rng=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    hit = event_hit_rate(panel)
    assert 0.0 <= hit.value <= 1.0
    assert hit.n_obs > 0
    assert hit.stat is not None and hit.p_value is not None

    sk = event_skewness(panel)
    assert sk.stat is None and sk.p_value is None

    pf = profit_factor(panel)
    assert pf.metadata["n_wins"] + pf.metadata["n_losses"] <= pf.n_obs
    ```

## See also

<div class="grid cards" markdown>

-   __`caar` / `bmp_z`__

    ---

    Mean-CAAR significance and BMP variance-robust $z$ on the same
    event sample.

    [api/metrics/caar →](caar.md)

-   __`clustering_hhi`__

    ---

    Event-date concentration index — read alongside `signal_density`
    when independence matters.

    [api/metrics/clustering →](clustering_hhi.md)

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
