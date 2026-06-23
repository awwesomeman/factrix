---
title: factrix.metrics.corrado_rank
---

::: factrix.metrics.corrado_rank
    options:
      show_root_members_full_path: true
      members:
        - corrado_rank

<hr>

!!! info "Event-study contracts"
    Corrado's primitive is
    $\text{signed\_rank} = \text{uniform\_rank}(\text{forward\_return}) \times \text{sign}(\text{factor})$
    — note that the direction adjustment is on the rank, **not** on
    the return itself. The
    [Event-study contracts table](../../reference/metric-applicability.md#abnormal-return-definition-per-metric)
    contrasts this with `caar` (magnitude-weighted) and `event_quality`
    (sign-only on the raw return). The
    [`estimation_window`](../../reference/metric-applicability.md#estimation_window)
    section specifies the per-asset baseline this test ranks against.

## Use cases

<div class="grid cards" markdown>

-   __Robustness screen against parametric CAAR / BMP__

    ---

    Rank-based two-sided $z$ on event-window abnormal returns. Robust
    to extreme returns, non-normal distributions, and cross-asset
    heteroscedasticity — flips the conclusion when parametric CAAR is
    being driven by a handful of outliers.

-   __Direction-adjusted nonparametric inference__

    ---

    Ranks are sign-flipped via `sign(factor)` before pooling, so the
    test handles two-sided signals (long / short events) directly —
    the Corrado-Zivney (1992) extension to the original one-directional
    test.

</div>

## Worked example — rank test alongside parametric CAAR

!!! example "corrado_rank on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.corrado_rank import corrado_rank
    from factrix.metrics.caar import compute_caar, caar
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        post_event_drift_bps=40.0, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    rank_out = corrado_rank(panel)
    print(rank_out.value, rank_out.stat, rank_out.metadata["p_value"])
    # 0.041  5.18  2.2e-07   (approximate)

    # Compare to parametric CAAR — divergence flags outlier-driven CAAR:
    caar_out = caar(compute_caar(panel), forward_periods=5)
    print(caar_out.stat, rank_out.stat)
    # 6.42   5.18
    ```

## See also

<div class="grid cards" markdown>

-   __`caar` / `bmp_z`__

    ---

    Parametric counterparts; read alongside Corrado for an
    outlier-robustness cross-check.

    [api/metrics/caar →](caar.md)

-   __`clustering_hhi`__

    ---

    The pooled-std denominator factrix uses diverges from the Corrado
    (1989) eq. (5) form when event-date clustering is present —
    inspect Herfindahl-Hirschman index (HHI) before reading $p$ at the size boundary.

    [api/metrics/clustering →](clustering_hhi.md)

-   __Statistical methods__

    ---

    Corrado nonparametric rank test specification and the deviation
    from eq. (5) factrix takes.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies, sample-size guards, and the per-asset
    full-sample ranking contract.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Sparse landing__

    ---

    Adjacent event-study metrics in the same cell.

    [api/metrics/individual-sparse →](individual-sparse.md)

</div>
