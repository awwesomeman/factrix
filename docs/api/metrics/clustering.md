---
title: factrix.metrics.clustering
---

::: factrix.metrics.clustering
    options:
      show_root_members_full_path: true
      members:
        - clustering_diagnostic

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Gate the CAAR independence assumption__

    ---

    Read `value` (Herfindahl-Hirschman index (HHI) on the event-date histogram) and
    `metadata["effective_n_dates"]` $= 1 / \mathrm{HHI}$. High HHI →
    events concentrate in few dates → cross-event independence under
    `caar`'s $t$-test is violated and the statistic may be inflated.

-   __Trigger the Kolari-Pynnönen adjustment__

    ---

    When `hhi_normalized` is high ($\geq 0.3$ is the threshold the BMP
    docstring calls out), switch on
    `bmp_test(kolari_pynnonen_adjust=True)` to absorb same-date shock
    sharing in the $z$ statistic.

</div>

## Worked example — HHI on event dates

!!! example "clustering_diagnostic on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.clustering import clustering_diagnostic
    from factrix.metrics.caar import bmp_test

    panel = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        cluster_dates=True, seed=2024,
    )

    diag = clustering_diagnostic(panel)
    print(diag.value,
          diag.metadata["effective_n_dates"],
          diag.metadata["hhi_normalized"])
    # 0.041  24.4  0.36   (approximate)

    # hhi_normalized >= 0.3 -> reach for the K-P adjustment:
    z = bmp_test(panel, kolari_pynnonen_adjust=True)
    ```

## See also

<div class="grid cards" markdown>

-   __`caar` / `bmp_test`__

    ---

    The downstream tests whose independence assumption this metric
    gates. `bmp_test(kolari_pynnonen_adjust=True)` is the formal
    correction.

    [api/metrics/caar →](caar.md)

-   __`signal_density`__

    ---

    Inverse firing frequency — pair with `clustering_hhi` since
    bars-per-event ignores temporal concentration.

    [api/metrics/event_quality →](event_quality.md)

-   __Metric applicability reference__

    ---

    Confounded-event handling and within-asset event clustering notes.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Sparse landing__

    ---

    Adjacent event-study metrics in the same cell.

    [api/metrics/individual-sparse →](individual-sparse.md)

</div>
