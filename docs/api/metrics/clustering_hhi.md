---
title: factrix.metrics.clustering_hhi
---

::: factrix.metrics.clustering_hhi
    options:
      show_root_members_full_path: true
      members:
        - clustering_hhi

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Gate the CAAR independence assumption__

    ---

    Read `value` (Herfindahl-Hirschman index (HHI) on the event-period histogram) and
    `metadata["effective_n_periods"]` $= 1 / \mathrm{HHI}$. High HHI →
    the events that exist are concentrated on few of the dates that have
    events. **This is one axis of three, and not the one the
    Kolari-Pynnönen adjustment acts on**: HHI is invariant to how many assets
    fire per date, so 20 assets all firing on the same 40 dates scores
    exactly what one asset firing once on each of those dates scores.

-   __See the axes HHI misses__

    ---

    `events_per_period_mean` is the cross-sectional axis — the Kish
    effective cluster size, and the quantity `bmp_z` /
    `event_hit_rate` / `event_ic` feed to their Kolari-Pynnönen deflator.
    Above 1, events share periods and those corrections bite (they are on by
    default; there is no threshold to act on).
    `share_events_in_bursts` is the temporal axis — the share of events whose
    same-asset predecessor sits within `cluster_window` periods, the regime
    the event tests' non-overlap stride removes.

</div>

## Worked example — HHI on event periods

!!! example "clustering_hhi on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.clustering_hhi import clustering_hhi
    from factrix.metrics.caar import bmp_z
    from factrix.preprocess import compute_forward_return

    # At this event rate the generator's calendar already puts several
    # events on the same period, which is the regime the diagnostic reads.
    raw   = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    diag = clustering_hhi(panel)
    print(diag.value,
          diag.metadata["effective_n_periods"],
          diag.metadata["hhi_normalized"],
          diag.metadata["events_per_period_mean"],
          diag.metadata["share_events_in_bursts"])

    # events_per_period_mean > 1 is what the K-P adjustment acts on; it is on
    # by default and is the identity when events do not share periods.
    z = bmp_z(panel, kolari_pynnonen_adjust=True)
    ```

## See also

<div class="grid cards" markdown>

-   __`caar` / `bmp_z`__

    ---

    The downstream tests whose independence assumption this metric
    gates. `bmp_z(kolari_pynnonen_adjust=True)` is the formal
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
