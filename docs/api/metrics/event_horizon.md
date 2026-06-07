---
title: factrix.metrics.event_horizon
---

::: factrix.metrics.event_horizon
    options:
      show_root_members_full_path: true
      members:
        - event_around_return

<hr>

!!! info "Offset conventions"
    Defaults: `offsets = [-6, -3, -1, 1, 6, 12, 24]`. Offset `0` is the
    event date itself and is excluded from the defaults; user-supplied
    `offsets` lists are honoured verbatim.

    | $k$ | Anchor | Formula | Sign-adjusted |
    |---|---|---|---|
    | $k > 0$ (post-event) | Cumulative from $t+1$ entry | `price[t+1+k] / price[t+1] − 1` | Yes — multiplied by `sign(factor)`. The reading is signal *quality*. |
    | $k < 0$ (pre-event) | Single bar at offset | `price[t+k] / price[t+k−1] − 1` | **No** — the reading is *leakage*, where the bar's directional response matters independent of the eventual signal sign. |
    | $k = 0$ (corner) | Single bar at event | `price[t] / price[t−1] − 1` | No — falls into the pre-event branch. Pass with care; the event-day bar is usually contaminated by the announcement itself. |

    The pre/post asymmetry is intentional. Mixing the two conventions
    on a single chart (post-event cumulative + pre-event single-bar) is
    the default factrix presentation; downstream consumers should not
    re-cumulate the pre-event leg.

!!! warning "Serial correlation across offsets"
    The binomial null at each offset assumes per-event independence at
    that offset. **Adjacent post-event offsets are serially correlated
    within the same event** — $k = 6$ and $k = 12$ share the $t+1$
    entry price and overlap on bars $[t+2, t+7]$. The reported
    per-offset $p$-values therefore have understated variance under
    the joint null across offsets; treat the curve as descriptive and
    read the binomial $p$ one offset at a time. See also the
    [confounded-event note](../../reference/metric-applicability.md#confounded-event-handling)
    on within-asset event clustering, which compounds the same issue.

## Use cases

<div class="grid cards" markdown>

-   __Right-size the event window__

    ---

    Read the post-event mean curve over $k = 1 \ldots K$ to locate the
    horizon where signed drift peaks before reverting. Drives the
    choice of `EventConfig.event_window_post` for downstream MFE/MAE
    and CAAR work.

-   __Pre-event leakage check__

    ---

    Inspect $k < 0$ mean returns: a healthy signal has flat pre-event
    means. The headline `event_around_return.value` is
    $\mathrm{mean}_{k < 0} |\mathrm{mean}_k|$, summarising the leakage
    score in a single number.

</div>

## Choosing a function

| Goal                                                          | Function                |
|---------------------------------------------------------------|-------------------------|
| Per-event, per-offset raw return table for custom plots / cuts | `compute_event_returns` |
| Per-offset summary (mean / median / quartiles / hit-rate) with pre-event leakage headline | `event_around_return`   |

## Worked example — leakage score + per-offset curve

!!! example "compute_event_returns → event_around_return on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.event_horizon import (
        compute_event_returns, event_around_return,
    )

    panel = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        post_event_drift=0.004, with_price=True, seed=2024,
    )

    rets = compute_event_returns(panel, offsets=[-6, -3, -1, 1, 6, 12, 24])
    print(rets.head())
    # ┌────────┬────────────┬──────────┬────────────────┐
    # │ offset ┆ date       ┆ asset_id ┆ signed_return  │
    # ├────────┼────────────┼──────────┼────────────────┤
    # │   -6   ┆ 2024-01-04 ┆ A0001    ┆  0.0012        │
    # │    1   ┆ 2024-01-04 ┆ A0001    ┆  0.0041        │
    # │  ...   ┆ ...        ┆ ...      ┆ ...            │
    # └────────┴────────────┴──────────┴────────────────┘

    out = event_around_return(panel)
    print(out.value)                              # mean |pre-event mean|
    print(out.metadata["per_offset"][6]["mean"])  # post-event signed mean at k=6
    # 0.0007   0.0094   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`mfe_mae`__

    ---

    Per-event excursion analysis on the same post-event window — peak
    favourable / adverse move and bars-to-peak.

    [api/metrics/mfe_mae →](mfe_mae.md)

-   __`caar` / `bmp_test`__

    ---

    Inferential CAAR / BMP tests on the chosen `forward_periods`
    horizon.

    [api/metrics/caar →](caar.md)

-   __`clustering_hhi`__

    ---

    Event-date Herfindahl-Hirschman index (HHI) — the serial-correlation caveat above compounds
    with same-date clustering.

    [api/metrics/clustering →](clustering_hhi.md)

-   __Metric applicability reference__

    ---

    Event-window / estimation-window contracts and confounded-event
    handling.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Sparse landing__

    ---

    Adjacent event-study metrics in the same cell.

    [api/metrics/individual-sparse →](individual-sparse.md)

</div>
