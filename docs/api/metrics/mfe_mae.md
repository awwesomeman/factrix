---
title: factrix.metrics.mfe_mae
---

::: factrix.metrics.mfe_mae
    options:
      show_root_members_full_path: true
      members:
        - mfe_mae

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Peak favourable vs adverse excursion__

    ---

    For each event, find the peak gain (MFE) and peak loss (MAE) over
    the post-event window, plus bars-to-peak. Descriptive of the
    *shape* of the post-event price path, not just its endpoint.

-   __Risk-adjusted favourability__

    ---

    Headline ratio $\mathrm{MFE}_{p50} / |\mathrm{MAE}_{p75}|$ pairs
    the median favourable excursion against the 75th percentile
    adverse excursion — captures whether typical upside exceeds worst-
    quartile downside.

-   __Cross-horizon / cross-regime comparison__

    ---

    Z-scored variants `mfe_z` / `mae_z` (divided by
    $\hat\sigma \sqrt{W}$) absorb the $\sqrt{W \cdot \sigma^2}$
    horizon scaling of order statistics — the apples-to-apples
    quantity for comparing event setups across windows or volatility
    regimes.

</div>

## Choosing a function

| Goal                                                                | Function           |
|---------------------------------------------------------------------|--------------------|
| Per-event MFE / MAE / bars-to-peak table for downstream cuts        | `compute_mfe_mae`  |
| Aggregate distribution summary (quantiles, ratio, z-scored siblings) | `mfe_mae` |

## Worked example — per-event excursion then summary

!!! example "compute_mfe_mae → mfe_mae on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.mfe_mae import compute_mfe_mae, mfe_mae

    panel = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        post_event_drift_bps=40.0, seed=2024,
    )

    per_event = compute_mfe_mae(panel, window=20, estimation_window=60)
    print(per_event.head())
    # ┌────────────┬──────────┬────────┬─────────┬────────┬────────┐
    # │ date       ┆ asset_id ┆  mfe   ┆  mae    ┆ mfe_z  ┆ mae_z  │
    # ├────────────┼──────────┼────────┼─────────┼────────┼────────┤
    # │ 2024-01-04 ┆ A0001    ┆ 0.031  ┆ -0.018  ┆  0.74  ┆ -0.43  │
    # │  ...       ┆ ...      ┆  ...   ┆  ...    ┆  ...   ┆  ...   │
    # └────────────┴──────────┴────────┴─────────┴────────┴────────┘

    out = mfe_mae(per_event)
    print(out.value,
          out.metadata["mfe_p50"], out.metadata["mae_p75"],
          out.metadata.get("mfe_mae_ratio_z"))
    # 1.27  0.024  -0.019  1.31   (approximate)
    ```

## See also

<div class="grid cards" markdown>

-   __`event_around_return`__

    ---

    Per-offset mean curve on the same post-event window — read for
    drift shape; MFE/MAE read for excursion magnitude.

    [api/metrics/event_horizon →](event_horizon.md)

-   __`caar` / `bmp_z`__

    ---

    Inferential CAAR / BMP $z$ on the endpoint of the same event
    window.

    [api/metrics/caar →](caar.md)

-   __`by_slice`__

    ---

    Per-slice excursion summaries (regime / universe / sector).

    [api/by-slice →](../by-slice.md)

-   __Metric applicability reference__

    ---

    Event-window / estimation-window contracts and price-data
    requirements.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Sparse landing__

    ---

    Adjacent event-study metrics in the same cell.

    [api/metrics/individual-sparse →](individual-sparse.md)

</div>
