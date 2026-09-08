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
    | $k < 0$ (pre-event) | Single bar at offset | `price[t+k] / price[t+k−1] − 1` | Yes — multiplied by `sign(factor)`, so directional leakage does not cancel across long and short events. |
    | $k = 0$ (corner) | Single bar at event | `price[t] / price[t−1] − 1` | Yes — multiplied by `sign(factor)`. Pass with care; the event-day bar is usually contaminated by the announcement itself. |

    The pre/post asymmetry is intentional. Mixing the two conventions
    on a single chart (post-event cumulative + pre-event single-bar) is
    the default factrix presentation; downstream consumers should not
    re-cumulate the pre-event leg.

!!! info "Three quantities, one curve"
    `per_offset[k]` reports an **abnormal** return, never a raw one, and the
    metric's headline `value` is a third quantity again. Keeping them apart
    is the difference between reading event alpha and reading the drift the
    asset had anyway.

    | Quantity | What it is | Where it appears |
    |---|---|---|
    | Raw signed mean return | The formula in the table above, sign-adjusted, with nothing subtracted | Not published; recover its mean as `mean + benchmark` |
    | Abnormal (excess) return | Each event's raw return minus its own asset's signed benchmark | `per_offset[k]["mean"]`, and every dispersion key beside it |
    | Leakage score | Mean of the absolute abnormal returns over the *negative* offsets | the metric's `value` |

    Each event uses its own asset's unconditional return **over the same
    horizon as the offset it is subtracted from**: that asset's single-period
    mean for offsets at or below zero, and `(1 + asset_mean)**k - 1` for
    offset $k > 0$. The benchmark is also multiplied by `sign(factor)` before
    subtraction. Pooling asset drifts first would leave an event-composition
    residue at one-bar offsets and a Jensen gap after compounding; both would
    read as event alpha on a panel that merely trends.

    Because the summary contains many events, `per_offset[k]["benchmark"]`
    is the **event-weighted mean signed benchmark actually subtracted**. It is
    the exact amount needed to recover the raw signed mean as
    `mean + benchmark`; no one scalar can undo the event-specific subtraction
    from the median or quantiles. `benchmark_weighting` names this contract.
    `baseline_bar_return_by_asset` publishes the underlying single-period
    mapping. `baseline_bar_return` remains a compact equal-asset-weighted panel
    diagnostic, and `n_assets_in_baseline` reports how many assets entered it;
    the pooled diagnostic is not used to price events.

!!! warning "Descriptive only — no p-value is produced"
    `event_around_return` runs no hypothesis test: `p_value` is `None`, and
    `per_offset[k]` carries
    `{benchmark, mean, median, p25, p75, hit_rate, n}` — the
    `hit_rate` is a raw fraction of positive signed returns, not a binomial
    test, and no offset carries a `p`.

    That is deliberate rather than an omission, because the reported curve
    breaks the independence any such test would need, twice over. **Adjacent
    post-event offsets are serially correlated within the same event** —
    $k = 6$ and $k = 12$ share the $t+1$ entry price and overlap on bars
    $[t+2, t+7]$ — so offsets are not separate trials. And **events on one
    asset closer than the horizon share bars with each other**; unlike the
    event significance tests (`caar`, `bmp_z`, `corrado_rank`,
    `event_hit_rate`), this metric applies no non-overlap sampling, because
    thinning the sample would only cost resolution on a curve nobody tests.
    Read the curve as a shape, and take significance from those metrics
    instead. See also the
    [confounded-event note](../../reference/metric-applicability.md#confounded-event-handling)
    on within-asset event clustering.

!!! warning "Invalid prices withdraw the curve"
    `event_around_return` needs finite per-asset unconditional bar-return baselines.
    Any observed non-finite or non-positive `price` invalidates that baseline,
    so the metric returns `value=NaN`, an empty `per_offset` mapping, and
    `WarningCode.METRIC_UNAVAILABLE` with `reason="invalid_price_data"`.
    A baseline that no valid price could form at all takes the same branch —
    empty `per_offset`, same warning — under `reason="no_finite_baseline_returns"`
    (no asset yielded a finite single-period return, e.g. a panel whose prices
    never fall on two adjacent periods); `n_invalid_prices` is `0` there, which
    is what separates the two.
    Raw paths may already have been formed before the baseline check, so
    `n_events` / `n_obs` still report the distinct events that computation
    actually used. Their per-offset computed/censored counts are deliberately
    withheld because the curve they would describe was discarded. This is the
    explicit short-circuit exception to the censor-audit contract below.
    Missing (`null`) observations remain allowed on ragged panels. If an asset
    can form an event path but has no adjacent-period returns from which to
    estimate its own baseline, its affected event-offset rows are censored as
    `missing_asset_baseline` rather than priced from another asset's drift. If
    no event row has an asset baseline, the metric short-circuits with
    `reason="no_event_asset_baselines"`. Fix invalid observed prices rather
    than interpreting a contaminated finite hit rate.

## Complete price paths and censoring audit

An evaluation panel produced by `compute_forward_return` has a shorter tail
than its raw price input. Pass that raw panel as `price_data` so an event that
remains eligible can still reach every available offset:

```python
import factrix as fx
from factrix.metrics.event_horizon import event_around_return

raw = fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=7)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
out = fx.evaluate(
    panel,
    price_data=raw,
    metrics={"path": event_around_return(offsets=[-1, 6, 12, 24])},
    factor_cols=["factor"],
    strict=False,
)["factor"].metrics["path"]
```

`offsets=` are counts of periods on the grid the walk reads: the evaluation
panel's own distinct dates without `price_data`, the price panel's grid with
it. Passing `price_data` therefore re-bases every offset, and an offset chosen
for a coarse evaluation grid reaches a different distance on the raw grid. See
[Offsets and windows are counted on the grid that supplies them](../evaluate.md#offsets-and-windows-are-counted-on-the-grid-that-supplies-them);
`evaluate_horizons` forwards its raw panel automatically, so its existing
`event_around_return` results change accordingly.

Each `per_offset[k]` reports `eligible`, `computed`, `censored`, and a
`censor_reasons` count mapping. Reasons distinguish an out-of-grid offset,
missing entry/exit price, invalid denominator, missing asset, and missing price
column. `n` remains the computed count used for the summary statistic.

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
        post_event_drift_bps=40.0, rng=2024,
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

-   __`caar` / `bmp_z`__

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
