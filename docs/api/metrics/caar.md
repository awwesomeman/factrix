---
title: factrix.metrics.caar
---

::: factrix.metrics.caar
    options:
      show_root_members_full_path: true
      members:
        - compute_caar
        - caar
        - bmp_z

<hr>

!!! info "Event-study contracts"
    `signed_car`, the `estimation_window` consumed by `bmp_z`, and
    factrix's confounded-event handling are documented in
    [Metric applicability § Event-study contracts](../../reference/metric-applicability.md#event-study-contracts).
    factrix computes **CAR** (sum of per-period abnormal returns), not
    BHAR; see the same section for the distinction.

!!! warning "`caar.value` is per period, not cumulative"
    Despite the name, `caar.value` is `CAR / h`, not the cumulative abnormal
    return of MacKinlay (1997) §4. `compute_forward_return` divides by
    `forward_periods` to put every horizon on a common per-period scale, and
    the event family reads that column, so the normalisation is inherited
    rather than undone here. With μ = 0.0008 per period over `h = 5`, the
    reported value is 0.001056 — the per-period average — not 5× that.

    Multiply by `forward_periods` to recover the cumulative quantity the name
    and the Brown-Warner / MacKinlay citations promise. Reading `caar.value`
    directly as a 5-period event CAR is off by 5×.

## Use cases

<div class="grid cards" markdown>

-   __Per-event-period CAAR series__

    ---

    The per-event-period weighted abnormal return series from a
    long-format panel. Consumed by `caar` for the significance test, and
    (where the magnitude-weighted form is wanted) available for
    per-slice summaries.

-   __Mean-CAAR significance, non-overlapping__

    ---

    Test $H_0: \mathbb{E}[\mathrm{CAAR}] = 0$ on the every-`overlap_periods`
    subsample of the per-event-period CAAR series to avoid the
    autocorrelation induced by overlapping forward returns. Default
    parametric test for the event-sparse cell.

-   __Event-induced variance, BMP $z$-test__

    ---

    Standardise each event's abnormal return by the asset's pre-event
    residual volatility before pooling. Robust to event-induced
    variance inflation that biases the ordinary CAAR $t$-test. The
    Kolari-Pynnönen same-period-correlation adjustment is on by default
    (identity when no two events share a period; 21.5% → 5.0% size at 4
    events per period on a null); `kolari_pynnonen_adjust=False` gives the
    unadjusted BMP for matching a source that reports it.

-   __Magnitude-weighted CAAR__

    ---

    With a continuous `factor` column, `compute_caar` returns the
    per-event regression-slope statistic in the Sefcik-Thompson (1986)
    lineage rather than the textbook equal-weighted MacKinlay CAAR —
    see the docstring for the input-contract table.

</div>

## Choosing a function

| Goal                                                         | Function       |
|--------------------------------------------------------------|----------------|
| Per-event-period CAAR table for downstream inspection / slicing | `compute_caar` |
| Mean-CAAR significance, deterministic non-overlap subsample   | `caar`         |
| Variance-robust event-induced significance (BMP standardised $z$) | `bmp_z`     |

## Event counts

`compute_caar` collapses same-period event rows before the `caar` test runs.
The event-study path therefore exposes these related counts:

| Field | Where to read it | Meaning |
|---|---|---|
| `n_events` | `compute_caar(...).select("date", "n_events")` | Raw event rows collapsed into each event period |
| `total_events` | `caar(...).metadata["total_events"]` | Sum of raw non-zero event rows behind the study |
| `n_event_periods` | `caar(...).metadata["n_event_periods"]` | Distinct event periods in the CAAR series |
| `n_event_periods_sampled` | `caar(...).metadata["n_event_periods_sampled"]` | Event dates kept by the non-overlap sampler used for the t-test |

`MetricResult.n_obs` equals `n_event_periods_sampled`, because that is the
sample entering the headline `p_value`. A large gap between `total_events` and
`n_event_periods` means events cluster on the same dates. A large gap between
`n_event_periods` and `n_event_periods_sampled` means the forward-return
windows overlap heavily, so the non-overlap sampler thins the effective test
sample.

For asset-allocation policy events, make sure the sparse factor sign encodes
the expected return direction, not just the raw event type. If `+1` means
"central-bank hike" but hikes are bearish for one asset group and bullish for
another, map the raw event into an asset-specific expected-return signal before
calling `compute_caar`, `event_hit_rate`, or `profit_factor`.

## Worked example — per-event-period CAAR then mean significance

!!! example "compute_caar → caar on a synthetic event panel"

    ```python
    import factrix as fx
    import polars as pl
    from factrix.metrics.caar import compute_caar, caar, bmp_z
    from factrix.preprocess import compute_forward_return

    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")

    raw   = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        post_event_drift_bps=40.0, rng=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    caar_df = compute_caar(panel)
    print(caar_df.head())
    # ┌────────────┬───────────┐
    # │ date       ┆ caar      │
    # ├────────────┼───────────┤
    # │ 2024-01-04 ┆  0.0041   │
    # │ 2024-01-11 ┆  0.0037   │
    # │ ...        ┆ ...       │
    # └────────────┴───────────┘

    out = caar(caar_df, overlap_periods=5)
    print(out.value, out.stat, out.p_value)
    # 0.0039  6.42  1.4e-09   (approximate)

    # Variance-robust alternative when same-period clustering is high:
    z_bmp = bmp_z(panel, estimation_window=60, overlap_periods=5,
                     kolari_pynnonen_adjust=True)
    ```

## See also

<div class="grid cards" markdown>

-   __`clustering_hhi`__

    ---

    Event-date HHI — when to switch on the Kolari-Pynnönen adjustment
    or read `caar`'s $t$ with caution.

    [api/metrics/clustering →](clustering_hhi.md)

-   __`by_slice`__

    ---

    Axis-agnostic slice dispatcher for per-slice CAAR summaries.

    [api/by-slice →](../by-slice.md)

-   __`slice_pairwise_test` / `slice_joint_test`__

    ---

    Cross-slice CAAR inference (Wald $\chi^2$ + Holm / Romano-Wolf
    adjusted $p$).

    [api/slice-test →](../slice-test.md)

-   __Statistical methods__

    ---

    CAAR cross-event $t$, BMP standardised AR $z$, Kolari-Pynnönen
    clustering adjustment.

    [reference/statistical-methods →](../../reference/statistical-methods.md)

-   __Metric applicability reference__

    ---

    When this metric applies, sample-size guards, and the event-study
    contracts that fix `signed_car`.

    [reference/metric-applicability →](../../reference/metric-applicability.md)

-   __Individual × Sparse landing__

    ---

    Adjacent event-study metrics in the same cell.

    [api/metrics/individual-sparse →](individual-sparse.md)

</div>
