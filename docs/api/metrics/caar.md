---
title: factrix.metrics.caar
---

::: factrix.metrics.caar
    options:
      show_root_members_full_path: true
      members:
        - caar
        - bmp_z

<hr>

!!! info "Event-study contracts"
    `signed_car`, the `estimation_window` consumed by `bmp_z`, and
    factrix's confounded-event handling are documented in
    [Metric applicability § Event-study contracts](../../reference/metric-applicability.md#event-study-contracts).
    factrix computes **CAR** (sum of per-period abnormal returns), not
    BHAR; see the same section for the distinction.

## Use cases

<div class="grid cards" markdown>

-   __Per-event-date CAAR series__

    ---

    The per-event-date weighted abnormal return series from a
    long-format panel. Consumed by `caar` for the significance test, and
    (where the magnitude-weighted form is wanted) available for
    per-slice summaries.

-   __Mean-CAAR significance, non-overlapping__

    ---

    Test $H_0: \mathbb{E}[\mathrm{CAAR}] = 0$ on the every-`forward_periods`
    subsample of the per-event-date CAAR series to avoid the
    autocorrelation induced by overlapping forward returns. Default
    parametric test for the event-sparse cell.

-   __Event-induced variance, BMP $z$-test__

    ---

    Standardise each event's abnormal return by the asset's pre-event
    residual volatility before pooling. Robust to event-induced
    variance inflation that biases the ordinary CAAR $t$-test; pair
    with `kolari_pynnonen_adjust=True` when the event-date Herfindahl-Hirschman index (HHI) flags
    same-date shock sharing.

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
| Per-event-date CAAR table for downstream inspection / slicing | `compute_caar` |
| Mean-CAAR significance, deterministic non-overlap subsample   | `caar`         |
| Variance-robust event-induced significance (BMP standardised $z$) | `bmp_z`     |

## Worked example — per-event-date CAAR then mean significance

!!! example "compute_caar → caar on a synthetic event panel"

    ```python
    import factrix as fx
    from factrix.metrics.caar import compute_caar, caar, bmp_z
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_event_panel(
        n_assets=200, n_dates=500, event_rate=0.02,
        post_event_drift=0.004, seed=2024,
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

    out = caar(caar_df, forward_periods=5)
    print(out.value, out.stat, out.metadata["p_value"])
    # 0.0039  6.42  1.4e-09   (approximate)

    # Variance-robust alternative when same-date clustering is high:
    z_bmp = bmp_z(panel, estimation_window=60, forward_periods=5,
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
