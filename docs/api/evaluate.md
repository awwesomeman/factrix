---
title: factrix.evaluate
---

> **Input contract** — the panel must satisfy the four-column floor
> documented in [Panel schema](panel-schema.md).

::: factrix.evaluate
    options:
      show_root_heading: true
      show_root_full_path: true
      show_root_toc_entry: true
      heading_level: 1
      separate_signature: true
      show_signature_annotations: true

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Single-factor significance__

    ---

    One panel + one [`AnalysisConfig`][factrix.AnalysisConfig]
    → one [`FactorProfile`][factrix.FactorProfile] carrying
    `primary_p` and the cell-specific statistics.

-   __Batch screening with FDR__

    ---

    Loop `evaluate` over candidate signal columns and feed the
    resulting list of profiles to [`bhy`][factrix.multi_factor.bhy]
    for false-discovery-rate control. See
    [Batch screening](../guides/batch-screening.md).

-   __Cross-cell apples-to-apples__

    ---

    Swap the `AnalysisConfig` factory to compare IC rank-ordering
    against Fama-MacBeth λ on the same panel, or individual-asset
    factors against broadcast macro factors. Return shape is identical
    across cells.

-   __TIMESERIES auto-routing__

    ---

    `Common × Continuous` with `N == 1` falls back to single-series
    OLS with Newey-West HAC SE, so single-asset macro factors flow
    through the same entry point without a parallel code path.

</div>

## Worked example — single-factor smoke test

!!! example "Synthetic panel → `evaluate` → read `primary_p` + `diagnose()`"

    Full runnable example complementing the doctest snippets in **Examples**
    above with realistic console output and a `diagnose()` dump.

    ```python
    import factrix as fx
    from factrix.preprocess import compute_forward_return

    raw   = fx.datasets.make_cs_panel(
        n_assets=100, n_dates=500, ic_target=0.08, seed=2024,
    )
    panel = compute_forward_return(raw, forward_periods=5)

    cfg     = fx.AnalysisConfig.individual_continuous(
        metric=fx.Metric.IC, forward_periods=5,
    )
    profile = fx.evaluate(panel, cfg)

    print("primary_p =", round(profile.primary_p, 4))
    # → primary_p = 0.0

    print(profile.diagnose())
    # {'identity': {'factor_id': 'factor', 'forward_periods': 5},
    #  'context': {},
    #  'cell':     {'scope': 'individual', 'signal': 'continuous',
    #               'metric': 'ic', 'mode': 'panel'},
    #  'n_obs':    494, 'n_pairs': 49400, 'n_periods': 494, 'n_assets': 100,
    #  'primary_p':     2.13e-40,
    #  'primary_stat':  14.60,
    #  'primary_stat_name': 't_nw',
    #  'warnings': [], 'info_notes': [],
    #  'stats':    {'mean': 0.0722, 't_nw': 14.60, 'p_nw': 2.13e-40},
    #  'metadata': {'t_nw': {'nw_lags': 5}, 'p_nw': {'nw_lags': 5}}}
    ```

## Config recipes — one per dispatch cell

Minimum-viable `AnalysisConfig` for each of the four cells. The
`evaluate(panel, cfg)` call site is identical; only `cfg` changes.

=== "Individual × Continuous (IC)"

    Rank predictive ordering — Spearman IC + NW HAC.

    ```python
    cfg = fx.AnalysisConfig.individual_continuous(
        metric=fx.Metric.IC, forward_periods=5,
    )
    ```

=== "Individual × Continuous (FM)"

    Unit-of-exposure premium — Fama-MacBeth λ.

    ```python
    cfg = fx.AnalysisConfig.individual_continuous(
        metric=fx.Metric.FM, forward_periods=5,
    )
    ```

=== "Individual × Sparse"

    Event study with `factor ∈ {-1, 0, +1}` triggers. Attach a `price`
    column on the panel to also get `event_around_return` /
    `mfe_mae_summary` in the profile.

    ```python
    cfg = fx.AnalysisConfig.individual_sparse(forward_periods=5)
    ```

=== "Common × Continuous"

    Broadcast macro factor (e.g. VIX). With `N == 1` on the panel,
    `evaluate` auto-routes to single-series OLS with NW HAC SE
    (`profile.mode == "TIMESERIES"`).

    ```python
    cfg = fx.AnalysisConfig.common_continuous(forward_periods=5)
    ```

=== "Common × Sparse"

    Broadcast event dummy (FOMC, index rebalance).

    ```python
    cfg = fx.AnalysisConfig.common_sparse(forward_periods=5)
    ```

Per-cell required / optional columns and the PANEL ↔ TIMESERIES Mode
derivation are documented in the **Dispatch lore** admonition above.

## Next steps

<div class="grid cards" markdown>

-   __Batch screening guide__

    ---

    Wires `evaluate` into the multi-factor FDR pipeline: loop over
    candidates while preserving `identity` / `context`; choose between
    [`bhy`][factrix.multi_factor.bhy] /
    [`partial_conjunction`][factrix.multi_factor.partial_conjunction] /
    [`bhy_hierarchical`][factrix.multi_factor.bhy_hierarchical];
    mixed-cell batches; `primary_p` vs `stats` at the FDR stage.

    [Read the guide →](../guides/batch-screening.md)

-   __Panel schema__

    ---

    New to the input contract? Start here for the four-column floor
    (`date`, `asset_id`, `factor`, `forward_return`), dtype semantics,
    and optional columns that activate extra metrics.

    [Read the schema →](panel-schema.md)

</div>

## See also

<div class="grid cards" markdown>

-   __TIMESERIES-mode conventions__

    ---

    The `N == 1` auto-routing rules and SE conventions for single-series
    paths.

    [reference/ts-mode-conventions →](../reference/ts-mode-conventions.md)

-   __PANEL vs TIMESERIES sample guard__

    ---

    Sample-size floors and the `InsufficientSampleError` recovery path.

    [guides/panel-timeseries →](../guides/panel-timeseries.md)

-   __`run_metrics` — descriptive twin__

    ---

    Computes the same statistics but makes no FDR claim. Use when you
    want the numbers without the inference framing.

    [api/run-metrics →](run-metrics.md)

</div>
