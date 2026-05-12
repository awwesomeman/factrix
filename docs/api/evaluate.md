# evaluate

??? abstract "API reference"

    ::: factrix.evaluate

???+ example "Worked example — single-factor smoke test"

    Build a synthetic cross-sectional panel, attach forward returns,
    run `evaluate`, and read the result.

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

    For the multi-factor / FDR pipeline that consumes this `profile`,
    see [Batch screening](../guides/batch-screening.md). For
    non-default signal column names and other parameter variants, expand
    **API reference** above.
