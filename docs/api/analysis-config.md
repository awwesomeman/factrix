---
title: factrix.AnalysisConfig
---

::: factrix.AnalysisConfig
    options:
      show_root_heading: true
      show_root_full_path: true
      show_root_toc_entry: true
      show_root_members_full_path: true
      heading_level: 1
      members:
        - individual_continuous
        - individual_sparse
        - common_continuous
        - common_sparse
        - to_dict
        - from_dict

<hr>

## Use cases

<div class="grid cards" markdown>

-   __Selecting a dispatch cell__

    ---

    Pick the factory whose `(scope, signal, metric)` tuple matches your
    factor — see **Choosing a factory** below. Construct → pass to
    [`evaluate`][factrix.evaluate].

-   __Switching inference within a cell__

    ---

    Same factory, different `estimator=` (e.g. `HansenHodrick()`
    instead of the default `NeweyWest()`) to swap the HAC kernel.

-   __Persisting an analysis spec__

    ---

    Use `to_dict` / `from_dict` to cache the config alongside results,
    or to keep a backtest reproducible after a code change.

-   __Failing fast on illegal cells__

    ---

    Every construction path runs `__post_init__`. An illegal
    `(scope, signal, metric)` triple raises
    [`IncompatibleAxisError`](errors.md) at
    *construction* time, not at `evaluate` time.

</div>

## Choosing a factory

| Your factor                                       | Factory                                   | Resulting cell                            |
|---------------------------------------------------|-------------------------------------------|-------------------------------------------|
| Per-asset real-valued signal, want rank IC        | [`individual_continuous(metric=Metric.IC)`](metrics/individual-continuous.md) | `(INDIVIDUAL, CONTINUOUS, IC)`            |
| Per-asset real-valued signal, want FM λ premium   | [`individual_continuous(metric=Metric.FM)`](metrics/individual-continuous.md) | `(INDIVIDUAL, CONTINUOUS, FM)`            |
| Per-asset `{-1, 0, +1}` event trigger             | [`individual_sparse()`](metrics/individual-sparse.md)                     | `(INDIVIDUAL, SPARSE, None)`              |
| Broadcast real-valued factor (e.g. VIX)           | [`common_continuous()`](metrics/common-continuous.md)                     | `(COMMON, CONTINUOUS, None)`              |
| Broadcast event dummy (FOMC, index rebalance)     | [`common_sparse()`](metrics/individual-sparse.md) | `(COMMON, SPARSE, None)`                  |

Direct construction (`AnalysisConfig(scope=..., signal=..., metric=...)`)
also works and runs the same validation, but the factories are the
documented public surface. Bypassing them buys nothing.

## Worked example — construct, evaluate, inspect the cell

!!! example "Build a config, evaluate, read the cell tuple back from the profile"

    ```python
    import factrix as fx

    cfg = fx.AnalysisConfig.individual_continuous(
        metric=fx.Metric.IC, forward_periods=5,
    )

    print(cfg)
    # AnalysisConfig(scope=<FactorScope.INDIVIDUAL>, signal=<Signal.CONTINUOUS>,
    #                metric=<Metric.IC>, forward_periods=5, ...)

    profile = fx.evaluate(panel, cfg)
    print(profile.diagnose()["cell"])
    # {'scope': 'individual', 'signal': 'continuous', 'metric': 'ic', 'mode': 'panel'}
    ```

    Illegal cells fail at construction:

    ```python
    # (COMMON, SPARSE, IC) is not a registered cell — Metric.IC is only legal
    # for INDIVIDUAL × CONTINUOUS.
    fx.AnalysisConfig(
        scope=fx.FactorScope.COMMON,
        signal=fx.Signal.SPARSE,
        metric=fx.Metric.IC,
    )
    # IncompatibleAxisError: (common, sparse, ic) is not a legal analysis cell.
    #   Use one of the four factory methods:
    #     AnalysisConfig.individual_continuous(metric=Metric.IC|Metric.FM)
    #     AnalysisConfig.individual_sparse()
    #     AnalysisConfig.common_continuous()
    #     AnalysisConfig.common_sparse()
    ```

## Persistence — round-trip via dict

`to_dict` / `from_dict` exist for caching configs alongside results and
keeping a backtest replayable across code revisions.

!!! example "Serialise to JSON and rehydrate via `from_dict`"

    ```python
    import json

    cfg = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.FM)

    payload = cfg.to_dict()
    # {'scope': 'individual', 'signal': 'continuous', 'metric': 'fm',
    #  'forward_periods': 5, 'estimator': 'newey_west', 'moment_estimator': None}

    with open("cfg.json", "w") as f:
        json.dump(payload, f)

    # ... later, possibly a different process / commit ...

    with open("cfg.json") as f:
        restored = fx.AnalysisConfig.from_dict(json.load(f))
    assert restored == cfg
    ```

`from_dict` runs the same `__post_init__` validation as the factories, so
a tampered or stale payload raises
[`IncompatibleAxisError`](errors.md) or
[`UnknownEstimatorError`](errors.md) up front rather
than failing silently later.

## See also

<div class="grid cards" markdown>

-   __`evaluate` — entry-point function__

    ---

    The function that consumes a validated `AnalysisConfig` and returns
    a `FactorProfile`.

    [api/evaluate →](evaluate.md)

-   __Concepts — three-axis taxonomy__

    ---

    `Scope × Signal × Metric` axes and the legal-cell lattice.

    [getting-started/concepts →](../getting-started/concepts.md)

-   __Statistical methods__

    ---

    Per-cell procedure rationale and SE conventions.

    [reference/statistical-methods →](../reference/statistical-methods.md)

-   __`list_estimators`__

    ---

    Enumerate HAC / moment estimators applicable to a given
    `(scope, signal)`.

    [api/list-estimators →](list-estimators.md)

-   __`suggest_config`__

    ---

    Recommend the nearest legal config from panel shape; the recovery
    path for `MissingConfigError` from `evaluate`.

    [api/suggest-config →](suggest-config.md)

</div>
