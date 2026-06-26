---
title: factrix.datasets
---

Synthetic panel generators for examples, tests, and documentation. All
emit raw canonical-column panels (`date, asset_id, price`, then one
`factor` column for the single-factor generators or `n_factors` factor
columns for the multi-factor ones); attach `forward_return` via
[`factrix.preprocess.compute_forward_return`](preprocess.md) before
passing to [`evaluate`](evaluate.md).

The dataset's `signal_horizon` is a property of the generated
synthetic signal, not a pipeline parameter. When
`evaluate()`'s `forward_periods == signal_horizon` the pipeline
realizes the nominal information coefficient (IC) / drift; other horizons realize a decayed
signal.

## Single-factor generators

::: factrix.datasets.make_cs_panel

::: factrix.datasets.make_event_panel

## Multi-factor generators

For factor-screening / multiple-testing workflows — each emits
`n_factors` factor columns on one shared panel.

::: factrix.datasets.make_multi_factor_panel

::: factrix.datasets.make_multi_factor_event_panel
