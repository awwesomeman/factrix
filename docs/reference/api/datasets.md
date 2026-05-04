# datasets

Synthetic panel generators for examples, tests, and documentation.
Both emit raw canonical-column panels (`date, asset_id, price,
factor`); attach `forward_return` (e.g. via
`factrix.preprocess.returns.compute_forward_return`) before passing
to [`evaluate`](evaluate.md).

The dataset's `signal_horizon` is a property of the generated
synthetic signal, not a pipeline parameter. When
`AnalysisConfig.forward_periods == signal_horizon` the pipeline
realizes the nominal IC / drift; other horizons realize a decayed
signal.

::: factrix.datasets.make_cs_panel

::: factrix.datasets.make_event_panel
