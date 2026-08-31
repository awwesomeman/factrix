<div align="center">

<img src="https://raw.githubusercontent.com/awwesomeman/factrix/main/docs/assets/factrix_banner_light.png" alt="factrix" />

</div>

<p align="center">
    <a href="https://github.com/awwesomeman/factrix/releases" title="Version">
        <img src="https://img.shields.io/github/v/release/awwesomeman/factrix?color=3670A0&label=version" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/actions/workflows/test.yml" title="test workflow">
        <img src="https://github.com/awwesomeman/factrix/actions/workflows/test.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/actions/workflows/docs-deploy-release.yml" title="docs workflow">
        <img src="https://github.com/awwesomeman/factrix/actions/workflows/docs-deploy-release.yml/badge.svg" />
    </a>
    <a href="https://www.python.org/downloads/" title="Python versions">
        <img src="https://img.shields.io/pypi/pyversions/factrix.svg?logo=python&logoColor=white" />
    </a>
    <a href="https://pola.rs/" title="Polars-native">
        <img src="https://img.shields.io/badge/polars-native-CD792C?logo=polars&logoColor=white" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/blob/main/LICENSE" title="License">
        <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
    </a>
    <a href="https://awwesomeman.github.io/factrix/latest/" title="Documentation">
        <img src="https://img.shields.io/badge/docs-mkdocs--material-526CFE?logo=materialformkdocs&logoColor=white" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/stargazers" title="Stars">
        <img src="https://img.shields.io/github/stars/awwesomeman/factrix?style=flat" />
    </a>
</p>

<h3 align="center"><b>Tests one factor. Screens a thousand.</b></h3>

<p align="center"><a href="https://awwesomeman.github.io/factrix/latest/"><b>📖 Full documentation</b></a></p>

## What factrix does

factrix is a Polars-native toolkit for testing whether a candidate factor has
predictive evidence. It provides factor-type-specific tests for
cross-sectional, event, and common factors, and rejects metric/data
combinations that do not apply.

- **Inference built for factor data** — overlap-aware standard errors,
  persistence diagnostics, and event-study methods.
- **Batch screening** — false-discovery-rate control for testing many factors.
- **Structured results** — estimates, p-values, metadata, and warnings remain
  separate instead of being collapsed into one score.

factrix covers factor inference and screening. Portfolio construction,
backtesting, and execution remain downstream; see
[Where factrix fits](https://awwesomeman.github.io/factrix/latest/where-factrix-fits/)
for the scope boundary and neighbouring tools.

## Installation

```bash
pip install factrix
# or
uv add factrix
```

See the [installation guide](https://awwesomeman.github.io/factrix/latest/getting-started/install/) for version pinning and development setup.

## Quickstart

```python
import factrix as fx
from factrix.metrics import ic

raw = fx.datasets.make_cs_panel(
    n_assets=100, n_dates=500, ic_target=0.08, rng=2024
)
data = fx.preprocess.compute_forward_return(raw, forward_periods=5)

results = fx.evaluate(
    data,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor"],
    forward_periods=5,
)
res = results["factor"]
ic_res = res.metrics["ic"]

print("ic_mean =", round(ic_res.value, 4))
print("p_value =", round(ic_res.p_value, 4))
```

The minimum input is a long Polars frame keyed by `date` and `asset_id`, with
a factor column and `forward_return`. Read the returned `p_value` together with
the result's sample metadata and warnings; the
[quickstart](https://awwesomeman.github.io/factrix/latest/getting-started/quickstart/)
shows the complete result shape.

## Next steps

- [Preparing data](https://awwesomeman.github.io/factrix/latest/guides/preparing-data/) — validate and align your own panel.
- [Choosing a metric](https://awwesomeman.github.io/factrix/latest/guides/choosing-metric/) — map a research question to a metric.
- [Multi-factor screening](https://awwesomeman.github.io/factrix/latest/examples/multi_factor_screening/) — BHY false-discovery-rate control across candidate factors.
- [Multi-horizon evaluation](https://awwesomeman.github.io/factrix/latest/api/multi-horizon/) — sweeping `forward_periods` with `evaluate_horizons`.
- [Panel vs timeseries](https://awwesomeman.github.io/factrix/latest/guides/panel-timeseries/) — understand data-shape dispatch.

## Documentation

- [**Get started**](https://awwesomeman.github.io/factrix/latest/) — installation and quickstart
- [**User guide**](https://awwesomeman.github.io/factrix/latest/guides/) — concepts, how-to guides, examples, and reference tables
- [**API reference**](https://awwesomeman.github.io/factrix/latest/api/) — entry points, result types, and per-metric pages
- [**Development**](https://awwesomeman.github.io/factrix/latest/development/contributing/) — contribution and design guidance
- [**Release notes**](https://awwesomeman.github.io/factrix/latest/development/changelog/) — changelog

## License

Released under the [Apache License 2.0](LICENSE).
