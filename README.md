<div align="center">

# factrix

</div>

<p align="center">
    <a href="https://github.com/awwesomeman/factrix/releases" title="Version">
        <img src="https://img.shields.io/github/v/release/awwesomeman/factrix?color=3670A0&label=version" />
    </a>
    <a href="https://www.python.org/downloads/" title="Python versions">
        <img src="https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white" />
    </a>
    <a href="https://pola.rs/" title="Polars-native">
        <img src="https://img.shields.io/badge/polars-native-CD792C?logo=polars&logoColor=white" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/blob/main/LICENSE" title="License">
        <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
    </a>
    <a href="https://awwesomeman.github.io/factrix/" title="Documentation">
        <img src="https://img.shields.io/badge/docs-mkdocs--material-526CFE?logo=materialformkdocs&logoColor=white" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/stargazers" title="Stars">
        <img src="https://img.shields.io/github/stars/awwesomeman/factrix?style=flat" />
    </a>
</p>

<h3 align="center"><b>Tests one factor. Screens a thousand.</b></h3>

A polars-native factor signal validator, **factrix**. It answers a single core question — **Does this factor carry statistical edge?** 

## Installation

```bash
uv pip install git+https://github.com/awwesomeman/factrix.git
```

See the [installation guide](https://awwesomeman.github.io/factrix/getting-started/install/) for `pip` / `conda`, version pinning, and development setup.

## Quickstart

```python
import factrix as fl
from factrix.preprocess.returns import compute_forward_return

raw   = fl.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

cfg     = fl.AnalysisConfig.individual_continuous(metric=fl.Metric.IC, forward_periods=5)
profile = fl.evaluate(panel, cfg)

print(profile.verdict(), '| primary_p =', round(profile.primary_p, 4))
# → pass | primary_p = 0.0
```

## Documentation

- [**Get Started**](https://awwesomeman.github.io/factrix/getting-started/) — install, quickstart, three-axis concepts
- [**Guides**](https://awwesomeman.github.io/factrix/guides/) — PANEL vs TIMESERIES, BHY batch screening, choosing a metric
- [**Reference**](https://awwesomeman.github.io/factrix/reference/metric-applicability/) — applicability tables, formulas, statistical methods
- [**Development**](https://awwesomeman.github.io/factrix/development/architecture/) — architecture, contributing

## License

Released under the [Apache License 2.0](LICENSE).
