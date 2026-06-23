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
        <img src="https://github.com/awwesomeman/factrix/actions/workflows/docs-deploy-release.yml/badge.svg?branch=main" />
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
    <a href="https://awwesomeman.github.io/factrix/latest/" title="Documentation">
        <img src="https://img.shields.io/badge/docs-mkdocs--material-526CFE?logo=materialformkdocs&logoColor=white" />
    </a>
    <a href="https://github.com/awwesomeman/factrix/stargazers" title="Stars">
        <img src="https://img.shields.io/github/stars/awwesomeman/factrix?style=flat" />
    </a>
</p>

<h3 align="center"><b>Tests one factor. Screens a thousand.</b></h3>

## Where factrix fits

**Does this factor possess predictive edge?**

factrix is the first Polars-native Python toolkit that picks the right statistical test for each factor type. Cross-sectional, event, common factor — each gets the tests that fit its data-generating process.

```
factor construction  →  factrix (inference)  →  strategy construction  →  backtest  →  live trading
                            ▲ you are here
```

For each candidate factor factrix answers — *is the predictive power real?* — and corrects for multiple testing when you screen at scale. Kill fakes before they cost you a backtest.

### Why factrix?

- **Type-routed evaluation** — Information Coefficient + Fama-MacBeth for cross-sectional factors; Cumulative Average Abnormal Return for events.
- **Financial statistics built in** — autocorrelation-robust standard errors (Newey-West), overlapping-forward-return correction, persistent-predictor flagging (Stambaugh bias), and false-discovery-rate control across batches (Benjamini-Hochberg-Yekutieli).
- **Polars-native** — modern Polars alternative to the pandas-based alphalens.

factrix stops at the inference — primary test plus diagnostic battery. It does not size positions, model slippage, optimise weights, or compose alphas; those belong to the later stages of the pipeline.

### Is factrix the right tool?

| You want to… | Use this |
|---|---|
| Inference on a factor (cross-sectional / event / common factor) | **factrix** |
| Screen many factors with multiple-testing correction | **factrix** |
| Backtest with positions / slippage / margin | [zipline-reloaded][zipline], [backtrader][backtrader], [bt][bt], [vectorbt][vectorbt], [nautilus_trader][nautilus] |
| Optimise portfolio weights | [skfolio][skfolio], [riskfolio-lib][riskfolio] |
| Returns-level tear-sheet (P&L diagnostics) | [pyfolio-reloaded][pyfolio], [QuantStats][quantstats] |
| Familiar cross-sectional tear-sheet | [alphalens-reloaded][alphalens] |
| End-to-end machine-learning pipeline | [qlib][qlib] |
| Deflated / probabilistic Sharpe today (commercial) | [mlfinlab][mlfinlab] |

[alphalens]: https://github.com/stefan-jansen/alphalens-reloaded
[vectorbt]: https://github.com/polakowo/vectorbt
[zipline]: https://github.com/stefan-jansen/zipline-reloaded
[backtrader]: https://github.com/mementum/backtrader
[bt]: https://github.com/pmorissette/bt
[nautilus]: https://github.com/nautechsystems/nautilus_trader
[skfolio]: https://skfolio.org/
[riskfolio]: https://github.com/dcajasn/Riskfolio-Lib
[pyfolio]: https://github.com/stefan-jansen/pyfolio-reloaded
[quantstats]: https://github.com/ranaroussi/quantstats
[qlib]: https://github.com/microsoft/qlib
[mlfinlab]: https://github.com/hudson-and-thames/mlfinlab
[full-comparison]: https://awwesomeman.github.io/factrix/latest/where-factrix-fits/

## Installation

```bash
pip install factrix
# or
uv add factrix
```

See the [installation guide](https://awwesomeman.github.io/factrix/latest/getting-started/install/) for version pinning and development setup.

## Typical usage

**Single factor — IC evaluation**

```python
import factrix as fx
from factrix.metrics import ic

raw   = fx.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
data  = fx.preprocess.compute_forward_return(raw, forward_periods=5)

results = fx.evaluate(
    data,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST)},
    factor_cols=["factor"],
    forward_periods=5,
)
res = results["factor"]
ic_res = res.metrics["ic"]

print('ic_mean =', round(ic_res.value, 4))
print('p_value =', round(ic_res.p_value, 4))
```

**Multi-factor BHY screening**

```python
fdr_results = fx.multi_factor.bhy(list(results.values()), metrics=["ic"], q=0.05)
bhy_ic = fdr_results["ic"]
print("survivors =", [r.factor for r in bhy_ic.survivors])
```

**Single-asset (timeseries) fallback**

```python
from factrix.metrics import ts_beta

# Automatically resolves structure axis to DataStructure.TIMESERIES when N == 1
results = fx.evaluate(
    single_asset_data,
    metrics={"ts_beta": ts_beta()},
    factor_cols=["macro_factor"],
    forward_periods=5,
)
print(results["macro_factor"].metrics["ts_beta"].value)
```

## Documentation

- [**Get Started**](https://awwesomeman.github.io/factrix/latest/) — install, quickstart, where factrix fits
- [**User Guide**](https://awwesomeman.github.io/factrix/latest/guides/) — concepts (three-axis design, architecture), how-to (PANEL vs TIMESERIES, BHY screening, slice analysis), examples
- [**API Reference**](https://awwesomeman.github.io/factrix/latest/api/) — entry points, results, lookup tables, per-metric pages
- [**Development**](https://awwesomeman.github.io/factrix/latest/development/contributing/) — contributing, design notes
- [**Release Notes**](https://awwesomeman.github.io/factrix/latest/development/changelog/) — changelog

## License

Released under the [Apache License 2.0](LICENSE).
