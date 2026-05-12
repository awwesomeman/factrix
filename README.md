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

factrix is the first Polars-native Python toolkit that picks the
right statistical test for each factor type. Cross-sectional, event,
common factor — each gets the tests that fit its data-generating
process.

```
factor construction  →  factrix (inference)  →  strategy construction  →  backtest  →  live trading
                            ▲ you are here
```

For each candidate factor factrix answers — *is the predictive
power real?* — and corrects for multiple testing when you screen
at scale. Kill fakes before they cost you a backtest.

### Why factrix?

- **Type-routed evaluation** — Information Coefficient + Fama-MacBeth
  for cross-sectional factors; Cumulative Average Abnormal Return for
  events. Each type ships its own multi-metric diagnostic battery.
- **Batch factor screening** — screen hundreds of candidate factors
  with cross-test multiple-testing correction in a single API call.
- **Financial statistics built in** — autocorrelation-robust standard
  errors (Newey-West), overlapping-forward-return correction,
  persistent-predictor flagging (Stambaugh bias), false-discovery-rate
  control across batches (Benjamini-Hochberg-Yekutieli). Hand-rolled
  Information Coefficient loops typically run `scipy.stats.ttest_1samp`
  on the IC series and ignore the autocorrelation induced by
  overlapping forward returns.
- **Polars-native** — modern Polars alternative to the pandas-based
  alphalens.

factrix stops at the inference — primary test plus diagnostic battery.
It does not size positions, model slippage, optimise weights, or
compose alphas; those belong to the later stages of the pipeline above.

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

[Where factrix fits — full comparison →][full-comparison]

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
from factrix.preprocess import compute_forward_return

raw   = fx.datasets.make_cs_panel(n_assets=100, n_dates=500, ic_target=0.08, seed=2024)
panel = compute_forward_return(raw, forward_periods=5)

cfg     = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC, forward_periods=5)
profile = fx.evaluate(panel, cfg)

print('primary_p =', round(profile.primary_p, 4))
print(profile.diagnose())   # WarningCode / InfoCode list
```

**Multi-factor BHY screening**

```python
profiles  = [fx.evaluate(p, cfg) for p in [panel_a, panel_b, panel_c, panel_d, panel_e]]
survivors = fx.multi_factor.bhy(profiles, threshold=0.05)
```

**Single-asset (timeseries) fallback**

```python
cfg     = fx.AnalysisConfig.individual_continuous(metric=fx.Metric.IC, forward_periods=5)
profile = fx.evaluate(single_asset_panel, cfg)  # mode auto-switches to TIMESERIES
print(profile.stats.get(fx.StatCode.TS_BETA))
```

## Documentation

- [**Get Started**](https://awwesomeman.github.io/factrix/latest/) — install, quickstart, where factrix fits
- [**User Guide**](https://awwesomeman.github.io/factrix/latest/guides/) — concepts (three-axis design, architecture), how-to (PANEL vs TIMESERIES, BHY screening, slice analysis), examples
- [**API Reference**](https://awwesomeman.github.io/factrix/latest/api/) — entry points, results, lookup tables, per-metric pages
- [**Development**](https://awwesomeman.github.io/factrix/latest/development/contributing/) — contributing, design notes
- [**Release Notes**](https://awwesomeman.github.io/factrix/latest/development/changelog/) — changelog

## License

Released under the [Apache License 2.0](LICENSE).
