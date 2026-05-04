# factrix

Polars-native **factor signal validation toolkit** for quantitative finance.

factrix screens whether individual factors carry statistical edge using IC,
Fama-Macbeth, CAAR, or timeseries β procedures, and applies multi-testing
correction (BHY) across factor families. It is a *signal validator*, not a
backtest engine — feed surviving factors into downstream portfolio
constructors.

## Documentation

- [Methodology reference](methodology_reference.md) — IC / FM / CAAR / TS-β derivations.
- [Statistical methods](statistical_methods.md) — NW HAC, test selection, BHY.
- [Metric applicability](metric_applicability.md) — when to use each metric, N/T thresholds.

## Source

- Repository: [github.com/awwesomeman/factrix](https://github.com/awwesomeman/factrix)
- Install + smoke test: see the [README](https://github.com/awwesomeman/factrix#readme).
