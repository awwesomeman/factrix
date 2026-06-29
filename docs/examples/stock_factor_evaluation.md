---
title: Stock factor evaluation
---

Evaluate a per-stock factor before it becomes a portfolio signal. This recipe
keeps the workflow inside factrix's boundary: preprocess the factor, validate
its cross-sectional edge, compare raw and neutralized variants, and read
feasibility diagnostics. It does not select the final factor mix or construct
portfolio weights.

The markdown page is the canonical write-up. A runnable notebook mirror lives at
[`examples/stock_factor_evaluation.ipynb`](https://github.com/awwesomeman/factrix/blob/main/examples/stock_factor_evaluation.ipynb).

## Factor type

This recipe uses dense, per-stock factors: one value per `(date, asset_id)`.
The detected cell should be `FactorScope.INDIVIDUAL`,
`FactorDensity.DENSE`, and `DataStructure.PANEL`.

The headline test is `ic(inference=fx.inference.NEWEY_WEST)`: per-date
Spearman rank correlation between factor and forward return, aggregated to a
Newey-West HAC t-statistic on the mean IC. We also run `fm_beta`,
`quantile_spread`, and `notional_turnover` as second reads.

## Use this when

- The factor varies across stocks on each date: momentum, value, quality,
  revision, short interest, low volatility, and similar cross-sectional signals.
- The universe is wide enough for clean cross-sectional inference
  (`n_assets >= 30` is the practical floor for first-pass research).
- You want to compare raw and neutralized variants before deciding whether the
  factor is a stock-selection signal or mostly a group exposure.

## 1. Setup

```python
import factrix as fx
import polars as pl
from factrix.metrics import fm_beta, ic, notional_turnover, quantile_spread
from factrix.preprocess import OrthogonalizeResult, compute_forward_return
from factrix.preprocess import cross_sectional_zscore, mad_winsorize, orthogonalize_factor
```

## 2. Build a panel with stock-style exposures

Real research panels usually carry sector, size, liquidity, and coverage
labels alongside the factor. The synthetic panel below starts with a known
factor, then adds simple sector and size exposures so the neutralization step
has something to remove.

```python
raw = fx.datasets.make_cs_panel(
    n_assets=240,
    n_dates=504,
    ic_target=0.04,
    seed=2024,
)

asset_num = pl.col("asset_id").str.extract(r"(\d+)$").cast(pl.Int64)
raw = raw.with_columns(
    (asset_num % 8).cast(pl.Utf8).alias("sector"),
    pl.col("price").log().rank(method="average").over("date").alias("size_exposure"),
)

raw = raw.with_columns(
    (
        pl.col("factor")
        + 0.35 * pl.col("size_exposure")
        + pl.when(pl.col("sector") == "0").then(0.50).otherwise(0.0)
    ).alias("value_raw")
)
```

In production, replace the synthetic `sector` and `size_exposure` columns with
point-in-time vendor fields. Keep those labels on the same date axis as the
factor and price panel; factrix will not fix an upstream calendar mismatch.

## 3. Winsorize and standardize the raw factor

factrix does not normalize factor values implicitly. For dense stock-selection
work, a common first-pass workflow is MAD winsorization within each date, then
cross-sectional z-scoring.

```python
raw = mad_winsorize(raw, factor_col="value_raw", n_mad=3.0)
raw = cross_sectional_zscore(raw, factor_col="value_raw").rename(
    {"factor_zscore": "value_raw_z"}
)
```

The evaluated "raw" variant below means "not exposure-neutralized"; it is still
winsorized and standardized.

## 4. Attach forward returns

Attach one forward-return horizon before running metrics. `compute_forward_return`
enters at `t + 1`, exits at `t + 1 + forward_periods`, and stores a per-period
normalized return.

```python
panel = compute_forward_return(raw, forward_periods=5).with_columns(
    pl.col("value_raw_z").alias("value_raw")
)
```

## 5. Create sector and size neutral variants

`orthogonalize_factor` residualizes one factor against a base exposure matrix
using per-date cross-sectional OLS. Categorical exposures should be dummy
encoded before they are passed in. Drop one sector dummy when an intercept is
present; the helper uses `lstsq`, but a full dummy set adds no information.

```python
sector_dummies = panel.select("date", "asset_id", "sector").to_dummies(
    columns=["sector"]
)
sector_cols = [c for c in sector_dummies.columns if c.startswith("sector_")]
sector_cols = sector_cols[1:]

panel = panel.join(
    sector_dummies.select("date", "asset_id", *sector_cols),
    on=["date", "asset_id"],
)


def add_neutral_factor(
    data: pl.DataFrame,
    *,
    factor_col: str,
    base_cols: list[str],
    out_col: str,
) -> tuple[pl.DataFrame, OrthogonalizeResult]:
    factor_df = data.select(
        "date",
        "asset_id",
        pl.col(factor_col).alias("factor"),
    )
    base_df = data.select("date", "asset_id", *base_cols)
    ortho = orthogonalize_factor(factor_df, base_df, base_cols=base_cols)
    out = data.join(
        ortho.data.select(
            "date",
            "asset_id",
            pl.col("factor").alias(out_col),
        ),
        on=["date", "asset_id"],
    )
    return out, ortho


panel, sector_ortho = add_neutral_factor(
    panel,
    factor_col="value_raw",
    base_cols=sector_cols,
    out_col="value_sector_neutral",
)
panel, size_ortho = add_neutral_factor(
    panel,
    factor_col="value_raw",
    base_cols=["size_exposure"],
    out_col="value_size_neutral",
)
panel, sector_size_ortho = add_neutral_factor(
    panel,
    factor_col="value_raw",
    base_cols=[*sector_cols, "size_exposure"],
    out_col="value_sector_size_neutral",
)
```

Read the orthogonalization result before looking at p-values. A high
`mean_r_squared` means a large share of the raw factor is explained by the base
exposures.

```python
for name, ortho in {
    "sector": sector_ortho,
    "size": size_ortho,
    "sector_size": sector_size_ortho,
}.items():
    print(
        f"{name:12s} "
        f"coverage={ortho.coverage:.1%} "
        f"mean_r_squared={ortho.mean_r_squared:.3f}"
    )
```

## 6. Evaluate raw and neutralized variants

Evaluate all variants in one call. The `factor_cols` names become the hypothesis
identities used later by `compare()` and `bhy()`.

```python
metrics = {
    "ic": ic(inference=fx.inference.NEWEY_WEST),
    "fm": fm_beta(),
    "spread": quantile_spread(n_groups=5),
    "turnover": notional_turnover(n_groups=5),
}

factor_cols = [
    "value_raw",
    "value_sector_neutral",
    "value_size_neutral",
    "value_sector_size_neutral",
]

results = fx.evaluate(
    panel,
    metrics=metrics,
    factor_cols=factor_cols,
    forward_periods=5,
)

board = fx.compare(
    list(results.values()),
    metrics=["ic", "fm", "spread", "turnover"],
    sort_by="ic_p_value",
    descending=False,
)
print(board)
```

Illustrative table shape:

```text
factor                       forward_periods   ic       ic_p_value   fm        spread    turnover
value_sector_size_neutral    5                 0.0376   8.3e-36      0.0113    0.00095   0.80
value_size_neutral           5                 0.0374   3.2e-33      0.0109    0.00077   0.79
value_raw                    5                 0.0015   0.80         0.0000   -0.00007   0.11
value_sector_neutral         5                 0.0009   0.88        -0.0000   -0.00007   0.12
```

Read the rows as a research comparison, not as an optimizer:

- If raw significance disappears after sector or size neutralization, the edge is
  likely a group exposure rather than stock-selection alpha.
- If a neutralized variant becomes stronger than the raw factor, a stable
  exposure was diluting the stock-level ordering.
- If IC survives but spread falls, the ranking still carries information but the
  tail portfolio may be weaker.
- `turnover` is a feasibility diagnostic; it has no p-value and does not decide
  whether the factor is "good".
- FM beta magnitudes depend on factor scale. For neutralized residuals, compare
  the sign, p-value, and stability before comparing raw beta levels.

## 7. Check whether all-market IC is really stock selection

A sector-level factor can pass all-market IC because sectors rotate. That is a
valid group-exposure result, but it is not the same as within-sector stock
selection. A high tie ratio or a low-cardinality warning should prompt this
check.

```python
per_sector = fx.by_slice(
    panel,
    ic(inference=fx.inference.NEWEY_WEST),
    by="sector",
    factor_col="value_raw",
    forward_periods=5,
)

for sector, res in per_sector.items():
    out = res.metrics["ic"]
    print(sector, out.value, out.p_value, out.metadata.get("tie_ratio"))
```

If the all-market IC is significant but within-sector IC is weak or undefined,
the next research question is sector rotation or allocation, not stock
selection. Keep that distinction outside the factor-ranking table so the
downstream strategy does not double-count a sector bet as idiosyncratic alpha.

## 8. Point-in-time checklist

factrix cannot infer whether a factor was available at the signal timestamp. A
future-return leak will look like a strong factor because the input hypothesis is
already invalid. Before passing data to factrix, check:

- Fundamentals, analyst data, and short-interest data are joined with
  `publish_date` or vendor availability timestamp, not just `report_date`.
- `join_asof(..., strategy="backward")` is used so later disclosures never fill
  earlier signal dates.
- Reporting lags are explicit when publish timestamps are unavailable.
- The universe is point-in-time and handles delistings or inactive names
  consistently.
- Prices used for forward returns are adjusted consistently with the factor's
  research question.
- Factor, price, sector, size, liquidity, and regime labels share the same date
  axis before `compute_forward_return`.
- A placebo or deliberate future-return leak smoke test is treated as a workflow
  check, not as proof that all leakage is impossible.

## What to do next

Use this page for the single-factor workflow. For batch FDR control across many
candidates, continue with [Multi-factor screening](multi_factor_screening.md).
For scalar gross-to-net cost arithmetic after `spread` and `turnover`, use
[`breakeven_cost` and `net_spread`](../api/metrics/tradability.md) as standalone
post-processing helpers.
