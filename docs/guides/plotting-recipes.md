---
title: Plotting recipes
---

# Plotting recipes

factrix does not ship a plotting module. The snippets below are reference code
to copy into a notebook and adapt; statistical computation stays in factrix,
while labels, themes, tooltips, and export policy stay with the caller.

Install Plotly 6 or newer in the notebook environment. Plotly Express accepts
the Polars frames below directly, so no pandas conversion or factrix-specific
wrapper is needed:

```console
python -m pip install "plotly>=6"
```

The Python fences are marked `Illustrative` because the factrix documentation
test environment intentionally does not install this optional backend. They
are still complete snippets once Plotly is present; CI syntax-compiles them but
does not claim to render the figures.

Every example uses a synthetic factrix dataset and keeps the full period grid.
Consequently, `compute_forward_return(..., forward_periods=h)` stamps
`overlap_periods == h`. If you pass a coarser `dates=` evaluation grid, use the
derived overlap stamp when calling non-overlapping producers rather than
assuming it still equals the economic horizon.

## IC path, cumulative diagnostic, and distribution

**Purpose.** Inspect the per-period rank IC, its cumulative diagnostic, its
distribution, and a chronological in-sample/out-of-sample split. Cumulative IC
is a sum of correlations, not a compounded strategy return.

**Callable and columns.** `compute_ic()` returns a dictionary keyed by factor.
Each frame contains `date`, `ic`, `tie_ratio`, `n_assets`, and `_drop_stats`.

```python title="Illustrative"
import factrix as fx
import plotly.express as px
import polars as pl
from factrix.metrics.ic import compute_ic

raw = fx.datasets.make_cs_panel(n_assets=80, n_dates=240, rng=7)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
ic_series = compute_ic(panel, factor_cols=("factor",))["factor"]

split_at = ic_series["date"][int(0.7 * ic_series.height)]
path = ic_series.with_columns(
    pl.col("ic").cum_sum().alias("cumulative_ic"),
    pl.when(pl.col("date") < split_at)
    .then(pl.lit("IS"))
    .otherwise(pl.lit("OOS"))
    .alias("sample"),
)

px.line(path, x="date", y="ic", color="sample", title="Per-period IC").show()
px.line(
    path, x="date", y="cumulative_ic", color="sample", title="Cumulative IC"
).show()
px.histogram(
    ic_series, x="ic", nbins=30, marginal="box", title="IC distribution"
).show()
```

**Variations.** Concatenate horizon results with a `forward_periods` literal
for an IC-decay chart. Add a predeclared regime label before grouping to compare
regimes. For raw versus orthogonalized factors or separate universes, compute
each producer frame independently and add a `variant` or `universe` column
before concatenating; do not pool their cross-sections by accident.

## Quantile shape and long-short spread

**Purpose.** Check whether returns move monotonically across factor buckets and
whether the top-minus-bottom spread is stable through time.

**Callables and columns.** `compute_group_returns()` returns `group` and
`mean_return`. `compute_spread_series()` returns a dictionary keyed by factor;
each frame contains `date`, `spread`, `top_return`, `bottom_return`, and
`universe_return`. Both functions assign the buckets, so plotting code must not
rebucket the panel.

```python title="Illustrative"
import factrix as fx
import plotly.express as px
from factrix.metrics.quantile import compute_group_returns, compute_spread_series

horizon = 5
raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=240, rng=11)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=horizon)

groups = compute_group_returns(
    panel, overlap_periods=horizon, n_groups=5, factor_col="factor"
)
spread = compute_spread_series(
    panel, overlap_periods=horizon, n_groups=5, factor_cols=("factor",)
)["factor"]

px.bar(
    groups, x="group", y="mean_return", title="Equal-date mean by bucket"
).show()
px.line(
    spread,
    x="date",
    y=["spread", "top_return", "bottom_return"],
    title="Non-overlapping quantile returns",
).show()
```

**Variations.** Facet precomputed frames by horizon, regime, variant, or
universe. Keep `overlap_periods` aligned with the frame's evaluation-grid stamp;
the producer deliberately samples before bucketing so the time-series chart is
not an overlapping pseudo-return series.

## Event abnormal-return path

**Purpose.** Inspect directional leakage before an event and the signed
abnormal-return path after it, with the producer's standard-error and censoring
audit intact.

**Callable and columns.** `event_around_return()` returns a `MetricResult`.
`metadata["per_offset"]` maps each offset to `mean`, `se`, quartiles,
`hit_rate`, `n`, and censoring counts. This is deliberately not a CAAR curve:
positive offsets are cumulative from one post-event entry, while zero and
negative offsets are single-bar returns. Both sides are already
direction-adjusted and benchmark-adjusted. Do not sign or accumulate them
again.

```python title="Illustrative"
import factrix as fx
import plotly.express as px
import polars as pl
from factrix.metrics.event_horizon import event_around_return

offsets = [-6, -3, -1, 1, 6, 12, 24]
raw = fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=13)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
metric = event_around_return(panel, price_data=raw, offsets=offsets)

path = pl.DataFrame(
    [
        {"offset": offset, **stats}
        for offset, stats in metric.metadata["per_offset"].items()
        if stats["mean"] is not None
    ]
).select("offset", "mean", "se", "p25", "p75", "n")

fig = px.line(path, x="offset", y="mean", error_y="se", markers=True)
fig.add_vline(x=0, line_dash="dot")
fig.update_layout(title="Published event abnormal-return path")
fig.show()
```

**Variations.** Use `p25` and `p75` as an interquartile band, or expose
`computed`, `censored`, and `censor_reasons` beside the chart. Take formal
significance from event inference metrics such as `caar`, `bmp_z`, or
`corrado_rank`; the path itself is descriptive and has no per-offset p-value.

## Raw versus adjusted p-values

**Purpose.** Audit which hypotheses survived FDR control without losing a
horizon or parameter-sweep identity.

**Callables and columns.** `EvaluationResult.to_frame()` supplies raw
`p_value`; a screening result's `to_frame()` supplies `adj_p` and `survived`.
Both start with `factor`, `forward_periods`, and sorted `params` columns.

```python title="Illustrative"
import factrix as fx
import plotly.express as px
import polars as pl
from factrix.metrics import ic

raw = fx.datasets.make_cs_panel(n_assets=80, n_dates=240, rng=17).with_columns(
    (-pl.col("factor")).alias("reversal")
)
panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
evaluated = fx.evaluate(
    panel,
    metrics={"ic": ic()},
    factor_cols=["factor", "reversal"],
)
screen = fx.multi_factor.bhy(list(evaluated.values()), metrics=["ic"])["ic"]

identity = ["factor", "forward_periods"]  # append params keys for a sweep
raw_p = pl.concat([r.to_frame() for r in evaluated.values()]).filter(
    pl.col("metric_name") == "ic"
).select(*identity, "p_value")
audit = raw_p.join(screen.to_frame(), on=identity, how="inner")

px.scatter(
    audit,
    x="p_value",
    y="adj_p",
    color="survived",
    hover_data=identity,
    title="Raw versus BHY-adjusted p-values",
).show()
```

**Variations.** Add every `params` key to `identity` for timeframe, universe,
or model sweeps. Cross-metric screens already include `metric` in their output;
join it to `metric_name` explicitly. Never join only on `factor`: the same name
can legally identify several hypotheses.

## Why there is no plotting wrapper

These recipes keep rendering out of the core wheel. Plotting libraries add
substantial install size and independent release churn; more importantly, a
wrapper would have to own subjective choices such as themes, bins, rolling
windows, split annotations, and export formats. Polars-native producer frames
are the stable bridge and let each user choose Plotly, matplotlib, Altair, or a
dashboard without factrix becoming the abstraction between them.

Plotly references: [data-frame arguments](https://plotly.com/python/px-arguments/)
and [Narwhals-backed dataframe support](https://plotly.com/python/performance/).
