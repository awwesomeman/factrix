---
title: factrix.inspect_data
---

::: factrix.inspect_data

<hr>

## Warning declarations

`inspect_data` surfaces every pre-flight advisory both as a structured
`Warning` record and as a framed `UserWarning`. Use the same declaration as an
evaluation when a regime is intentional:

```python
import factrix as fx

panel = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
info = fx.inspect_data(
    panel,
    expected_warnings=("few_assets", "unreliable_se_short_periods"),
)
```

The records remain in `info.warnings`, `info.factors[col].warnings`, and each
metric verdict's `warnings`, marked with `expected=True`; only their stderr
echoes stop. Unknown codes are rejected so a typo cannot silently suppress
nothing.

<hr>

## Usability Tiers

`inspect_data` partitions public metrics into three distinct groups based on the inspected data shape and the metric's declarative `sample_threshold`:

- **Usable**: The metric is fully applicable and the data shape satisfies all warning thresholds (`warn_*`). This is the safest set to run out-of-the-box.
- **Degraded**: The metric is applicable but runs with a warning because the sample size is borderline (falls between the hard `min_*` floor and the soft `warn_*` threshold).
- **Unusable**: The metric cannot be run on this data, either because of a cell mismatch or because a hard sample floor (`min_*`) is violated.

Scalar-input helpers such as `breakeven_cost` and `net_spread` are also listed
as **unusable** for panel data. They consume already computed scalar values
(`quantile_spread.value`, `notional_turnover.value`) rather than a panel, so run
the upstream diagnostics first and call the helper directly.

<hr>

## Resolved floor for a configured metric

The tiers above read each metric's **default-configuration** `sample_threshold`
(as do `list_metrics` / `metrics_summary`). The floor a run gates on can differ:
the configuration changes it (`ic(inference=NEWEY_WEST)` needs 20 periods, the
default non-overlapping t-test 50), and stride-scaled floors follow the panel's
`overlap_periods` (`positive_rate()` needs 10 periods at `overlap_periods=1`, 50
at 5). `sample_requirements` resolves the floor for an instance at a horizon —
the same resolution `evaluate` and the `slice_period_*` tests apply — so a
coverage audit (regime slices, IS/OOS splits) can be planned against the
number the run will actually use. `evaluate(strict=True)` raises on a hard
`min_*` breach; `strict=False` short-circuits the metric to NaN with a
`metric_unavailable` warning; the soft `warn_*` tier always returns a value
and attaches the axis' degraded-tier warning code.

::: factrix.sample_requirements

<hr>

## Scope detection and missing cells

`FactorScope` asks one question: does the factor take the same value across
every asset at a given period (`COMMON`, a broadcast macro series) or vary
across the cross-section (`INDIVIDUAL`)?

The question is answered from **finite** cells only. A `null`, `NaN` or `±inf`
factor cell is a missing observation, not a distinct factor value, so it cannot
manufacture cross-sectional variation:

- A broadcast factor with a gap on one asset stays `COMMON`.
- A period left with a single finite cell is still compatible with `COMMON` —
  one observation cannot contradict a broadcast structure.
- An asset with no row at all on a period is likewise not variation.
- A period with **no** finite cell carries no evidence either way and is
  ignored. `properties.scope_reason` reports how many of the panel's periods
  the decision was read from, so an ignored period is visible rather than
  silent.

If *every* period is unidentifiable — the column has no finite cell anywhere —
the broadcast property cannot be established at all. Routing then falls back to
the unrestricted `FactorScope.INDIVIDUAL` (`COMMON` is its special case, and
claiming it would assert a structure no observation supports) and a
`factor_scope_unidentifiable` data-level warning is emitted. Every sample floor
is violated at `n_pairs = 0` in that state, so read the warning as a missing
factor column rather than as a scope verdict.

`evaluate` dispatches each factor column through the same detector, so the
pre-flight `scope` and the cell a run is routed to cannot disagree.

<hr>

## Result structure

`inspect_data` returns a `DataInspection` carrying the detected data
properties (`properties`), the per-metric applicability verdicts
(`metrics`, plus the `usable` / `degraded` / `unusable` partitions), and
any data-level `warnings`. Each entry in the metrics group is a
`MetricApplicability`.

The tier groups are `MetricApplicabilityGroup` objects: they expose `.names`
and `.to_metrics_dict()`, and slicing or concatenating them with `+` preserves
those helpers.

<hr>

## Multi-factor input: per-factor results

`evaluate` dispatches each factor column independently — one
`EvaluationResult` per column, each routed to its own `(scope, density,
structure)` cell. Inspection reports at the same granularity, so a preflight
verdict maps one-to-one onto the `evaluate` output it predicts:

| `inspect_data(data, factor_cols=cols)` | `evaluate(data, ..., factor_cols=cols)` |
|---|---|
| `info.factors[col].properties` | the cell `results[col]` was dispatched to |
| `info.factors[col].usable` / `.degraded` / `.unusable` | which entries of `results[col].metrics` return a value, a degraded value, or a `metric_unavailable` NaN |
| `info.factors[col].warnings` | the data-level warnings that column raises |
| `info.factors[col].usable.to_metrics_dict()` | the `metrics=` argument to run that column safely |

```python
import factrix as fx
import polars as pl
from factrix.preprocess import compute_forward_return

raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120)
data = compute_forward_return(
    raw.with_columns(pl.col("factor").mean().over("date").alias("macro")),
    forward_periods=5,
)
info = fx.inspect_data(data, factor_cols=["factor", "macro"])

for col, f in info.factors.items():
    print(col, f.properties.scope.value, f.properties.density.value, f.usable.names)

# Run each column with the metrics its own verdict cleared. strict=False keeps
# a metric whose run-time floor binds tighter than the pre-flight one as a NaN
# placeholder with a reason, rather than raising for the whole batch.
for col, f in info.factors.items():
    results = fx.evaluate(
        data,
        metrics=f.usable.to_metrics_dict(),
        factor_cols=[col],
        forward_periods=5,
        strict=False,
    )
```

Two granularities travel together on purpose. A screen over dozens of
candidate columns still wants one answer to "what is this panel", so
`properties`, `metrics` and the tier partitions remain a concise aggregate and
describe the **first** inspected column — exactly `factors[<first column>]`.
Single-factor callers therefore never touch the mapping; multi-factor callers
never have to guess which column an aggregate refers to.

When columns disagree on `FactorScope` or `FactorDensity`, the aggregate
`warnings` carry a `cross_factor_*_mismatch` naming the basis column, the
columns that disagree with it, and the value each one carries. A later
column's own advisories (low cardinality, frequent events, scope
unidentifiable, sample shape) live on `factors[col].warnings` rather than
being merged into one stream where the column they belong to would be lost.

One panel-level caveat: `n_periods` and `n_assets` are properties of the data,
so every `FactorInspection` reports the panel's counts rather than the periods
that column happens to cover. Where a column's own coverage gates a metric it
does so through the stage-one profile, which *is* per column — a column whose
IC cross-sections survive on 20 periods is blocked at `ic`'s 50-period floor
while its sibling in the same panel is not. The same rule applies to
time-series-first common betas: their `n_assets` floors use only the per-asset
regressions that survive `compute_common_betas`' complete-pair, minimum-history,
and factor-variation filters, rather than the raw panel universe. A
metric whose run-time gate is data content rather than sample shape is
mirrored the same way: `common_quantile_spread` is reported unusable when the
per-period factor history carries fewer than `n_groups * 2` distinct values,
the comparison it short-circuits `insufficient_factor_variation` on.

::: factrix.DataInspection

---

::: factrix.FactorInspection

---

::: factrix.DataProperties

---

::: factrix.MetricApplicabilityGroup

---

::: factrix.MetricApplicability
