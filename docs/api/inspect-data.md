---
title: factrix.inspect_data
---

::: factrix.inspect_data

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

## Result structure

`inspect_data` returns a `DataInspection` carrying the detected data
properties (`properties`), the per-metric applicability verdicts
(`metrics`, plus the `usable` / `degraded` / `unusable` partitions), and
any data-level `warnings`. Each entry in the metrics group is a
`MetricApplicability`.

The tier groups are `MetricApplicabilityGroup` objects: they expose `.names`
and `.to_metrics_dict()`, and slicing or concatenating them with `+` preserves
those helpers.

::: factrix.DataInspection

---

::: factrix.DataProperties

---

::: factrix.MetricApplicabilityGroup

---

::: factrix.MetricApplicability
