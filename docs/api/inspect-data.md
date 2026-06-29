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

## Result structure

`inspect_data` returns a `DataInspection` carrying the detected data
properties (`properties`), the per-metric applicability verdicts
(`metrics`, plus the `usable` / `degraded` / `unusable` partitions), and
any data-level `warnings`. Each entry in the metrics group is a
`MetricApplicability`.

::: factrix.DataInspection

---

::: factrix.DataProperties

---

::: factrix.MetricApplicabilityGroup

---

::: factrix.MetricApplicability
