---
title: factrix.MetricSpec & Registration
---

# MetricSpec & Registration

factrix uses declarative specification and a centralized registry to topologically sort and execute evaluations. You can document first-party or third-party metric behaviors and dependencies using these tools.

## MetricSpec

::: factrix.MetricSpec

<hr>

## `@metric_spec` Decorator

::: factrix.metric_spec

<hr>

## Metric Registration

::: factrix.metrics.register


---
title: SpecRole and Visibility
---

# SpecRole and Visibility

factrix uses roles to differentiate between public, user-facing metrics and internal pipeline stages.

## SpecRole

::: factrix._axis.SpecRole

---

## Historical: `Visibility` Enum

In older versions of factrix (prior to v0.14.0), visibility was controlled using a dedicated `Visibility` enum. In v0.14.0, this was unified into the `role` attribute of `MetricSpec` using `SpecRole`:

- **`Visibility.PUBLIC`** maps to **`SpecRole.METRIC`**: Denotes a user-facing, result-producing metric that appears in `list_metrics()` and can be evaluated.
- **`Visibility.INTERNAL`** maps to **`SpecRole.PIPELINE`**: Denotes a stage-1 intermediate calculation (e.g., `compute_ic`) that is executed as a dependency in the DAG executor, but whose raw output is excluded from final result partitions.
