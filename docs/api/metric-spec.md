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
title: SpecRole
---

# SpecRole

factrix uses roles to differentiate between public, user-facing metrics and internal pipeline stages.

::: factrix._axis.SpecRole
