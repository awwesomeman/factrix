---
title: factrix.estimators
---

# factrix.estimators

`factrix.estimators` exposes stateless, lowercase functional aliases for the core statistical estimators. These functions execute calculations directly on raw NumPy arrays without requiring configuration or class instantiation.

---

## `newey_west`

::: factrix.estimators.newey_west

---

## `hansen_hodrick`

::: factrix.estimators.hansen_hodrick

---

## `driscoll_kraay`

::: factrix.estimators.driscoll_kraay

---

## `block_bootstrap`

::: factrix.estimators.block_bootstrap

---

## `list_estimators`

::: factrix.list_estimators

Programmatic discovery of the estimator surface above. `list_estimators()`
returns the registered estimators (a `list[str]` of names; `format="json"`
returns structured dicts). Applicability to a given cell is checked by the
estimator itself at evaluate time, not by a filter argument.

```python
import factrix as fx

fx.list_estimators()                 # ['BlockBootstrap', 'DriscollKraay', 'GMM', ...]
fx.list_estimators(format="json")    # structured per-estimator metadata
```
