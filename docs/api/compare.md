---
title: factrix.compare
---

::: factrix.compare

Leaderboard renderer that stacks N evaluation results side by side as a
[polars `DataFrame`](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html).
Pure projection — no metric is recomputed.

```python
import factrix as fx
from factrix.metrics import ic, quantile_spread

results = fx.evaluate(
    data,
    metrics={"ic": ic(inference=fx.inference.NEWEY_WEST), "spread": quantile_spread()},
    factor_cols=candidates,
)
# evaluate() returns a dict keyed by factor; compare() takes the list of results.
df = fx.compare(list(results.values()), metrics=["ic", "spread"], sort_by="ic")
```

## Input parameters

```python
compare(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    sort_by: str | None = None,
    descending: bool = True,
) -> pl.DataFrame
```

### Column layout

The returned `pl.DataFrame` contains the following columns:

- `factor`: The name of the evaluated factor.
- `forward_periods`: The forward periods horizon.
- Context keys: All context keys present across the evaluation results, ordered by first appearance.
- `<metric_name>`: The metric value (e.g. `ic`).
- `<metric_name>_p_value`: The metric p-value if applicable (e.g. `ic_p_value`).
- `rank`: Rank column (only populated when `sort_by` is set).

## Parameter details

| Kwarg | Default | Meaning |
|-------|---------|---------|
| `metrics` | (required) | `list[str]` of metric labels to include in the leaderboard. |
| `sort_by` | `None` | The metric label to sort the leaderboard by. `None` keeps the original list order. |
| `descending` | `True` | Whether to sort in descending order (higher is better). Set to `False` for lower-is-better metrics. |
