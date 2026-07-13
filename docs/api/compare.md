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
- Params keys: All `params` keys present across the evaluation results, ordered by first appearance.
- `<metric_label>`: The metric value, where `metric_label` is the key in
  `EvaluationResult.metrics` and the string passed in `metrics=[...]` (e.g.
  `ic`).
- `<metric_label>_p_value`: The metric p-value if applicable (e.g.
  `ic_p_value`). With a custom evaluation label such as
  `metrics={"ic_nw": ic(...)}`, pass `metrics=["ic_nw"]` to `compare()` and
  the p-value column is `ic_nw_p_value`.
- `rank`: Rank column, present only when `sort_by` is set (the column is absent otherwise).

## Parameter details

| Kwarg | Default | Meaning |
|-------|---------|---------|
| `metrics` | (required) | `list[str]` of metric labels to include in the leaderboard. |
| `sort_by` | `None` | Any output column produced before ranking: `factor`, `forward_periods`, `params` keys, metric value columns such as `ic`, or p-value columns such as `ic_p_value` / `<metric_label>_p_value`. `None` keeps the original list order. |
| `descending` | `True` | Whether to sort in descending order (higher is better). Set to `False` for lower-is-better metrics. |

`rank` is created after sorting, so it is not a valid `sort_by` key. To sort
by significance, use the p-value column and set `descending=False`:

```python
df = fx.compare(results, metrics=["ic"], sort_by="ic_p_value", descending=False)
```

For signed metrics such as `predictive_beta`, sorting by the raw value answers
"largest positive effect", not "strongest evidence". A strongly negative but
highly significant factor will rank low under `sort_by="predictive_beta"` with
the default descending order. Use the p-value column for significance screens,
or sort on `abs(value)` in caller code when magnitude regardless of sign is the
intended ranking.
