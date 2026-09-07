---
title: factrix.compare
---

::: factrix.compare

Leaderboard renderer that stacks N evaluation results side by side as a
[polars `DataFrame`](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html).
Pure projection — no metric is recomputed.

```python title="Illustrative"
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

```text
compare(
    results: list[EvaluationResult],
    *,
    metrics: list[str],
    sort_by: str | None = None,
    descending: bool | None = None,
    rank_method: Literal["min", "dense", "ordinal"] = "min",
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
| `descending` | `None` | Sort direction for `sort_by`. `None` resolves it from the column under the [direction rule](#sort-direction); `True` / `False` states it outright and always wins. There is no global direction default. |
| `rank_method` | `"min"` | How equal `sort_by` values are numbered: `"min"` (tied rows share the best rank, next rank skips: `1, 1, 3`), `"dense"` (share the rank, no gap: `1, 1, 2`), `"ordinal"` (every row numbered `1..N`). Polars' `Expr.rank` methods. |

### Sort direction

`compare()` has no global `descending` default: a lower-is-better key must never
inherit a higher-is-better sort. With `descending=None` the direction comes from
the column itself.

| `sort_by` column | Resolved direction |
|---|---|
| `<metric_label>_p_value` | ascending — a smaller p-value is stronger evidence |
| `factor`, `forward_periods`, `params` keys | ascending — these label a row, they do not score it |
| `ic`, `ic_ir`, `quantile_spread`, `quantile_spread_vw`, `common_quantile_spread`, `net_spread`, `breakeven_cost` | descending — higher is better by the metric's own definition |
| `rank_turnover`, `notional_turnover` | ascending — turnover is a cost driver |
| any other metric label | `UserInputError` — pass `descending` explicitly |

The last row covers custom evaluation labels (`metrics={"ic_nw": ic(...)}`) and
signed metrics such as `predictive_beta`, `fm_beta` and `spanning_alpha`, where
"largest positive value" is a different question from "strongest effect".

### Ties, order, and missing values

Rows sort on `sort_by`, then on `factor` and `forward_periods` ascending, then on
any `params` column of a sortable dtype, so the output does not depend on the
order of the input `results`. Rows equal on every one of those columns are
indistinguishable and keep input order among themselves; under `"min"` and
`"dense"` they carry the same rank anyway.

A `null` or `NaN` `sort_by` value sorts **last** in both directions and carries a
`null` rank under every `rank_method` — a row with no value has no place in the
ranking.

### Examples

`rank` is created after sorting, so it is not a valid `sort_by` key.

```python title="Illustrative"
# Alpha / information-ratio style metric: higher is better, resolved for you.
df = fx.compare(results, metrics=["ic", "ic_ir"], sort_by="ic_ir")

# Turnover: lower is better, resolved the same way.
df = fx.compare(results, metrics=["rank_turnover"], sort_by="rank_turnover")

# Significance screen; equally significant factors share one rank.
df = fx.compare(results, metrics=["ic"], sort_by="ic_p_value", rank_method="min")

# A custom label carries no direction, so state one.
df = fx.compare(results, metrics=["ic_nw"], sort_by="ic_nw", descending=True)
```

For signed metrics such as `predictive_beta`, sorting by the raw value answers
"largest positive effect", not "strongest evidence". A strongly negative but
highly significant factor ranks low under `sort_by="predictive_beta",
descending=True` — which is why those labels have no resolved direction and must
be given one. Use the p-value column for significance screens,
or sort on `abs(value)` in caller code when magnitude regardless of sign is the
intended ranking.
