---
title: factrix.multi_factor.bhy_across_metrics
---

::: factrix.multi_factor.bhy_across_metrics

<hr>

## When to use it

Use `bhy_across_metrics` when the research process may select any tested
factor × metric cell. Unlike [`bhy`](bhy.md), which deliberately runs one
screen per metric label, this function flattens all declared labels into one
Benjamini-Hochberg-Yekutieli family.

```python title="Illustrative"
screen = fx.multi_factor.bhy_across_metrics(
    results,
    metrics=["ic", "spread"],
    q=0.05,
)

screen.family_size   # {(): n_results * 2} under the default policy
screen.family        # declared / computed / inactive / unsubmitted / adjusted + policy
screen.to_frame()    # factor | metric | p_value | adj_p | survived | active
```

The survivor unit is one `(EvaluationResult, metric label)` hypothesis. Taking
the unique factor names afterward does **not** provide factor-level FDR control.
For the claim that a factor works on at least `k` predeclared endpoints, use
[`partial_conjunction_across_metrics`](partial-conjunction-across-metrics.md).

## Family and audit contract

- `metrics` must contain at least two unique inferential labels. A descriptive
  output with `p_value=None` fails loudly.
- Entry order is result-major, then caller-supplied metric order.
- `expand_over` retains the existing `bhy` meaning: it partitions by a result
  field or `params` key; metric labels remain pooled inside each bucket.
- An inactive cell (`insufficient_*` or `degenerate_variance`) stays in
  `entries` with `active=False`. Under the default `inactive_policy="count"`
  it stays in `family_size` at an inert `p = 1`; under `"exclude"` it leaves
  the family and carries `adj_p=NaN` — see [inactive
  candidates](multi-factor.md#inactive-candidates).
- Other missing or invalid p-values raise rather than silently changing the
  family.

## Result fields

| Field | Meaning |
|---|---|
| `entries` | Every traceable `MetricHypothesis`, including inactive and eliminated cells |
| `adj_p_all` | BHY-adjusted p-values aligned with `entries`; NaN for a cell that entered no family |
| `survivors` / `adj_p` | Passing cell hypotheses and their adjusted p-values |
| `metrics` | Metric labels in declared order |
| `expand_over` | Keys partitioning separately reported families |
| `family_size` | Factor × metric family size per bucket — the `m` each step-up ran on, including declared-unsubmitted cells |
| `family` | Declared / computed / inactive / unsubmitted / adjusted counts and the `inactive_policy` used |

::: factrix.multi_factor.CrossMetricBhyResult
    options:
      show_root_toc_entry: false
      heading_level: 3

::: factrix.multi_factor.MetricHypothesis
    options:
      show_root_toc_entry: false
      heading_level: 3
