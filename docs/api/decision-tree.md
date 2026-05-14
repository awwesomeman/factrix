---
title: Cross-function reference
---

Function semantics that do not belong on any single function's page —
the comparison-axes matrix (which `compare` / `by_slice` /
`slice_*_test` to pick), and cross-function topics where the same
keyword shifts meaning across functions (`expand_over`, regime
analysis).

For *"which function do I reach for given my research question"*, the
[API reference landing](index.md) is the SSOT: the **Function flow**
graph shows pipeline-level relationships, **Typical patterns** maps
goals to pipelines, and the **Entry points** table carries a *When to
read* line per function. Per-function docstrings carry the same
*When to use* summary at the head of each page.

Two sections:

- **Comparison axes** — `row × column` matrix for every "compare N
  things across M dimensions" question.
- **Cross-function topics** — concepts whose meaning shifts across
  functions (`expand_over`, regime analysis).

---

## Comparison axes

Every "compare N things across M dimensions" question reduces to a
choice of *row variable* × *column variable*. The seven shipped
comparisons:

| You want to compare | Function | Row | Column |
|---|---|---|---|
| N factors × 1 primary metric | `compare(profiles)` | factor | primary stat |
| N factors × M descriptive metrics | `compare(bundles)` | factor | each standalone metric |
| N factors × 1 primary, post-FDR | `compare(survivors)` | survivor | primary stat + `adj_q` |
| N factors × M estimators *(planned v0.14)* | `by_estimator(profiles, estimators=…)` | factor | each estimator's (t, p) |
| 1 metric × K slices (descriptive) | `by_slice(metric, df, label=…)` | slice value | metric value + n_obs |
| 1 metric × K slices (pairwise tests) | `slice_pairwise_test(metric, df, label=…)` | slice pair | test statistic + adjusted p |
| 1 metric × K slices (omnibus test) | `slice_joint_test(metric, df, label=…)` | — | one χ² + p |

Read the matrix as: row × column = the cells of the returned
`pl.DataFrame`. None of these functions recompute the underlying
metric — they are projections over artefacts the compute functions
(`evaluate` / `run_metrics` / a metric callable) already produced.
The exception is `by_estimator`, which re-runs the inference layer
only, not the metric pipeline.

---

## Cross-function topics

### `expand_over` is not one concept

The keyword is reused across screening functions but the **what is
expanded** differs. Confusing the three semantics is the most common
multiplicity-correction bug in user code.

| Function | `expand_over` meaning | What ends up in the false discovery rate (FDR) family |
|---|---|---|
| `bhy(profiles, expand_over=["regime_id"])` | Add the listed context keys as hypothesis dimensions | `factor_id × forward_periods × regime_id` — every (factor, horizon, regime) row is an independent hypothesis |
| `partial_conjunction(profiles, k_of_m, expand_over=["regime_id"])` | Group rows by everything *except* the listed keys; require k-of-m passes within each group | Each non-expanded group yields one partial-conjunction p; Benjamini-Yekutieli (BHY) then runs over the resulting reduced family |
| `by_estimator` *(planned)* | Does not accept `expand_over` | — |

Two practical rules:

- **Universe / regime as sample restriction** — `filter(...)` then
  `bhy(...)`. Standard FDR over `factor_id × forward_periods`.
- **Universe / regime as a hypothesis dimension** —
  `bhy(profiles, expand_over=[axis])`. FDR over `factor_id × forward_periods × axis`.

See [`multi_factor`](multi-factor.md) for the pinned discussion that
walks through the sample-restriction-vs-hypothesis-dimension split.

### Regime analysis — four functions for four questions

"I want to analyse regimes" is a leading question — the right function
depends on what specifically is being asked.

| Question | Function | Why |
|---|---|---|
| "What does factor X look like in each regime?" (descriptive) | `by_slice(metric, df, label="regime_id")` → `SliceResult` | Single-axis slice surface; no inference claim |
| "Which (factor, regime) cells are FDR-significant?" | `bhy(profiles, expand_over=["regime_id"])` | regime is a hypothesis dimension; FDR scope expands accordingly |
| "Is factor X consistent across all (or k of) regimes?" | `partial_conjunction(profiles, k_of_m=k, expand_over=["regime_id"])` | The PC null is "fails in ≥ k regimes"; BHY then on the reduced PC p-values |
| "Do regime means differ statistically?" | `slice_pairwise_test(metric, df, label="regime_id")` or `slice_joint_test(...)` | Between-slice inference; family-internal MTC; no cell-level FDR claim |

Three notes:

- **Lookahead bias on ex-post regime labels.** If the regime label is
  built using future information (e.g. "high-vol regime" defined by
  realised vol over the window), every result above inherits that
  bias. Cleanest fix: pre-publish the regime rule and apply it with no
  lookahead; otherwise treat the analysis as descriptive only.
- **Effect-size comparison across regimes** is not a separate function
  — it is a paper-craft question. Quote the relevant per-regime
  effect sizes from `by_slice` output and reason in prose; do not add
  a `compare(across="regime")` knob to `compare`.
- **`by_estimator` does not interact with regimes.** It re-runs the
  inference layer for a fixed cell; it has no regime axis.

---

## See also

- [API reference landing](index.md) — function-centric overview and function-flow graph
- [Concepts](../getting-started/concepts.md) — three-axis taxonomy
- [Quickstart § Next steps](../getting-started/quickstart.md#next-steps) — exit ramps from a single `FactorProfile`
- [Errors](errors.md) — error code → fix mapping
