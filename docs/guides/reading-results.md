---
title: Reading results
---

Each entry point in factrix returns a frozen result dataclass. This page
walks the field order for the three a quant researcher will encounter:

- [`FactorProfile`](../api/factor-profile.md) — what `evaluate()` returns
  for one factor (the primary inferential artifact).
- [`Survivors`](../api/multi-factor.md) — what `bhy()` /
  `partial_conjunction()` / `bhy_hierarchical()` return after False
  Discovery Rate (FDR) screening.
- [`MetricsBundle`](../api/metrics-bundle.md) — what `run_metrics()`
  returns (descriptive twin of `FactorProfile`).

The cheat sheet below is the scanning version; the sections after walk
each field with its reading rationale.

## Cheat sheet

| Result | First read | Then | Reference fields |
|---|---|---|---|
| `FactorProfile` | `primary_p` | `primary_stat` / `primary_stat_name` | `warnings` · `info_notes` · `stats` · `metadata` |
| `Survivors` | `survivor[i] iff adj_p[i] <= q` | `profiles[i].primary_p` alongside `adj_p[i]` | `expand_over` · `n_tests` |
| `MetricsBundle` | per-metric `.value` | `.significance` (`***` / `**` / `*` / `""`) | `.stat` · `.metadata` · `.n_obs` |

---

## `FactorProfile` — single-factor `evaluate()` result

```python
profile = fx.evaluate(panel, cfg)
```

Read fields in the order below — this matches `profile.diagnose()` key
order, which itself reflects the reader-flow from "what was tested" to
"what was the answer" to "what diagnostics support it".

### 1. Identity — what was tested

| Field | Type | Notes |
|---|---|---|
| `factor_id` | `str` | Stable id stamped from the factor column name; the hypothesis dimension that screening / `compare` group by. |
| `forward_periods` | `int` | Forward-return horizon. |
| `identity` | `tuple[str, int]` | `(factor_id, forward_periods)`; aligns with `MetricsBundle.identity` so downstream `compare()` can stack the two artifacts. |
| `context` | `Mapping[str, Any]` | Extra hypothesis-dimension keys (`regime_id`, `universe`, ...); empty by default, populated by upstream slicing. |
| `config.scope` / `config.signal` / `config.metric` / `mode` | enums | The four-axis dispatch coordinate that selected the procedure. |

### 2. Sample axes — was the test underpowered?

| Field | Type | Reads as |
|---|---|---|
| `n_obs` | `int` | Total observations the estimator consumed. |
| `n_pairs` | `int` | Cross-sectional pairs (per-date asset count summed). |
| `n_periods` | `int` | Number of dates. |
| `n_assets` | `int` | Unique asset count. |

Compare to the per-cell `MIN_*` floors in
[Reference § Sample-size constants](../reference/metric-applicability.md#sample-size-constants).

### 3. Primary significance — the canonical readout

| Field | Type | Notes |
|---|---|---|
| `primary_p` | `float` | The procedure-canonical p-value. The one number to read against the significance threshold. Read directly; do not rebuild it from `stats`. |
| `primary_stat` | `float \| None` | The paired test statistic. The type allows `None` for future procedures that report only a p-value; every shipped procedure today populates it. |
| `primary_stat_name` | `StatCode` | Which inference machinery produced the `(stat, p)` pair — `t_nw`, `t_clu`, `w_nw`, ... |

This trio is the single source of truth for "is this factor real?"
against the procedure's null hypothesis. See
[Architecture § Invariants](../development/architecture.md#invariants)
items 5 and 6 for the underlying contract: `primary_p` is a real
probability for every legal cell × mode and never auto-rebinds in
response to warnings.

### 4. Flag sets — interpretation risks

| Field | Type | Reads as |
|---|---|---|
| `warnings` | `frozenset[WarningCode]` | Recoverable risks that do **not** invalidate `primary_p` but flag it for caller-side judgment (`UNRELIABLE_SE_SHORT_PERIODS`, `PERSISTENT_REGRESSOR`, ...). The user decides whether to filter before screening. |
| `info_notes` | `frozenset[InfoCode]` | Informational; not actionable for pass/fail. |

Full enum and trigger conditions:
[Reference § Warning / info / stat codes](../reference/warning-codes.md).

### 5. Full stats / metadata

| Field | Type | Reads as |
|---|---|---|
| `stats` | `Mapping[StatCode, float]` | Every statistic the procedure produced (`mean`, `t_nw`, `p_nw`, `ic_ir`, ...). Use when a diagnostic beyond `primary_p` is needed. |
| `metadata` | `Mapping[StatCode, dict]` | Per-stat context (`nw_lags`, `ic_definition`, ...). |

The `profile.diagnose()` method returns a flat dict shaped in this same
reader order — see
[Quickstart § profile.diagnose() and warnings](../getting-started/quickstart.md#profilediagnose-and-warnings)
for a runnable example.

---

## `Survivors` — after Benjamini-Hochberg-Yekutieli (BHY) / partial conjunction / hierarchical FDR

```python
survivors = fx.multi_factor.bhy(profiles, q=0.05)
```

BHY (Benjamini-Hochberg-Yekutieli step-up procedure) controls the FDR
across a declared family of profiles.

### Contract

`survivor[i] iff adj_p[i] <= q`. This duality holds across every
screening function (`bhy` / `partial_conjunction` / `bhy_hierarchical`).
`profiles` and `adj_p` are aligned by input order: `profiles[i]` pairs
with `adj_p[i]`.

### Reading order

| Field | Type | Reads as |
|---|---|---|
| `adj_p[i]` | `np.ndarray[float]` | Bucket-local adjusted p-value for profile `i`; the survival readout. |
| `profiles[i].primary_p` | `float` | Underlying raw p-value, for comparing raw vs adjusted significance. |
| `q` | `float` | Family-wise FDR target the survivors were selected against. |
| `expand_over` | `tuple[str, ...]` | Context keys added as hypothesis dimensions for the screening; empty tuple if none. |
| `n_tests` | `Mapping[tuple, int]` | Per-bucket family size, keyed by `expand_over` values. |

### Anti-pattern

Do **not** re-run `bhy` on a filtered subset of survivors. FDR control
is a family-level property; it is not preserved across hand-picked
sub-families. See
[`multi_factor`](../api/multi-factor.md) for the pinned discussion of
sample-restriction vs hypothesis-dimension splits.

---

## `MetricsBundle` — descriptive twin of `FactorProfile`

```python
bundle = fx.run_metrics(panel, cfg)["factor"]
```

`run_metrics` fans out the cell's standalone descriptive metrics; the
bundle wraps the per-metric `MetricOutput` map.

### Reading order

| Field | Type | Reads as |
|---|---|---|
| `bundle.identity` | `tuple[str, int]` | `(factor_id, forward_periods)`; aligns with `FactorProfile.identity` for `compare(profiles, bundles)`. |
| `bundle.metrics["<name>"]` | `MetricOutput` | Per-metric record — see fields below. |
| `bundle.skipped` | `Mapping[str, str]` | Metrics that did not auto-run, with one-line reasons. |
| `bundle.context` | `Mapping[str, Any]` | Sample-restriction / conditioning dimensions; empty unless populated by `factrix.by_slice`. |

Each `MetricOutput` carries:

| Field | Type | Reads as |
|---|---|---|
| `name` | `str` | Metric identifier (`ic`, `ic_ir`, `oos_decay`, ...). |
| `value` | `float` | Headline number for the metric. |
| `significance` | `str \| None` | Procedure-dependent ladder: `***` / `**` / `*` / `""`. |
| `stat` | `float \| None` | Test statistic, where applicable. |
| `n_obs` | `int \| None` | Sample size the metric's estimator consumed. |
| `metadata` | `dict` | Per-metric context (`p_value`, `stat_type`, ...). |

A `MetricsBundle` does not carry a single `primary_p` because
"descriptive" rules out a single canonical inference target by design;
reach for `evaluate()` when a single p-value is what is needed.

---

## See also

- [`FactorProfile`](../api/factor-profile.md) / [`Survivors`](../api/multi-factor.md) / [`MetricsBundle`](../api/metrics-bundle.md) — full symbol references.
- [Quickstart § profile.diagnose() and warnings](../getting-started/quickstart.md#profilediagnose-and-warnings) — runnable end-to-end example.
- [Errors](../api/errors.md) — exception classes for the failure modes the result trio does not cover (`InsufficientSampleError`, `ModeAxisError`, ...).
- [Architecture § Invariants](../development/architecture.md#invariants) — `primary_p` semantic contract (items 5 and 6).
- [Batch screening with BHY](batch-screening.md) — `Survivors` lifecycle end-to-end.
- [Standalone metrics](standalone-metrics.md) — `MetricsBundle` composition patterns.
