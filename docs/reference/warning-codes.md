---
title: Warning codes
---

Structured `WarningCode` payloads attached to every
`EvaluationResult.warnings`. Use these as the SSOT
when you need to filter, route, or trigger downstream behaviour from
factrix output without parsing free-text strings.

- **`WarningCode`** — risk flags surfaced on `EvaluationResult.warnings`.
  Does **not** affect `MetricResult.p_value`;
  the user decides whether to pre-filter on warnings before
  multi-factor Benjamini-Hochberg-Yekutieli (BHY).

Each member's trigger / meaning is sourced from
`factrix._codes.WarningCode.description` (single source of truth, also
surfaced at runtime on each result's `warnings`). For the per-procedure
breakdown of which codes a given pipeline can emit, see
[Architecture § Procedure pipelines](../development/architecture.md#procedure-pipelines).

The scalar statistics that populate `EvaluationResult.metrics` are not
enum-keyed — each `MetricResult` exposes its statistic on
`MetricResult.stat` with `stat_type` / `h0` / `method` in
`MetricResult.metadata`; see [Stat keys by metric](stat-keys-by-metric.md).

## WarningCode

--8<-- "docs/reference/_generated_warning_codes.md"

::: factrix.WarningCode
    options:
      show_root_heading: false
      show_source: false
      members: false
