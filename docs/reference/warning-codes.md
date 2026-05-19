---
title: Warning, info, and stat codes
---

Structured enum payloads attached to every
`FactorProfile`. Use these as the SSOT
when you need to filter, route, or trigger downstream behaviour from
factrix output without parsing free-text strings.

The three enums:

- **`WarningCode`** — risk flags surfaced by `profile.diagnose()`.
  Does **not** affect `primary_p`;
  the user decides whether to pre-filter on warnings before
  multi-factor Benjamini-Hochberg-Yekutieli (BHY).
- **`InfoCode`** — information-severity diagnose annotations (e.g.
  scope-axis collapsed under `Mode = TIMESERIES`).
- **`StatCode`** — canonical names for the scalar statistics that
  populate `FactorProfile.metrics`.

Each member's trigger / meaning is sourced from
`factrix._codes.<Code>.description` (single source of truth, also
returned at runtime by `profile.diagnose()`). For the per-procedure
breakdown of which codes a given pipeline can emit, see
[Architecture § Procedure pipelines](../development/architecture.md#procedure-pipelines).

## WarningCode

--8<-- "docs/reference/_generated_warning_codes.md"

::: factrix.WarningCode
    options:
      show_root_heading: false
      show_source: false
      members: false

## InfoCode

--8<-- "docs/reference/_generated_info_codes.md"

::: factrix.InfoCode
    options:
      show_root_heading: false
      show_source: false
      members: false

## StatCode

--8<-- "docs/reference/_generated_stat_codes.md"

::: factrix.StatCode
    options:
      show_root_heading: false
      show_source: false
      members: false
