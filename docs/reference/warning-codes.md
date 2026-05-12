# Warning, info, and stat codes

Structured enum payloads attached to every
[`FactorProfile`](../api/factor-profile.md). Use these as the SSOT
when you need to filter, route, or trigger downstream behaviour from
factrix output without parsing free-text strings.

The three enums:

- **`WarningCode`** — risk flags surfaced by `profile.diagnose()`.
  Does **not** affect `primary_p`;
  the user decides whether to pre-filter on warnings before
  multi-factor BHY.
- **`InfoCode`** — information-severity diagnose annotations (e.g.
  scope-axis collapsed under `Mode = TIMESERIES`).
- **`StatCode`** — canonical names for the scalar statistics that
  populate `FactorProfile.metrics`.

For the per-metric mapping of `WarningCode` to the procedure that
emits it, see
[Architecture § Procedure pipelines](../development/architecture.md#procedure-pipelines).

## WarningCode

::: factrix.WarningCode

## InfoCode

::: factrix.InfoCode

## StatCode

::: factrix.StatCode
