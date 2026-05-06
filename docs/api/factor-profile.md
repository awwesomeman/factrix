# FactorProfile

Procedure-canonical analysis result for a single factor. Every
registered procedure produces an instance of this dataclass with
cell-specific scalars keyed in the `stats` mapping; adding a new
metric does not grow the schema.

See [Metric applicability](../reference/metric-applicability.md) for `n_obs`
and `n_assets` thresholds per procedure.

::: factrix.FactorProfile
    options:
      show_root_heading: false
      members:
        - verdict
        - diagnose

## `diagnose()` return schema

`profile.diagnose()` returns a flat `dict[str, Any]` for human triage
and AI-agent consumption. The top-level keys are stable across every
registered cell; the `stats` sub-dict varies by procedure.

### Top-level keys

| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `mode` | `str` | `profile.mode.value` | `"panel"` or `"timeseries"` |
| `n_obs` | `int` | `profile.n_obs` | Cell-canonical effective sample size — see [§ `n_obs` semantics](#n_obs-semantics-by-cell) |
| `n_assets` | `int` | `profile.n_assets` | `panel["asset_id"].n_unique()` of the input |
| `primary_p` | `float` | `profile.primary_p` | Procedure-canonical p-value used by `verdict()` and `multi_factor.bhy` |
| `warnings` | `list[str]` | `sorted(w.value for w in profile.warnings)` | Sorted [`WarningCode`](../reference/warning-codes.md#warningcode) string values |
| `info_notes` | `list[str]` | `sorted(i.value for i in profile.info_notes)` | Sorted [`InfoCode`](../reference/warning-codes.md#infocode) string values |
| `stats` | `dict[str, float]` | `{k.value: v for k, v in profile.stats.items()}` | [`StatCode`](../reference/warning-codes.md#statcode) string keys; per-cell content varies |

The dict is JSON-serialisable as long as `stats` values are plain
`float` (procedures populate it that way; downstream wrappers are
expected to preserve this). Calling `json.dumps(profile.diagnose())`
on any registered cell's output is supported.

### `n_obs` semantics by cell

`n_obs` is the *cell-canonical* sample size — the denominator behind
`primary_p`. It is **not** always `len(panel)`:

| Dispatch cell | `n_obs` is |
|---------------|------------|
| `(individual, continuous, ic, panel)` | `T` — number of dates contributing to the per-date IC series |
| `(individual, continuous, fm, panel)` | `T` — number of dates with a valid OLS slope |
| `(individual, sparse, None, panel)` | `T` — densified panel-period count (unique dates in the panel) after CAAR event-date back-fill |
| `(common, continuous, None, panel)` | `N` — number of assets entering the cross-asset t-test on `E[β]` |
| `(common, sparse, None, panel)` | `N` |
| `(common, continuous, None, timeseries)` | `T` — single-series sample length |
| `(*, sparse, None, timeseries)` | `T` — period count of the dummy regression |

Reading `n_obs` together with `n_assets` disambiguates whether a small
`n_obs` came from a short series (low `T`) or a thin cross-section
(low `N`).

### `stats` keys by cell

Every cell populates the keys for its primary statistic (`*_MEAN`,
`*_T_NW`, `*_P`) plus diagnostic keys specific to its procedure. Keys
appear in `stats` as `StatCode.value` strings (e.g. `"ic_mean"`).

| Dispatch cell | `stats` keys populated |
|---------------|------------------------|
| `(individual, continuous, ic, panel)` | `IC_MEAN`, `IC_T_NW`, `IC_P`, `NW_LAGS_USED` |
| `(individual, continuous, fm, panel)` | `FM_LAMBDA_MEAN`, `FM_LAMBDA_T_NW`, `FM_LAMBDA_P`, `NW_LAGS_USED` |
| `(individual, sparse, None, panel)` | `CAAR_MEAN`, `CAAR_T_NW`, `CAAR_P`, `NW_LAGS_USED` |
| `(common, continuous, None, panel)` | `TS_BETA`, `TS_BETA_T_NW`, `TS_BETA_P`, `FACTOR_ADF_P` |
| `(common, sparse, None, panel)` | `TS_BETA`, `TS_BETA_T_NW`, `TS_BETA_P` |
| `(common, continuous, None, timeseries)` | `TS_BETA`, `TS_BETA_T_NW`, `TS_BETA_P`, `FACTOR_ADF_P`, `NW_LAGS_USED` |
| `(*, sparse, None, timeseries)` (sentinel) | `TS_BETA`, `TS_BETA_T_NW`, `TS_BETA_P`, `LJUNG_BOX_P`, `EVENT_TEMPORAL_HHI`, `NW_LAGS_USED` |

`NW_LAGS_USED` appears whenever a procedure aggregates a time series
through a Newey-West HAC kernel (Bartlett, Newey-West 1994 auto-lag
with Hansen-Hodrick overlap floor). Cross-asset aggregations
(`common_*` PANEL) use a plain cross-section t-test and therefore
have no `NW_LAGS_USED` entry.

`FACTOR_ADF_P` is a CONTINUOUS-only persistence diagnostic — sparse
cells skip it because the `{0, R}` event-trigger signal (zero on
non-event entries) makes the unit-root null degenerate.

### Example

A worked `diagnose()` call with rendered output lives in
[Quickstart § `profile.diagnose()` and warnings](../getting-started/quickstart.md#profilediagnose-and-warnings).
For reference, the JSON shape on the IC PANEL cell is:

```json
{
  "mode": "panel",
  "n_obs": 494,
  "n_assets": 100,
  "primary_p": 2.13e-40,
  "warnings": [],
  "info_notes": [],
  "stats": {
    "ic_mean": 0.0722,
    "ic_t_nw": 14.60,
    "ic_p": 2.13e-40,
    "nw_lags_used": 5.0
  }
}
```

For the meaning of each `StatCode` see the
[`StatCode` reference](../reference/warning-codes.md#statcode); for
`WarningCode` / `InfoCode` triggers see
[Architecture § Procedure pipelines](../development/architecture.md#procedure-pipelines).
