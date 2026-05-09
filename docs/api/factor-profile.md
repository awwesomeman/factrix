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
| `identity` | `dict[str, Any]` | `{"factor_id", "forward_periods"}` | Hypothesis tuple — see [Identity vs context](identity.md) |
| `context` | `dict[str, Any]` | `dict(profile.context)` | Sample-restriction dimensions (`universe_id` / `regime_id` / ...) |
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
| `(individual, continuous, ic, panel)` | number of dates contributing to the per-date IC series |
| `(individual, continuous, fm, panel)` | number of dates with a valid OLS slope |
| `(individual, sparse, None, panel)` | densified panel-period count (unique dates in the panel) after CAAR event-date back-fill |
| `(common, continuous, None, panel)` | number of assets entering the cross-asset t-test on `E[β]` |
| `(common, sparse, None, panel)` | number of assets entering the cross-asset event-dummy t-test |
| `(common, continuous, None, timeseries)` | single-series sample length |
| `(*, sparse, None, timeseries)` | period count of the dummy regression |

Reading `n_obs` together with `n_assets` disambiguates whether a small
`n_obs` came from a short series (low `T`) or a thin cross-section
(low `N`).

#### Inference-stage denominator, not raw `N × T`

`n_obs` is the second-stage denominator behind `primary_p`, never raw
`N × T` (factrix procedures aggregate in two stages — see
[Architecture § Terminology — aggregation regime](../development/architecture.md#terminology--aggregation-regime)).
An IC PANEL run with `T = 30` and `N = 500` has `n_obs = 30` (the
per-date IC series feeding the NW HAC `t`-test), not `15000`. The
`MIN_PERIODS_HARD` / `UNRELIABLE_SE_SHORT_PERIODS` guards and
`primary_p` itself read the second-stage value.

`n_assets` is the raw panel width (`panel["asset_id"].n_unique()`)
with fixed semantics across cells. It is a **sample-period union**,
not a per-date count, so it does not bias `primary_p` for time-series-
axis tests where the cross-section is not the test denominator.

In cross-asset cells (`(common, *, None, panel)`) the cross-asset
t-test reads its own inference-stage `N` — the count of assets
surviving the `compute_ts_betas` per-asset filter (≥`MIN_TS_OBS`
non-null observations) — which can be materially smaller than the
union when assets enter the panel late or with sparse history. The
`SMALL_CROSS_SECTION_N` / `BORDERLINE_CROSS_SECTION_N` guards in
`_compute_common_panel` threshold on that filtered N (matching the
test's actual `dof = N - 1`); reading the warning together with
`primary_p` is the canonical signal. Reading `n_assets` alone is
optimistic in this corner — `suggest_config` mirrors the same
pre-filter so its preview warning agrees with what `evaluate()` will
emit.

#### Consumers

| Consumer | Reads `n_obs` | Reads `n_assets` |
|---|---|---|
| `profile.diagnose()` payload | yes | yes |
| `MIN_PERIODS_HARD` / `UNRELIABLE_SE_SHORT_PERIODS` guards | yes | — |
| `MIN_ASSETS` guards (`SMALL_CROSS_SECTION_N`, `BORDERLINE_CROSS_SECTION_N`) | — | yes |
| `InsufficientSampleError.actual_periods` | yes | — |
| `verdict()` | — | — |
| `multi_factor.bhy` family partition | — | — |

`verdict()` thresholds on `primary_p`; BHY partitions on
`(dispatch cell, forward horizon)` and runs step-up on p-values —
neither path reads these fields.

#### Why one polymorphic `n_obs` instead of split `n_periods` + `n_cs`

Every registered cell runs one primary test with one denominator.
Splitting `n_obs` would force every consumer (`diagnose()` printers,
warning-code interpreters, `InsufficientSampleError` recovery) to
dispatch on cell to pick which field is the test denominator. The
polymorphic field surfaces it directly; the per-cell table above
resolves the axis when needed.

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

### `stats` provenance — two paths

`profile.stats` is populated only by the procedure that ran inside
`evaluate()`; the keys above are the full enumeration.

| Path | Lives in | What it produces | Pluggable? |
|---|---|---|---|
| **Procedure-internal** | `factrix/_stats/` helpers (`_newey_west_t_test`, `_adf`, `_ljung_box_p`, …) invoked from `factrix/_procedures.py` | The `StatCode` keys listed above on `profile.stats` | No — the per-cell stat set is hard-coded by the registered procedure. |
| **Standalone metrics** | `factrix/metrics/*.py`, listed by [`list_metrics`](list-metrics.md) | A separate [`MetricOutput`](metric-output.md) per call, returned to the user | Yes — call any number after `evaluate()` returns. |

`fl.metrics.quantile_spread(...)` and friends return a `MetricOutput`
to the caller; they do not mutate `profile.stats`.

See also [Stat keys by metric](../reference/stat-keys-by-metric.md)
for the per-metric `MetricOutput.metadata` schema (primary vs
auxiliary keys) — the standalone-metrics analogue of this section's
per-cell `StatCode` table.

#### `StatCode` → statistical method

Each procedure-internal `StatCode` maps to one section of
[Statistical methods](../reference/statistical-methods.md):

| `StatCode` | Method |
|---|---|
| `IC_MEAN`, `IC_T_NW`, `IC_P` | [HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns) — Newey-West HAC `t` on the per-date IC series |
| `FM_LAMBDA_MEAN`, `FM_LAMBDA_T_NW`, `FM_LAMBDA_P` | [HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns) — NW HAC `t` on the per-date Fama-MacBeth λ series |
| `CAAR_MEAN`, `CAAR_T_NW`, `CAAR_P` | [HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns) — non-overlapping `t` / BMP `z` on the per-event-date CAAR series |
| `TS_BETA`, `TS_BETA_T_NW`, `TS_BETA_P` | [HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns) — cross-asset `t` on `E[β]` (PANEL) or NW HAC single-series `t` (TIMESERIES) |
| `NW_LAGS_USED` | [HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns) — Newey-West 1994 auto-lag with Hansen-Hodrick overlap floor |
| `FACTOR_ADF_P` | [Persistence diagnostics under near-unit-root predictors](../reference/statistical-methods.md#4-persistence-diagnostics-under-near-unit-root-predictors) — ADF unit-root test on the continuous factor |
| `LJUNG_BOX_P` | [Architecture § Procedure pipelines](../development/architecture.md#-sparse---n1-ts-dummy--time-series-only) — Ljung-Box on the TS-dummy single-asset residual |
| `EVENT_TEMPORAL_HHI` | [Architecture § Procedure pipelines](../development/architecture.md#-sparse---n1-ts-dummy--time-series-only) — Herfindahl concentration of event dates over the calendar grid |

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
