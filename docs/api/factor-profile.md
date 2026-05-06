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

`n_obs` reports the **second-stage** sample size — the denominator
behind the inference test that produces `primary_p`. It is never raw
`N × T`. factrix procedures aggregate in two stages
([Architecture § Terminology — aggregation regime](../development/architecture.md#terminology--aggregation-regime)):
the first stage reduces the panel to a per-date or per-asset series,
the second stage runs the inferential test over that series.

Concretely, an IC PANEL run with `T = 30` dates and `N = 500` assets
per date has `n_obs = 30` (the per-date IC series feeding the NW HAC
`t`-test), not `15000`. Treating `n_obs` as a raw cell count would
overstate degrees of freedom by orders of magnitude — the warning
guards (`MIN_PERIODS_HARD`, `UNRELIABLE_SE_SHORT_PERIODS`) and the
`primary_p` itself read the second-stage value.

#### `n_assets` is the raw panel width

`n_assets = panel["asset_id"].n_unique()` is computed once on the
input and has fixed semantics across cells — the cross-section width
of the union over the sample period. It is the test denominator
**only** in cells where the second stage is a cross-asset aggregation
(`(common, continuous, None, panel)`, `(common, sparse, None, panel)`);
elsewhere it is metadata for warnings and routing, not for inference.

#### Consumers

| Consumer | Reads `n_obs` | Reads `n_assets` |
|---|---|---|
| `profile.diagnose()` payload | yes | yes |
| `MIN_PERIODS_HARD` / `UNRELIABLE_SE_SHORT_PERIODS` guards | yes | — |
| `MIN_ASSETS` guards (`SMALL_CROSS_SECTION_N`, `BORDERLINE_CROSS_SECTION_N`) | — | yes |
| `InsufficientSampleError.actual_periods` | yes | — |
| `verdict()` | — | — |
| `multi_factor.bhy` family partition | — | — |

Neither `verdict()` nor `multi_factor.bhy` reads these fields:
`verdict()` thresholds on `primary_p`; BHY partitions families on
`(dispatch cell, forward horizon)` and runs step-up on p-values.

#### Why one polymorphic `n_obs` instead of split `n_periods` + `n_cs`

Every registered cell runs **one** primary test with **one**
denominator. Splitting `n_obs` into `n_periods` and `n_cs` would
force every consumer (`diagnose()` printers, warning-code interpreters,
`InsufficientSampleError` recovery code) to dispatch on cell to pick
which field to read — and the field that matters is always the test
denominator. The polymorphic-`n_obs` design surfaces that field
directly; the per-cell semantic table above resolves "which axis am
I looking at" in one place when needed. `n_assets` stays as a raw
panel descriptor for the cross-asset cells and for `MIN_ASSETS`
guards.

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

`profile.stats` is populated by **one path only**: the procedure that
ran inside `evaluate()`. The keys above are the full enumeration —
nothing else is auto-merged.

| Path | Lives in | What it produces | Pluggable? |
|---|---|---|---|
| **Procedure-internal** | `factrix/_stats/` helpers (`_newey_west_t_test`, `_adf`, `_ljung_box_p`, …) invoked from `factrix/_procedures.py` | The `StatCode` keys listed above on `profile.stats` | No — the per-cell stat set is hard-coded by the registered procedure. |
| **Standalone metrics** | `factrix/metrics/*.py`, listed by [`list_metrics`](list-metrics.md) | A separate [`MetricOutput`](metric-output.md) per call, returned to the user | Yes — call any number after `evaluate()` returns. |

The user-invoked path is **independent**: a call like
`fl.metrics.quantile_spread(...)` returns a `MetricOutput` to the
caller. It does not mutate `profile.stats`. To layer follow-up metrics
into a single agent-readable payload, the caller assembles them
explicitly (typically a `dict` of `MetricOutput` keyed by metric name).

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
