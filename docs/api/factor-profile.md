# FactorProfile

Procedure-canonical analysis result for a single factor. Every
registered procedure produces an instance of this dataclass with
cell-specific scalars keyed in the `stats` mapping; adding a new
metric does not grow the schema.

See [Metric applicability](../reference/metric-applicability.md) for
`n_obs` and `n_assets` thresholds per procedure.

::: factrix.FactorProfile
    options:
      show_root_heading: false
      members:
        - diagnose

## `diagnose()` return schema

`profile.diagnose()` returns a flat `dict[str, Any]` for human triage
and AI-agent consumption. The top-level keys are stable across every
registered cell; the `stats` sub-dict varies by procedure.

### Top-level keys

Key order follows the reader-flow seven questions (#246): hypothesis →
context → dispatch cell → sample axes → primary significance → flag
sets → raw stats / metadata.

| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `identity` | `dict[str, Any]` | `{"factor_id", "forward_periods"}` | Hypothesis tuple — see [Identity vs context](identity.md) |
| `context` | `dict[str, Any]` | `dict(profile.context)` | Sample-restriction dimensions (`universe_id` / `regime_id` / ...) |
| `cell` | `dict[str, Any]` | `{"scope", "signal", "metric", "mode"}` | Dispatch coordinate — the four axes that selected the procedure |
| `n_obs` | `int` | `profile.n_obs` | Cell-canonical final-stage test denominator — see [§ sample axes](#sample-axes-by-cell) |
| `n_pairs` | `int` | `profile.n_pairs` | Non-null (period, asset) pair count in the raw panel — first-stage observation count |
| `n_periods` | `int` | `profile.n_periods` | Unique periods in the raw panel (calendar time, any-non-null union) |
| `n_assets` | `int` | `profile.n_assets` | Unique assets in the raw panel (any-non-null union); `1` for single-asset TIMESERIES |
| `primary_p` | `float` | `profile.primary_p` | Procedure-canonical p-value consumed by `multi_factor.bhy` |
| `primary_stat` | `float \| None` | `profile.primary_stat` | Test statistic paired with `primary_p` (`None` for no-test-stat primaries like block-bootstrap p) |
| `primary_stat_name` | `str` | `profile.primary_stat_name.value` | `stats`-key slug for the primary statistic (e.g. `"t_nw"`) — the dataclass field is `StatCode`, serialised here to its `.value`. Invariant: `stats[primary_stat_name] == primary_stat` when not `None` |
| `warnings` | `list[str]` | `sorted(w.value for w in profile.warnings)` | Sorted [`WarningCode`](../reference/warning-codes.md#warningcode) string values |
| `info_notes` | `list[str]` | `sorted(i.value for i in profile.info_notes)` | Sorted [`InfoCode`](../reference/warning-codes.md#infocode) string values |
| `stats` | `dict[str, float]` | `{k.value: v for k, v in profile.stats.items()}` | [`StatCode`](../reference/warning-codes.md#statcode) string keys; per-cell content varies |

The dict is JSON-serialisable as long as `stats` values are plain
`float` (procedures populate it that way; downstream wrappers are
expected to preserve this). Calling `json.dumps(profile.diagnose())`
on any registered cell's output is supported.

### Sample axes by cell

Four sample-size axes sit at the top level of `diagnose()`. Each
answers one question and never overlaps with another (#246):

| Axis | Question it answers | Definition |
|------|---------------------|------------|
| `n_obs` | How many observations did the test see? | Cell-canonical final-stage test denominator |
| `n_pairs` | How dense is the panel? | Non-null `(period, asset)` pair count (first-stage) |
| `n_periods` | How long is the time axis? | Unique periods in the raw panel, any-non-null union |
| `n_assets` | How wide is the cross-section? | Unique assets in the raw panel, any-non-null union |

Derived quantities:

- **Sparsity numerator**: `n_pairs / (n_periods * n_assets)`
- **Raw envelope**: `n_periods * n_assets`
- **Test-axis identification**: compare `n_obs` against the three
  envelope axes — for IC / FM PANEL `n_obs == n_periods` (test axis is
  time); for COMMON PANEL the cross-asset test reports `n_obs == N`
  where `N` is the filtered cross-section that survived
  `compute_ts_betas`' `MIN_TS_OBS` filter.

#### `n_obs` per cell

| Dispatch cell | `n_obs` is |
|---------------|------------|
| `(individual, continuous, ic, panel)` | number of dates contributing to the per-date IC series |
| `(individual, continuous, fm, panel)` | number of dates with a valid OLS slope |
| `(individual, sparse, None, panel)` | densified panel-period count (unique dates in the panel) after CAAR event-date back-fill |
| `(common, continuous, None, panel)` | number of assets entering the cross-asset t-test on `E[β]` |
| `(common, sparse, None, panel)` | number of assets entering the cross-asset event-dummy t-test |
| `(common, continuous, None, timeseries)` | single-series sample length |
| `(*, sparse, None, timeseries)` | period count of the dummy regression |

#### Definition boundaries

The four axes carry the following invariants — pinned here so future
cell registrations cannot quietly drift the semantics:

1. **`n_obs` is the final-stage test denominator** for `primary_p` —
   sample-length after procedure-internal trimming (winsorize /
   outlier drop / min-period filter), not before.
2. **`n_obs` does not deduct effective DoF** — autocorrelation
   adjustments (NW HAC effective n, overlapping-window inflation)
   live inside `stats` / `metadata` for the estimator that uses
   them, not in `n_obs`.
3. **`n_pairs` is the first-stage observation count** — non-null
   `(period, asset)` rows entering the cell before any second-stage
   aggregation. Always `n_pairs >= n_obs`.
4. **`n_periods` / `n_assets` use the any-non-null union** — counts a
   period (resp. asset) if any of `factor` / `forward_return` on that
   period (resp. asset) is non-null. Calendar time, not event time.
5. **`n_obs = 0` is a legal degenerate value**, not an exception —
   `primary_p` is `NaN` and the relevant `WarningCode` fires.
6. **`n_obs` is paired with `primary_*`, not with secondary `stats`
   entries** — e.g. ADF run on the factor surfaces its own sample
   size inside `stats[FACTOR_ADF_*]` / `metadata[FACTOR_ADF_*]`.
7. **MetricOutput sample-count is per-primitive, not per-cell** —
   `factrix.metrics.*` primitives carry their own `n_obs` (the count
   the metric primitive saw). It is the same family name as
   `FactorProfile.n_obs` but a different scope: per-metric estimator
   vs. cell-canonical final-stage test.

#### Inference-stage denominator, not raw envelope

`n_obs` reflects the sample the **primary estimator actually saw**,
not the raw `n_periods * n_assets` rectangle. An IC PANEL run with
`n_periods = 30` and `n_assets = 500` has `n_obs = 30` (the per-date
IC series feeding the NW HAC `t`-test), not `15000` — that
panel-envelope total is recoverable as `n_periods * n_assets` for
callers who want it.

In cross-asset cells (`(common, *, None, panel)`) the cross-asset
t-test reads its own inference-stage `N` — the count of assets
surviving the `compute_ts_betas` per-asset filter
(≥ `MIN_TS_OBS` non-null observations) — which can be materially
smaller than `n_assets` when assets enter the panel late or with
sparse history. The `SMALL_CROSS_SECTION_N` /
`BORDERLINE_CROSS_SECTION_N` guards in `_compute_common_panel`
threshold on the filtered `N` (matching the test's actual
`dof = n_obs - 1`); reading the warning together with `primary_p`
is the canonical signal. `suggest_config` mirrors the same
pre-filter so its preview warning agrees with what `evaluate()`
will emit.

#### Consumers

| Consumer | Reads `n_obs` | Reads `n_assets` |
|---|---|---|
| `profile.diagnose()` payload | yes | yes |
| `MIN_PERIODS_HARD` / `UNRELIABLE_SE_SHORT_PERIODS` guards | yes | — |
| `MIN_ASSETS` guards (`SMALL_CROSS_SECTION_N`, `BORDERLINE_CROSS_SECTION_N`) | — | yes |
| `InsufficientSampleError.actual_periods` | yes | — |
| `multi_factor.bhy` family partition | — | — |

BHY partitions on `(dispatch cell, forward horizon)` and runs step-up
on p-values — it does not read the sample axes.

#### Why surface `n_obs` alongside `n_pairs` + envelope axes

The earlier design exposed `n_obs` alone (polymorphic by cell). The
polymorphism survives — `n_obs` still means different things across
cells — but is anchored by three companion axes so the reader never
has to reverse-engineer which axis a small `n_obs` came from:

- `n_pairs` separates panel sparsity from envelope shrinkage.
- `n_periods` and `n_assets` give the envelope shape directly.
- The per-cell table above identifies which axis `n_obs` lives on.

The cost is four columns instead of one; the win is that an AI agent
or new user reading `diagnose()` cold can answer "is this small `n`
from a short series, a thin cross-section, or a sparse panel?"
without consulting docs.

### `stats` keys by cell

After #187's flattening, every cell populates the same primary keys
(`MEAN`, `T_NW`, `P_NW`); cell identity lives on `profile.config`
(`scope` / `signal` / `metric`), so the StatCode no longer encodes it.
Diagnostic keys carry an explicit `FACTOR_` / `RESID_` / `EVENT_`
prefix because their target sits outside `config`. Keys appear in
`stats` as `StatCode.value` strings (e.g. `"mean"`, `"factor_adf_p"`).

| Dispatch cell | `stats` keys populated |
|---------------|------------------------|
| `(individual, continuous, ic, panel)` | `MEAN`, `T_NW`, `P_NW` |
| `(individual, continuous, fm, panel)` | `MEAN`, `T_NW`, `P_NW` |
| `(individual, sparse, None, panel)` | `MEAN`, `T_NW`, `P_NW` |
| `(common, continuous, None, panel)` | `MEAN`, `T_NW`, `P_NW`, `FACTOR_ADF_TAU`, `FACTOR_ADF_P` |
| `(common, sparse, None, panel)` | `MEAN`, `T_NW`, `P_NW` |
| `(common, continuous, None, timeseries)` | `MEAN`, `T_NW`, `P_NW`, `FACTOR_ADF_TAU`, `FACTOR_ADF_P` |
| `(*, sparse, None, timeseries)` (sentinel) | `MEAN`, `T_NW`, `P_NW`, `RESID_LJUNG_BOX_Q`, `RESID_LJUNG_BOX_P`, `EVENT_HHI_VALUE` |

`FACTOR_ADF_*` is a CONTINUOUS-only persistence diagnostic — sparse
cells skip it because the `{0, R}` event-trigger signal (zero on
non-event entries) makes the unit-root null degenerate.

### `profile.metadata` — hyperparameter records

`profile.metadata: Mapping[StatCode, Mapping[str, Any]]` mirrors
`stats`: for any populated stat, the same key in `metadata` returns
the inner dict of hyperparameters that produced it. Stats with no
hyperparameter (`MEAN`) are absent rather than mapping to `{}`. Tests
that share a hyperparameter populate the inner dict under each key
the test produced.

| Cell | Populated `metadata` keys | Inner dict |
|---|---|---|
| IC / FM / CAAR PANEL | `T_NW`, `P_NW` | `{"nw_lags": <resolved bandwidth>}` |
| `(common, continuous, None, panel)` | `FACTOR_ADF_TAU`, `FACTOR_ADF_P` | `{"lag_order": 0}` |
| `(common, continuous, None, timeseries)` | `T_NW`, `P_NW`, `FACTOR_ADF_TAU`, `FACTOR_ADF_P` | NW `nw_lags` + ADF `lag_order` |
| `(*, sparse, None, timeseries)` | `T_NW`, `P_NW`, `RESID_LJUNG_BOX_Q`, `RESID_LJUNG_BOX_P`, `EVENT_HHI_VALUE` | NW `nw_lags` + Ljung-Box `lag_h` + HHI `n_bins` |
| `(common, sparse, None, panel)` | (none — cross-asset t has no hyperparam) | — |

`profile.diagnose()["metadata"]` serialises with `StatCode.value`
strings as outer keys (e.g. `"p_nw"`) and plain dicts inside. Reading
order pattern: `profile.stats[StatCode.P_NW]` for the value, then
`profile.metadata[StatCode.P_NW]` for "how was this computed".

### `stats` provenance — two paths

`profile.stats` is populated only by the procedure that ran inside
`evaluate()`; the keys above are the full enumeration.

| Path | Lives in | What it produces | Pluggable? |
|---|---|---|---|
| **Procedure-internal** | `factrix/_stats/` helpers (`_newey_west_t_test`, `_adf`, `_ljung_box`, …) invoked from `factrix/_procedures.py` | The `StatCode` keys listed above on `profile.stats` | No — the per-cell stat set is hard-coded by the registered procedure. |
| **Standalone metrics** | `factrix/metrics/*.py`, listed by [`list_metrics`](list-metrics.md) | A separate [`MetricOutput`](metric-output.md) per call, returned to the user | Yes — call any number after `evaluate()` returns. |

`fx.metrics.quantile_spread(...)` and friends return a `MetricOutput`
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
| `MEAN`, `T_NW`, `P_NW` | [HAC SE under overlapping returns](../reference/statistical-methods.md#1-hac-se-under-overlapping-returns) — Newey-West HAC `t` on the cell primary series (IC mean / FM λ / CAAR / E[β] / β); convention selected by the procedure dispatched for `profile.config` |
| `P_HH` | Reserved (#184) — Hansen-Hodrick rectangular-kernel HAC p-value |
| `P_GMM` | Reserved — Hansen (1982) GMM J-test p-value |
| `FACTOR_ADF_TAU`, `FACTOR_ADF_P` | [Persistence diagnostics under near-unit-root predictors](../reference/statistical-methods.md#4-persistence-diagnostics-under-near-unit-root-predictors) — ADF τ statistic and unit-root p-value on the continuous factor |
| `RESID_LJUNG_BOX_Q`, `RESID_LJUNG_BOX_P` | [Architecture § Procedure pipelines](../development/architecture.md#-sparse---n1-ts-dummy--time-series-only) — Ljung-Box Q statistic and p-value on the TS-dummy single-asset residual |
| `EVENT_HHI_VALUE` | [Architecture § Procedure pipelines](../development/architecture.md#-sparse---n1-ts-dummy--time-series-only) — Herfindahl concentration of event dates over the calendar grid |

### Example

A worked `diagnose()` call with rendered output lives in
[Quickstart § `profile.diagnose()` and warnings](../getting-started/quickstart.md#profilediagnose-and-warnings).
For reference, the JSON shape on the IC PANEL cell is:

```json
{
  "identity": {"factor_id": "momentum_12_1", "forward_periods": 5},
  "context": {},
  "cell": {
    "scope": "individual",
    "signal": "continuous",
    "metric": "ic",
    "mode": "panel"
  },
  "n_obs": 494,
  "n_pairs": 49400,
  "n_periods": 494,
  "n_assets": 100,
  "primary_p": 2.13e-40,
  "primary_stat": 14.60,
  "primary_stat_name": "t_nw",
  "warnings": [],
  "info_notes": [],
  "stats": {
    "mean": 0.0722,
    "t_nw": 14.60,
    "p_nw": 2.13e-40
  }
}
```

For the meaning of each `StatCode` see the
[`StatCode` reference](../reference/warning-codes.md#statcode); for
`WarningCode` / `InfoCode` triggers see
[Architecture § Procedure pipelines](../development/architecture.md#procedure-pipelines).
