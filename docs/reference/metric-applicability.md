---
title: Metric applicability
---

!!! abstract "Answers"
    Can my data run this? — applicability gates and sample-size thresholds per metric.
    For computation pipeline (aggregation order, inference SE), see [Metric pipelines](metric-pipelines.md).
    For output schema and metadata keys, see [Stat keys by metric](stat-keys-by-metric.md).
    If you are still deciding *which* metric to use, see the task-oriented [Choosing a metric](../guides/choosing-metric.md) guide first.

A single matrix that maps each public metric to the analysis cell
where it is in scope, the canonical inferential role it plays in
that cell, and the sample-size threshold that gates it. Per-metric
formulae, parameters, and Notes / References live in the
[Metrics API pages](../api/metrics/index.md); this page is the
cross-metric overview. For the runtime API that returns the per-cell
metric list programmatically, see
[`list_metrics`](../api/list-metrics.md).

## Sample dimensions

factrix expresses sample size on three axes (see
[Concepts](../getting-started/concepts.md) for cell taxonomy):

- `N` — assets in the panel (`asset_id` unique count).
- `T` — date count (`date` unique count).
- `K` — non-zero event count for `Sparse` factors
  (`filter(factor != 0).height`).
- `T/h` — non-overlapping date count given
  `forward_periods = h`.

`PanelMode` is derived from `N` at evaluate-time: `PANEL` for `N ≥ 2`,
`TIMESERIES` for `N == 1`. The dispatch registry routes to the cell's
procedure in either PanelMode; the metric's applicability does not change
across Modes, only the sample axis that constrains it.

## Other metrics by family

Primary metrics (`ic`, `fm_beta`, `caar`, `ts_beta`) are the SSOT.
The remaining
metrics group by family below; the section heading carries the cell
context, so each per-row schema reduces to *Metric / Sample axis /
Min sample*. `MIN_*` constants resolve to values in the
[Sample-size constants table](#sample-size-constants).

### Information coefficient (IC) family — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`ic_newey_west`][factrix.metrics.ic.ic_newey_west] | `T` (full series) | `T ≥ MIN_ASSETS_PER_DATE_IC` |
| [`ic_ir`][factrix.metrics.ic.ic_ir] | `T` | `T ≥ MIN_ASSETS_PER_DATE_IC` |

### FM family — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`pooled_beta`][factrix.metrics.fm_beta.pooled_beta] | `N × T` | `N ≥ 10`, effective clusters `G ≥ 3` |
| [`beta_sign_consistency`][factrix.metrics.fm_beta.beta_sign_consistency] | `T` (β series) | `T ≥ MIN_FM_PERIODS_HARD` |

### Quantile / Monotonicity / Concentration — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | `T/h` | `T/h ≥ MIN_PORTFOLIO_PERIODS_HARD`; per-date `N ≥ n_groups` |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | `T/h` | as `quantile_spread` |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | `T/h` | per-date `N ≥ n_groups`; series `≥ MIN_MONOTONICITY_PERIODS` |
| [`top_concentration`][factrix.metrics.concentration.top_concentration] | `T/h` | `T/h ≥ MIN_PORTFOLIO_PERIODS_HARD`; warn if `T/h < MIN_PORTFOLIO_PERIODS_WARN` |

### Tradability — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`turnover`][factrix.metrics.tradability.turnover] | `T` | `T ≥ 2·forward_periods + 1` |
| [`notional_turnover`][factrix.metrics.tradability.notional_turnover] | `T` | as `turnover` |
| [`breakeven_cost`][factrix.metrics.tradability.breakeven_cost] | scalar | `notional_turnover > 0` |
| [`net_spread`][factrix.metrics.tradability.net_spread] | scalar | spread + cost provided |

### Event-quality — Cell: Individual × Sparse

| Metric | Sample axis | Min sample |
|---|---|---|
| [`bmp_test`][factrix.metrics.caar.bmp_test] | `K` | `K ≥ MIN_EVENTS_HARD`; `estimation_window` periods per asset |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`event_ic`][factrix.metrics.event_quality.event_ic] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | `K` | `K ≥ 2` |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | per-offset `K` | `K ≥ MIN_EVENTS_HARD` |
| [`mfe_mae_summary`][factrix.metrics.mfe_mae.mfe_mae_summary] | `K` | `K ≥ MIN_EVENTS_HARD`; `price` column required |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | `K`, `N` | `N ≥ 2`; `K ≥ MIN_EVENTS_HARD` |
| [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank] | `K` × estimation window | `K ≥ MIN_EVENTS_HARD`; per-asset `T ≥ 30` |

### TS-β family — Cell: Common × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`mean_r_squared`][factrix.metrics.ts_beta.mean_r_squared] | `N` | as `ts_beta` |
| [`ts_beta_sign_consistency`][factrix.metrics.ts_beta.ts_beta_sign_consistency] | `N` | `N ≥ 2` |
| [`ts_quantile_spread`][factrix.metrics.ts_quantile.ts_quantile_spread] | `T` | `T ≥ MIN_PORTFOLIO_PERIODS_HARD`; factor `n_unique ≥ n_groups × 2` |
| [`ts_asymmetry`][factrix.metrics.ts_asymmetry.ts_asymmetry] | `T` | factor has both signs (Gate B); each side `n_unique ≥ 2` for method B (Gate C) |

### Spread-series consumers — not cell-bound

| Metric | Sample axis | Min sample |
|---|---|---|
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | `T` | aligned spread series |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | `T` | as `spanning_alpha` |

### Series-tools — not cell-bound

| Metric | Sample axis | Min sample |
|---|---|---|
| [`hit_rate`][factrix.metrics.hit_rate.hit_rate] | series length | `T ≥ MIN_ASSETS_PER_DATE_IC` |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | `T` | `T ≥ 10` (literal floor) |
| [`oos_decay`][factrix.metrics.oos_decay.oos_decay] | `T` | `T ≥ 2 × MIN_OOS_PERIODS` |

## Sample-size constants
[](){ #sample-size-constants }

Every `MIN_*` constant referenced in the `Min sample` column above,
consolidated here so the value, axis, tier, and consumer set are
visible at one glance instead of three source files. The two-tier
`_HARD` / `_WARN` model and the descriptive-only single-tier rule
are described in [Sample-size sensitivity](#sample-size-sensitivity--the-two-tier-_hard--_warn-model)
below.

| Constant | Value | Axis | Tier | Source module | Used by |
|---|---|---|---|---|---|
| `MIN_ASSETS_PER_DATE_IC` | 10 | per-date `N` | hard | `factrix/_types.py` | `compute_ic` (drops dates with `N < 10`) → consumed by `ic`, `ic_newey_west`, `ic_ir`, `hit_rate` |
| `MIN_EVENTS_HARD` | 4 | `K` (event count) | hard | `factrix/_types.py` | `caar`, `bmp_test`, `event_hit_rate`, `event_ic`, `profit_factor`, `event_skewness`, `event_around_return`, `mfe_mae_summary`, `clustering_hhi`, `corrado_rank` |
| `MIN_EVENTS_WARN` | 30 | `K` | warn | `factrix/_types.py` | `caar` only (Brown-Warner literature floor; descriptive event-quality metrics use HARD only) |
| `MIN_OOS_PERIODS` | 5 | `T` (per split) | hard | `factrix/_types.py` | `oos_decay` (effective floor `T ≥ 2 × MIN_OOS_PERIODS = 10`) |
| `MIN_PORTFOLIO_PERIODS_HARD` | 3 | `T/h` | hard | `factrix/_types.py` | `quantile_spread`, `quantile_spread_vw`, `top_concentration`, `ts_quantile_spread`, `ts_asymmetry` |
| `MIN_PORTFOLIO_PERIODS_WARN` | 20 | `T/h` | warn | `factrix/_types.py` | `top_concentration` only (`quantile_spread` and the `ts_*` siblings are descriptive at the WARN tier and gate on HARD only) |
| `MIN_MONOTONICITY_PERIODS` | 5 | `T/h` | hard | `factrix/_types.py` | `monotonicity` |
| `MIN_PERIODS_HARD` | 20 | `T` (TIMESERIES) | hard | `factrix/_stats/constants.py` | TIMESERIES procedures (`individual_continuous` / `common_continuous` at `N == 1`); raises `InsufficientSampleError` |
| `MIN_PERIODS_WARN` | 30 | `T` (TIMESERIES) | warn | `factrix/_stats/constants.py` | same procedures; tags `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` |
| `MIN_ASSETS` | 10 | `N` | warn | `factrix/_stats/constants.py` | PANEL `common_continuous` and `suggest_config`; tags `WarningCode.SMALL_CROSS_SECTION_N` |
| `MIN_ASSETS_WARN` | 30 | `N` | warn | `factrix/_stats/constants.py` | same; tags `WarningCode.BORDERLINE_CROSS_SECTION_N` |
| `MIN_BROADCAST_EVENTS_HARD` | 5 | `K` (broadcast dummy) | hard | `factrix/_stats/constants.py` | `(COMMON, SPARSE, None, PANEL)` procedure |
| `MIN_BROADCAST_EVENTS_WARN` | 20 | `K` (broadcast dummy) | warn | `factrix/_stats/constants.py` | same; tags `WarningCode.SPARSE_COMMON_FEW_EVENTS` |
| `MIN_FM_PERIODS_HARD` | 4 | `T` (λ series) | hard | `factrix/metrics/fama_macbeth.py` | `fm_beta`, `beta_sign_consistency` |
| `MIN_FM_PERIODS_WARN` | 30 | `T` (λ series) | warn | `factrix/metrics/fama_macbeth.py` | `fm_beta` (Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) over-rejects below); ties to `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` |
| `MIN_TS_OBS` | 20 | `T` per asset | hard | `factrix/metrics/ts_beta.py` | `compute_ts_betas` (drops assets with `T < 20`); upstream of `ts_beta`, `mean_r_squared`, `ts_beta_sign_consistency` |

Naming caveats:

- `MIN_ASSETS_PER_DATE_IC` (10) and `MIN_ASSETS` (10) are different
  constants with the same value: the first gates **per-date** asset
  count for IC; the second gates **panel-wide** `N` for the
  cross-asset *t* on E[β]. The IC variant was renamed from
  `MIN_IC_PERIODS` because the historical "PERIODS" suffix did not
  match the per-date axis it actually checks.
- `MIN_ASSETS = 10` deliberately omits the `_HARD` suffix — the `N`
  axis only **warns** (small `N` is well-defined statistics, just
  weak), so the `_HARD` convention (which means "raise") would
  mislead.
- `MIN_BROADCAST_EVENTS_*` is named after its procedure domain to
  avoid colliding with `MIN_EVENTS_*` in `_types.py` (CAAR statistic
  vs broadcast-dummy regression — different gates).

For non-overlapping metrics (`ic`, `caar`, …) the effective floor is
`_scaled_min_periods(base, forward_periods)` (in
`factrix.metrics._helpers`), which scales the base constant by the
forward-return horizon `h`.

## Sample-size sensitivity — the two-tier `_HARD` / `_WARN` model

Inferential metrics enforce two separate floors:

- **`_HARD`** — the **mathematical floor**. Below it, the statistic is
  not defined (e.g. `t = (μ − 0) / (s / √n)` requires `n ≥ 2`; the FM
  small-sample HAC needs at least a few lagged covariances). `n < HARD`
  short-circuits to `MetricResult(value=NaN, metadata={"reason": ...})`.
- **`_WARN`** — the **literature / power floor**. The statistic *is*
  computable, but the SE is biased small or power is poor; the metric
  returns the stat, emits a `UserWarning`, and adds a `WarningCode` to
  `MetricResult.metadata["warning_codes"]` so `FactorProfile.warnings`
  can propagate it. `n ≥ WARN` is silent.

**Descriptive metrics** (`clustering_hhi`, `corrado_rank`,
`event_around_return`, `event_hit_rate`,
`event_ic`, `profit_factor`, `event_skewness`, `mfe_mae_summary`,
`quantile_spread`, `ts_quantile_spread`, `ts_asymmetry`, `bmp_test`)
enforce **`_HARD` only** — they have no formal H₀ under which power
can be characterised, so the literature `_WARN` tier is undefined
for them. They accept smaller-`n` inputs than the inferential
canonicals.

A few specific caveats worth flagging:

- **`MIN_FM_PERIODS_HARD = 4` / `MIN_FM_PERIODS_WARN = 30`** for
  `fm_beta`. `T = 4` is the math floor at which the NW HAC `t`
  is computable; the small-sample HAC is known to **over-reject**, so
  in `T ∈ [4, 30)` the metric emits
  `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` and treats *p*-values as
  anti-conservative. Forty periods is the practical floor in the
  panel-econometrics literature.
- **`MIN_EVENTS_HARD = 4` / `MIN_EVENTS_WARN = 30`** for the
  event-study CAAR `t`. Brown & Warner (1985) tabulate well-behaved
  power at `K ≥ 50` and use `K ≥ 30` as the conventional minimum; in
  `K ∈ [4, 30)` the parametric `caar` is under-powered and
  `WarningCode.FEW_EVENTS` fires. The `bmp_test` /
  `corrado_rank` siblings only partly mitigate.
- **`MIN_PORTFOLIO_PERIODS_HARD = 3` / `MIN_PORTFOLIO_PERIODS_WARN = 20`**
  in `top_concentration` and `ts_quantile_spread`. Below 3 there is
  no spread / concentration t to compute; in `[3, 20)` the metric
  returns the stat with `WarningCode.BORDERLINE_PORTFOLIO_PERIODS`.
  **`MIN_OOS_PERIODS = 5`** in `oos_decay` remains
  single-tier — the metric is now descriptive-only (no `p_value` in
  metadata), so a literature power floor is moot. Treat its output as
  descriptive; the formal inference reading should rely on the
  primary metric until the underlying series is materially longer.

## Below-threshold behaviour

When the input fails a sample threshold, factrix never silently
returns a meaningful-looking result. Three deterministic outcomes:

- **Short-circuit** — the metric returns
  `MetricResult(value=NaN, metadata={"reason": "..."})` and
  `FactorProfile.primary_p` is conservatively pinned to `1.0` so
  `primary_p` is at or above the user's chosen threshold.
- **Fallback** — the dispatch registry routes to a degraded but
  semantically distinct procedure (e.g. `Common × Continuous` at
  `N == 1` falls back to single-asset ordinary least squares (OLS) without HAC). `diagnose()`
  fires an `info`-severity rule recording the fallback.
- **Degraded** — the metric runs but a Profile field is set to
  `None` (e.g. `clustering_hhi=None` for single-asset event signals).
  `diagnose()` records the degradation.

Structural errors (wrong cell, missing column, `N == 1` on a cell that
requires `PANEL` DataStructure) raise `ValueError` / [`FactrixError`][factrix.FactrixError] rather than
falling back.

## Event-study contracts

The Individual × Sparse cell hosts a family of metrics whose
abnormal-return definitions, estimation windows, and overlap
conventions diverge from each other by design. Surfacing the contracts
here so each metric page can reference one canonical definition.

### Abnormal-return definition per metric

factrix follows MacKinlay (1997) event-window vocabulary but each
metric instantiates the abnormal-return primitive differently:

| Metric | Per-row primitive | Why this form |
|---|---|---|
| [`caar`][factrix.metrics.caar.caar], [`bmp_test`][factrix.metrics.caar.bmp_test] | `signed_car = forward_return × factor` (magnitude preserved) | Generalises MacKinlay's signed CAAR to continuous factors (Sefcik-Thompson 1986 lineage); on `factor ∈ {0, ±1}` it reduces to the textbook signed CAAR. |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate], [`event_ic`][factrix.metrics.event_quality.event_ic], [`profit_factor`][factrix.metrics.event_quality.profit_factor], [`event_skewness`][factrix.metrics.event_quality.event_skewness] | `signed_car = forward_return × sign(factor)` (sign-only) | These metrics measure direction quality independent of factor magnitude; magnitude-weighting would conflate "direction was right" with "magnitude was big". |
| [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank] | `signed_rank = uniform_rank(forward_return) × sign(factor)` | Corrado (1989) ranks the raw return distribution, then direction-adjusts the rank. The sign-adjustment is on the rank, not the return. |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | Post-event (k > 0): `sign(factor) × cumulative_return`; pre-event (k < 0): unsigned single-bar return | Asymmetric on purpose: post-event reads signal *quality*, pre-event reads *leakage* — leakage is independent of eventual direction and must be inspected unsigned. |

The shared function "abnormal return" therefore covers four different
estimators. Use the table above when comparing factrix output to
literature numbers.

### CAR vs BHAR

factrix computes **CAR** (cumulative abnormal return — sum of per-period
abnormal returns) throughout. **BHAR** (buy-and-hold abnormal return —
compounded `∏(1 + r) − 1` minus benchmark) is **not** computed by any
metric. Two implications:

- The `caar` *t*-test is the cross-event mean of per-event-date CAAR,
  not BHAR. CAR is appropriate for short-horizon windows (the bias
  between CAR and BHAR is `O(h²)` for horizon `h`); for multi-year
  buy-and-hold studies, compute BHAR externally.
- `event_around_return` post-event uses simple cumulative returns
  (`price[t+1+k] / price[t+1] − 1`), not period-by-period sums. This is
  arithmetic accumulation, distinct from BHAR's geometric compounding.

### `estimation_window`

The estimation window is the per-asset pre-event sample used to fit
the abnormal-return baseline. factrix uses it in two places:

- [`bmp_test`][factrix.metrics.caar.bmp_test]: standardises each
  event's abnormal return by the event's own pre-event SE, computed
  over the estimation window.
- [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank]:
  ranks the abnormal return against the per-asset distribution drawn
  from the estimation window.

Conventions:

- **Length**: per-asset `T ≥ 30` non-event observations is the
  literature default (Brown-Warner 1985 §2.B). factrix enforces
  this floor at the `corrado_rank` per-asset gate.
- **Alignment**: ends one period before the event date; gap-before
  -event of zero (no skip period). Users running a skip-period
  convention must pre-shift the panel.
- **Overlap exclusion**: factrix's primitives do **not** drop
  pre-event windows that overlap an earlier event for the same asset.
  In practice this means contaminated estimation windows for clustered
  events; use `clustering_hhi` to gauge severity and consider
  pre-filtering the panel for tightly clustered names.
- **Forward-return horizon**: the event window is `forward_periods`
  bars; the estimation window is the **pre-event** sample, so
  `EventConfig.forward_periods` does not affect estimation-window
  length.

### Confounded-event handling

When two events for the same `asset_id` fall within each other's
forward window, factrix's procedures **do not deduplicate or skip**
the inner event. The chosen mitigation depends on the metric:

| Metric | Behaviour under within-asset overlap |
|---|---|
| [`caar`][factrix.metrics.caar.caar] | Per-event-date CS-mean is computed first, then NW HAC is applied to the calendar-time CAAR series. The `forward_periods − 1` floor on the lag (Hansen-Hodrick 1980) absorbs MA(h−1) overlap structure. Within-asset clustering on the same date inflates the per-date variance; the period reindex + HAC handles the time-axis component but not the asset-axis component. |
| [`bmp_test`][factrix.metrics.caar.bmp_test] | The Kolari-Pynnönen adjustment (`kolari_pynnonen_adjust=True`) corrects the BMP statistic for cross-sectional dependence on the same event date. It does **not** correct same-asset event clustering. |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate], [`event_ic`][factrix.metrics.event_quality.event_ic] | Each event row is counted independently; same-asset overlapping events double-contribute to the binomial / Spearman statistic. The null implicitly assumes independence — under heavy clustering the variance is understated. |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | Same: each `(asset, event_date)` row is independent in the binomial null at every offset. Adjacent-offset hit rates are also serially correlated within the same event (k=6 and k=12 share the t+1 entry price), which the binomial null does not adjust for. |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | Quantifies cross-sectional concentration on event dates only. Does not detect within-asset temporal clustering — pair with `signal_density` for the asset-axis view. |

Operationally: trust `caar` *p*-values when `clustering_hhi`
Herfindahl-Hirschman index (HHI) is low and `signal_density` shows events well-spaced per asset;
otherwise downweight the parametric *p* and lean on `corrado_rank`
or external block-bootstrap.
