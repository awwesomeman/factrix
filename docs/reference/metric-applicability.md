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
cross-metric overview. For the runtime API that returns the family-grouped
metric catalog programmatically, see
[`list_metrics`](../api/metrics/index.md#factrix.list_metrics); for
structure-aware per-panel applicability, use
[`inspect_data`](../api/inspect-data.md).

## Sample dimensions

factrix expresses sample size on three axes (see
[Concepts](../getting-started/concepts.md) for cell taxonomy):

- `n_assets` — assets in the panel (`asset_id` unique count).
- `T` — date count (`date` unique count).
- `K` — non-zero event count for `Sparse` factors
  (`filter(factor != 0).height`).
- `T/h` — non-overlapping date count given
  `forward_periods = h`.

`DataStructure` is derived from `n_assets` at evaluate-time: `PANEL` for
`n_assets >= 2`, `TIMESERIES` for `n_assets == 1`. Each metric's `MetricSpec` declares the cell it
applies to; the metric's applicability does not change across structures, only
the sample axis that constrains it.

## Other metrics by family

The inferential entry points (`ic`, `fm_beta`, `caar`, `ts_beta`) are the SSOT.
The remaining
metrics group by family below; the section heading carries the cell
context, so each per-row schema reduces to *Metric / Sample axis /
Min sample*. `MIN_*` constants resolve to values in the
[Sample-size constants table](#sample-size-constants).

### Information coefficient (IC) family — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`ic_ir`][factrix.metrics.ic.ic_ir] | `T` | `T ≥ MIN_PERIODS_HARD`; warn if `T < MIN_PERIODS_WARN` |

### FM family — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`pooled_beta`][factrix.metrics.fm_beta.pooled_beta] | `n_assets × T` | `n_assets >= 10`, effective clusters `G >= 3` |
| [`fm_beta_sign_consistency`][factrix.metrics.fm_beta.fm_beta_sign_consistency] | `T` (β series) | `T ≥ MIN_FM_PERIODS_HARD` |

### Quantile / Monotonicity / Concentration — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`quantile_spread`][factrix.metrics.quantile.quantile_spread] | `T/h` | `T/h >= MIN_PORTFOLIO_PERIODS_HARD`; per-date `n_assets >= n_groups` |
| [`quantile_spread_vw`][factrix.metrics.quantile.quantile_spread_vw] | `T/h` | as `quantile_spread` |
| [`k_spread`][factrix.metrics.k_spread.k_spread] | `T/h` | `T/h >= MIN_PORTFOLIO_PERIODS_HARD`; per-date `n_assets >= 2 * k` |
| [`monotonicity`][factrix.metrics.monotonicity.monotonicity] | `T/h` | per-date `n_assets >= n_groups`; series `>= MIN_MONOTONICITY_PERIODS_HARD` |
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
| [`bmp_z`][factrix.metrics.caar.bmp_z] | `K` | `K ≥ MIN_EVENTS_HARD`; `estimation_window` periods per asset |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`event_ic`][factrix.metrics.event_quality.event_ic] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`profit_factor`][factrix.metrics.event_quality.profit_factor] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`event_skewness`][factrix.metrics.event_quality.event_skewness] | `K` | `K ≥ MIN_EVENTS_HARD` |
| [`signal_density`][factrix.metrics.event_quality.signal_density] | `K` | `K ≥ 2` |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | per-offset `K` | `K ≥ MIN_EVENTS_HARD` |
| [`mfe_mae`][factrix.metrics.mfe_mae.mfe_mae] | `K` | `K ≥ MIN_EVENTS_HARD`; `price` column required |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | `K`, `n_assets` | `n_assets >= 2`; `K >= MIN_EVENTS_HARD` |
| [`corrado_rank`][factrix.metrics.corrado_rank.corrado_rank] | `K` × estimation window | `K ≥ MIN_EVENTS_HARD`; per-asset `T ≥ 30` |

### TS-β family — Cell: Common × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`mean_r_squared`][factrix.metrics.ts_beta.mean_r_squared] | `n_assets` | `n_assets >= 1` |
| [`ts_beta_sign_consistency`][factrix.metrics.ts_beta.ts_beta_sign_consistency] | `n_assets` | `n_assets >= 2` |
| [`ts_quantile_spread`][factrix.metrics.ts_quantile.ts_quantile_spread] | `T` | `T ≥ MIN_PORTFOLIO_PERIODS_HARD`; factor `n_unique ≥ n_groups × 2` |
| [`ts_asymmetry`][factrix.metrics.ts_asymmetry.ts_asymmetry] | `T` | factor has both signs; each side `n_unique ≥ 2` for method B |

### Single-asset dense — Cell: Timeseries × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`predictive_beta`][factrix.metrics.predictive_beta.predictive_beta] | `T` | `T ≥ MIN_PERIODS_HARD`; warn if `T < MIN_PERIODS_WARN` |

### Spread-series consumers — not cell-bound

| Metric | Sample axis | Min sample |
|---|---|---|
| [`spanning_alpha`][factrix.metrics.spanning.spanning_alpha] | `T` | aligned spread series |
| [`greedy_forward_selection`][factrix.metrics.spanning.greedy_forward_selection] | `T` | as `spanning_alpha` |

### IC-series diagnostics — Cell: Individual × Continuous

| Metric | Sample axis | Min sample |
|---|---|---|
| [`hit_rate`][factrix.metrics.hit_rate.hit_rate] | series length | `T ≥ MIN_SERIES_PERIODS_HARD` |
| [`ic_trend`][factrix.metrics.trend.ic_trend] | `T` | `T ≥ 10` (literal floor) |
| [`oos_decay`][factrix.metrics.oos_decay.oos_decay] | `T` | `T ≥ 2 × MIN_OOS_PERIODS_HARD` |

### Directional sign diagnostics — not cell-bound

| Metric | Sample axis | Min sample |
|---|---|---|
| [`directional_hit_rate`][factrix.metrics.directional_hit_rate.directional_hit_rate] | pooled `(date, asset)` signs | non-overlapping obs `≥ MIN_DIRECTIONAL_PAIRS_HARD`; warn if below `MIN_DIRECTIONAL_PAIRS_WARN` |

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
| `MIN_IC_ASSETS_HARD` | 2 | per-date `n_assets` | hard | `factrix/_types.py` | `compute_ic` (drops dates with pairwise-complete `n_assets < 2`) → consumed by `ic`, `ic_ir` |
| `MIN_IC_ASSETS_WARN` | 10 | per-date `n_assets` | warn | `factrix/_types.py` | `ic`, `ic_ir`, `inspect_data`; tags `WarningCode.FEW_ASSETS` when retained IC dates have `n_assets < 10` |
| `MIN_SERIES_PERIODS_HARD` | 10 | `T` | hard | `factrix/_types.py` | `ic` post-stride sampled IC series, `hit_rate`, and series-mean non-overlap pre-flight |
| `MIN_DIRECTIONAL_PAIRS_HARD` | 10 | pooled pairs | hard | `factrix/_types.py` | `directional_hit_rate` |
| `MIN_DIRECTIONAL_PAIRS_WARN` | 30 | pooled pairs | warn | `factrix/_types.py` | `directional_hit_rate`; tags `WarningCode.FEW_DIRECTIONAL_PAIRS` |
| `MIN_EVENTS_HARD` | 4 | `K` (event count) | hard | `factrix/_types.py` | `caar`, `bmp_z`, `event_hit_rate`, `event_ic`, `profit_factor`, `event_skewness`, `event_around_return`, `mfe_mae`, `clustering_hhi`, `corrado_rank` |
| `MIN_EVENTS_WARN` | 30 | `K` | warn | `factrix/_types.py` | `caar` only (Brown-Warner literature floor; descriptive event-quality metrics use HARD only) |
| `MIN_OOS_PERIODS_HARD` | 5 | `T` (per split) | hard | `factrix/_types.py` | `oos_decay` (effective floor `T ≥ 2 × MIN_OOS_PERIODS_HARD = 10`) |
| `MIN_PORTFOLIO_PERIODS_HARD` | 3 | `T/h` | hard | `factrix/_types.py` | `quantile_spread`, `quantile_spread_vw`, `top_concentration`, `ts_quantile_spread`, `ts_asymmetry` |
| `MIN_PORTFOLIO_PERIODS_WARN` | 20 | `T/h` | warn | `factrix/_types.py` | `top_concentration` only (`quantile_spread` and the `ts_*` siblings are descriptive at the WARN tier and gate on HARD only) |
| `MIN_MONOTONICITY_PERIODS_HARD` | 5 | `T/h` | hard | `factrix/_types.py` | `monotonicity` |
| `MIN_PERIODS_HARD` | 20 | `T` | hard | `factrix/_stats/constants.py` | Shared hard floor for HAC / time-series inference |
| `MIN_PERIODS_WARN` | 30 | `T` | warn | `factrix/_stats/constants.py` | Shared warn floor for HAC / time-series inference; tags `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` |
| `MIN_ASSETS_WARN` | 30 | `n_assets` | warn | `factrix/_stats/constants.py` | PANEL `common_continuous`; tags `WarningCode.FEW_ASSETS` (severity from `n_assets`) |
| `MIN_FM_ASSETS_HARD` | 3 | per-date `n_assets` | hard | `factrix/metrics/_primitives/_fm_betas.py` | `compute_fm_betas` (drops dates with pairwise-complete `n_assets < 3`) -> consumed by `fm_beta`, `fm_beta_sign_consistency` |
| `MIN_FM_ASSETS_WARN` | 10 | per-date `n_assets` | warn | `factrix/metrics/_primitives/_fm_betas.py` | `fm_beta`, `fm_beta_sign_consistency`, `inspect_data`; tags `WarningCode.FEW_ASSETS` when retained FM dates have `n_assets < 10` |
| `MIN_FM_PERIODS_HARD` | 4 | `T` (λ series) | hard | `factrix/metrics/fm_beta.py` | `fm_beta`, `fm_beta_sign_consistency` |
| `MIN_FM_PERIODS_WARN` | 30 | `T` (λ series) | warn | `factrix/metrics/fm_beta.py` | `fm_beta` (Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) over-rejects below); ties to `WarningCode.UNRELIABLE_SE_SHORT_PERIODS` |
| `MIN_TS_PERIODS_HARD` | 20 | `T` per asset | hard | `factrix/metrics/_primitives/_ts_betas.py` | `compute_ts_betas` (drops assets with `T < 20`); upstream of `ts_beta`, `mean_r_squared`, `ts_beta_sign_consistency` |

Naming caveats:

- `MIN_IC_ASSETS_HARD` (2) gates the **per-date** pairwise-complete
  asset count for IC (dates below it are dropped). `MIN_IC_ASSETS_WARN`
  (10) is the reliability floor: dates still run, but `ic` / `ic_ir`
  surface `WarningCode.FEW_ASSETS`. Both are distinct from the
  panel-wide `n_assets` cross-asset guard, which lives solely on
  `MIN_ASSETS_WARN`.
- `MIN_FM_ASSETS_HARD` (3) gates the **per-date** pairwise-complete
  asset count for FM per-date OLS. `MIN_FM_ASSETS_WARN` (10) mirrors the
  IC thin-cross-section tier: dates still run, but `fm_beta` /
  `fm_beta_sign_consistency` surface `WarningCode.FEW_ASSETS`.
- `MIN_ASSETS_WARN = 30` is a single warn floor (no `_HARD`) — the `n_assets`
  axis only **warns** (small `n_assets` is well-defined statistics, just
  weak), so the `_HARD` convention (which means "raise") would mislead;
  severity scales with `n_assets` rather than splitting into tiers.
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
  `MetricResult.metadata["warning_codes"]` so `EvaluationResult.warnings`
  can propagate it. `n ≥ WARN` is silent.

**Descriptive metrics** (`clustering_hhi`, `corrado_rank`,
`event_around_return`, `event_hit_rate`,
`event_ic`, `profit_factor`, `event_skewness`, `mfe_mae`,
`quantile_spread`, `ts_quantile_spread`, `ts_asymmetry`, `bmp_z`)
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
  `WarningCode.FEW_EVENTS` fires. The `bmp_z` /
  `corrado_rank` siblings only partly mitigate.
- **`MIN_PORTFOLIO_PERIODS_HARD = 3` / `MIN_PORTFOLIO_PERIODS_WARN = 20`**
  in `top_concentration` and `ts_quantile_spread`. Below 3 there is
  no spread / concentration t to compute; in `[3, 20)` the metric
  returns the stat with `WarningCode.BORDERLINE_PORTFOLIO_PERIODS`.
  **`MIN_OOS_PERIODS_HARD = 5`** in `oos_decay` remains
  single-tier — the metric is now descriptive-only (no `p_value` in
  metadata), so a literature power floor is moot. Treat its output as
  descriptive; the formal inference reading should rely on the
  mainstream metric until the underlying series is materially longer.

## Below-threshold behaviour

When the input fails a sample threshold, factrix never silently
returns a meaningful-looking result. The outcome is deterministic and
keyed to the tier the input falls below:

- **Hard floor** — the metric short-circuits to
  `MetricResult(value=NaN, metadata={"reason": "..."})`. Its `p_value`
  is conservatively set to `1.0` so a downstream screening pass treats
  the short-circuit as non-significant rather than a false discovery,
  and the `NaN` value makes the data shortage impossible to misread as
  a valid zero.
- **Warn floor** — the metric is computable but under-powered: it
  returns the statistic, emits a `UserWarning`, and attaches a
  `WarningCode` that the DAG executor lifts into
  `EvaluationResult.warnings`, so the interpretation risk is auditable.
- **Upstream unavailable** — a metric whose required upstream producer
  short-circuited is itself skipped with a `NaN` `MetricResult` +
  `WarningCode.UPSTREAM_UNAVAILABLE`, rather than being run on missing
  data.

Structural errors (wrong cell, missing column, `n_assets == 1` on a metric
whose `MetricSpec` cell requires `PANEL`) raise
[`FactrixError`][factrix.FactrixError] (e.g. `IncompatibleAxisError`)
under `strict=True` rather than short-circuiting.

## Event-study contracts

The Individual × Sparse cell hosts a family of metrics whose
abnormal-return definitions, estimation windows, and overlap
conventions diverge from each other by design. Surfacing the contracts
here so each metric page can reference one canonical definition.

For sparse event factors, the non-zero sign encodes the **expected return
direction**. It is not necessarily the raw event type. A raw taxonomy such as
`hike=+1` / `cut=-1` should be mapped into the asset's expected bullish/bearish
direction before entering the metrics. Magnitude may carry event strength for
metrics that preserve magnitude, but the sign should already mean "positive
return expected" vs "negative return expected".

### Abnormal-return definition per metric

factrix follows MacKinlay (1997) event-window vocabulary but each
metric instantiates the abnormal-return primitive differently:

| Metric | Per-row primitive | Why this form |
|---|---|---|
| [`caar`][factrix.metrics.caar.caar], [`bmp_z`][factrix.metrics.caar.bmp_z] | `signed_car = forward_return × factor` (magnitude preserved) | Generalises MacKinlay's signed CAAR to continuous factors (Sefcik-Thompson 1986 lineage); on `factor ∈ {0, ±1}` it reduces to the textbook signed CAAR. |
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

- [`bmp_z`][factrix.metrics.caar.bmp_z]: standardises each
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
| [`caar`][factrix.metrics.caar.caar] | Per-event-date CS-mean is computed first, then a calendar-aware non-overlap subsample keeps event dates at least `forward_periods` calendar periods apart before the t-test. This avoids overlap-induced dependence while preserving the event-only mean; dense zero-fill and NW HAC are not used on this path. Within-asset clustering can still make event rows dependent, so read the vanilla t-test cautiously when event calendars are crowded. |
| [`bmp_z`][factrix.metrics.caar.bmp_z] | The Kolari-Pynnönen adjustment (`kolari_pynnonen_adjust=True`) corrects the BMP statistic for cross-sectional dependence on the same event date. It does **not** correct same-asset event clustering. |
| [`event_hit_rate`][factrix.metrics.event_quality.event_hit_rate], [`event_ic`][factrix.metrics.event_quality.event_ic] | Each event row is counted independently; same-asset overlapping events double-contribute to the binomial / Spearman statistic. The null implicitly assumes independence — under heavy clustering the variance is understated. |
| [`event_around_return`][factrix.metrics.event_horizon.event_around_return] | Same: each `(asset, event_date)` row is independent in the binomial null at every offset. Adjacent-offset hit rates are also serially correlated within the same event (k=6 and k=12 share the t+1 entry price), which the binomial null does not adjust for. |
| [`clustering_hhi`][factrix.metrics.clustering_hhi.clustering_hhi] | Quantifies cross-sectional concentration on event dates only. Does not detect within-asset temporal clustering — pair with `signal_density` for the asset-axis view. |

Operationally: trust `caar` *p*-values when `clustering_hhi`
Herfindahl-Hirschman index (HHI) is low and `signal_density` shows events well-spaced per asset;
otherwise downweight the parametric *p* and lean on `corrado_rank`
or external block-bootstrap.
