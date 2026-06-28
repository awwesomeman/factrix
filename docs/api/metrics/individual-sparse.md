---
title: Individual sparse
---

Metrics for the `Individual × Sparse` cell — event signals where most
`(date, asset_id)` cells carry `factor == 0` and the few non-zero cells
mark events. The schema is `{0, R}` — zero on non-event entries,
any real value otherwise (`R` is unrestricted — positive, negative,
or any magnitude). Common forms: `{0, 1}` for a pure event flag, or
`{0, R}` for any real-valued magnitude; both flow through (see
[`caar`](caar.md) for the input-form table).

Sparse routing is zero-value based. Null factor cells are missing values and
do not count as non-events. Fill missing upstream rows to `0` only when that is
the research contract; otherwise leave them null and let pairwise sample
counts reflect the missing factor. To use these sparse event metrics on a
continuous exposure, categorical regime, or raw event taxonomy, first map the
event-of-interest upstream into an explicit `{0, R}` event column.

Below 50% zeros is not a hard refusal when that zero state exists: automatic
discovery stays dense, but an explicitly requested sparse metric runs with a
`frequent_event_signal` warning.

The non-zero sign is interpreted as the **expected return direction**.
For raw event taxonomies such as `hike=+1` / `cut=-1`, map the event type into
an asset-specific bullish/bearish signal before running sparse metrics. If a
hike is expected to hurt duration assets but help a currency basket, the two
asset groups should receive opposite factor signs on the same policy date.

At `n_assets == 1`, sparse factors still use the event-density path. Most
sparse metrics whose `MetricSpec` leaves `DataStructure` wildcarded run
directly on the single asset's event axis; metrics that need an asset
cross-section, such as `clustering_hhi`, remain unavailable. Do not pool
unrelated assets just to clear an asset-count guard: that changes the
return-generating process being tested.

Always-in-market `{-1, +1}` signals are not sparse event signals because there
is no non-event zero state. Treat them as dense directional signals: use
`predictive_beta` for the single-asset magnitude slope and
`directional_hit_rate` for sign prediction.

## Event semantics

| Signal / question | Metric family | Contract |
|---|---|---|
| `{0, 1}` event flag | `caar`, `bmp_z`, event diagnostics | Non-zero rows are event observations; sign is positive by construction. |
| `{0, R}` or `{-R, 0, +R}` event magnitude | `caar`, `bmp_z` | `forward_return * factor`; magnitude is preserved in the abnormal-return estimate. |
| Does event size rank realized event payoff? | `event_ic` | Spearman correlation between `abs(factor)` and `signed_car`, where `signed_car = forward_return * sign(factor)`. |
| Was the event direction right? | `event_hit_rate`, `profit_factor`, `event_skewness` | Uses `signed_car`; mostly sign diagnostics, not magnitude-weighted CAAR. |

| Metric | Page |
|---|---|
| Cumulative average abnormal return — $t$-test or BMP $z$-test | [`caar`](caar.md) |
| Skewness / hit-rate / win-loss diagnostics on per-event returns | [`event_quality`](event_quality.md) |
| MFE / MAE order-statistic excursion within an event window | [`mfe_mae`](mfe_mae.md) |
| Event-window horizon decay | [`event_horizon`](event_horizon.md) |
| Herfindahl-Hirschman index (HHI) on event dates — flags clustering that violates BMP/CAAR independence | [`clustering`](clustering_hhi.md) |
| Non-parametric Corrado rank test — robust to non-Gaussian returns | [`corrado`](corrado_rank.md) |

`caar` carries the abnormal-return significance test; the rest are
descriptive event-profile diagnostics.

The `Common × Sparse` cell shares this same `{0, R}` column contract
and scope-agnostic event-time metrics — the dispatcher selects them for
any `SPARSE × PANEL` factor, broadcast or per-asset. The cross-cell
landing page is [Common sparse](common-sparse.md).

When events cluster on the same date (earnings season, macro release),
prefer `caar.bmp_z(kolari_pynnonen_adjust=True)` over the vanilla
$t$-test — the [`clustering`](clustering_hhi.md) HHI tells you when this
matters.
