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

| Metric | Role | Page |
|---|---|---|
| Cumulative average abnormal return — $t$-test or BMP $z$-test | Primary | [`caar`](caar.md) |
| Skewness / hit-rate / win-loss diagnostics on per-event returns | Profile | [`event_quality`](event_quality.md) |
| MFE / MAE order-statistic excursion within an event window | Profile | [`mfe_mae`](mfe_mae.md) |
| Event-window horizon decay | Profile | [`event_horizon`](event_horizon.md) |
| Herfindahl-Hirschman index (HHI) on event dates — flags clustering that violates BMP/CAAR independence | Profile | [`clustering`](clustering_hhi.md) |
| Non-parametric Corrado rank test — robust to non-Gaussian returns | Profile | [`corrado`](corrado_rank.md) |

The `Common × Sparse` cell shares this column contract but routes
through a distinct procedure (`_CommonSparsePanelProcedure` — per-asset
ordinary least squares (OLS) $\beta$ on the broadcast dummy → cross-asset $t$). Its metrics live
under [Common continuous](common-continuous.md).

When events cluster on the same date (earnings season, macro release),
prefer `caar.bmp_test(kolari_pynnonen_adjust=True)` over the vanilla
$t$-test — the [`clustering`](clustering_hhi.md) HHI tells you when this
matters.
