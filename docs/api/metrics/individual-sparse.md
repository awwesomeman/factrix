# Individual sparse

Metrics for the `Individual × Sparse` cell — event signals where most
`(date, asset_id)` cells carry `factor == 0` and the few non-zero cells
mark events. The default contract is `factor ∈ {-1, 0, +1}` for signed
events; magnitude-weighted continuous values also flow through (see
[`caar`](caar.md) for the input-form table).

| Metric | Role | Page |
|---|---|---|
| Cumulative average abnormal return — $t$-test or BMP $z$-test | Primary | [`caar`](caar.md) |
| Skewness / hit-rate / win-loss diagnostics on per-event returns | Profile | [`event_quality`](event_quality.md) |
| MFE / MAE order-statistic excursion within an event window | Profile | [`mfe_mae`](mfe_mae.md) |
| Event-window horizon decay | Profile | [`event_horizon`](event_horizon.md) |
| HHI on event dates — flags clustering that violates BMP/CAAR independence | Profile | [`clustering`](clustering.md) |
| Non-parametric Corrado rank test — robust to non-Gaussian returns | Profile | [`corrado`](corrado.md) |

The `Common × Sparse` cell shares this column contract but routes
through a distinct procedure (`_CommonSparsePanelProcedure` — per-asset
OLS $\beta$ on the broadcast dummy → cross-asset $t$). Its metrics live
under [Common continuous](common-continuous.md).

When events cluster on the same date (earnings season, macro release),
prefer `caar.bmp_test(kolari_pynnonen_adjust=True)` over the vanilla
$t$-test — the [`clustering`](clustering.md) HHI tells you when this
matters.
