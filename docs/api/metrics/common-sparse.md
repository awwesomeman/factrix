---
title: Common sparse
---

Metrics for the `Common × Sparse` cell — a market-wide event dummy
broadcast across `n_assets` assets (e.g. an FOMC-announcement or macro-release
flag that is identical for every asset on a given date). This cell
combines two traits:

1. **Sparse `{0, R}` signal shape**: the factor is zero on non-event
   entries and any real value otherwise (`{0, 1}` for a pure event flag
   is the simplest form), like [Individual sparse](individual-sparse.md).
2. **Common scope**: the same value is broadcast to every asset on an
   event date rather than varying cross-sectionally.

The sparse side of this contract is still zero-value based: `0` is the
non-event state, while null means missing factor data. Fill nulls to `0` only
when missing should mean "no event". To use sparse event metrics on continuous
macro scores or regime labels, first map the event-of-interest into an explicit
`{0, R}` event column.

The non-zero sign still means expected return direction. A raw policy-event
dummy is `Common × Sparse` only when the same signed interpretation applies to
every asset. If the raw event needs asset-specific bullish/bearish mapping,
create the mapped signal first; the resulting factor may become
`Individual × Sparse` because values differ by asset on the same date.

Because the event-time column contract is identical to
`Individual × Sparse`, this cell **reuses the same scope-agnostic sparse
metrics** — there is no separate Common-sparse module set. Use the
[Individual sparse](individual-sparse.md) landing page as the metric list;
the dispatcher selects those same metrics for any sparse factor whose
registered cell matches the derived data structure.

At `n_assets == 1`, the `COMMON` / `INDIVIDUAL` distinction is moot: the sparse
metric runs directly on the single-asset event series if its registered cell
allows `DataStructure.TIMESERIES`. This remains an event-density workflow, not
the `common_continuous` OLS-beta path. A dense always-in-market `{-1, +1}`
macro state should be routed as a dense directional signal instead.

Because every event shares the same date across assets, this cell is
especially exposed to event-date clustering — prefer
`caar.bmp_z(kolari_pynnonen_adjust=True)` over the vanilla $t$-test
and read [`clustering`](clustering_hhi.md) first.
