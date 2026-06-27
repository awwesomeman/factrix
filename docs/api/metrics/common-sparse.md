---
title: Common sparse
---

Metrics for the `Common × Sparse` cell — a market-wide event dummy
broadcast across `N` assets (e.g. an FOMC-announcement or macro-release
flag that is identical for every asset on a given date). This cell
combines two traits:

1. **Sparse `{0, R}` signal shape**: the factor is zero on non-event
   entries and any real value otherwise (`{0, 1}` for a pure event flag
   is the simplest form), like [Individual sparse](individual-sparse.md).
2. **Common scope**: the same value is broadcast to every asset on an
   event date rather than varying cross-sectionally.

Because the event-time column contract is identical to
`Individual × Sparse`, this cell **reuses the same scope-agnostic sparse
metrics** — there is no separate Common-sparse module set. Use the
[Individual sparse](individual-sparse.md) landing page as the metric list;
the dispatcher selects those same metrics for any sparse factor whose
registered cell matches the derived data structure.

Because every event shares the same date across assets, this cell is
especially exposed to event-date clustering — prefer
`caar.bmp_z(kolari_pynnonen_adjust=True)` over the vanilla $t$-test
and read [`clustering`](clustering_hhi.md) first.
