from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._metric_index import cell
from factrix._types import EPSILON
from factrix.metrics._decorators import metric


@metric(
    cell=cell(
        None, FactorDensity.SPARSE, DataStructure.PANEL, raw="(*, SPARSE, PANEL)"
    ),
    aggregation=Aggregation.EVENT_TIME,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_event_returns(
    data: pl.DataFrame,
    *,
    offsets: list[int] | None = None,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    r"""Per-event return at multiple time offsets relative to event date.

    Offsets are **asymmetric by design**: post-event offsets are cumulative
    from a common entry, pre-event offsets are single-bar returns. Read the
    output curve accordingly — it is not a CAR path on both sides of zero.

    An offset is a step on the **panel's period grid**, not on the asset's own
    rows: ``k`` periods after the event is ``k`` grid periods after it for
    every name, and an offset landing on a period the asset does not have
    yields no return for that event rather than reaching on to the asset's
    next row — which on a ragged panel would sit further out on the grid than
    the offset advertises. ``idx`` below is that grid position.

    Post-event (``k > 0``) — cumulative holding return::

        entry = prices[idx + 1]                 # bar after the event
        signed_return = sign(factor) * (prices[idx + 1 + k] / entry - 1)

    Every positive offset shares the same entry, so ``k`` measures a
    ``k``-bar *cumulative* move and the post-event offsets form a growth
    path: ``k = 6`` includes everything ``k = 3`` measured. Entry is
    ``idx + 1`` (not the event bar) so the series is free of look-ahead —
    only prices strictly after the signal enter the return.

    Pre-event (``k <= 0``) — single-bar return at that offset::

        signed_return = sign(factor) * (prices[idx + k] / prices[idx + k - 1] - 1)

    Each negative offset is an *independent one-bar* return at that lag,
    **not** a cumulative run-up into the event. ``k = -6`` is the return of
    the bar six periods before the event, not the six-bar move ending at the
    event. This is what the pre-event window is used for: leakage detection
    asks "is any individual pre-event bar already drifting in the signal's
    direction?", and a cumulative pre-event CAR would smear a single leaky
    bar across every longer lag, making the leaking bar impossible to
    localise. The cost is that the two sides of the curve are not on a
    common scale — post-event values grow with ``|k|`` while pre-event
    values do not — so never subtract or concatenate them into one CAR
    series.

    Note that the two branches also differ in which bar is the base:
    post-event returns are all based at ``idx + 1`` (skipping the event
    bar), while a pre-event return at ``k`` is based at ``idx + k - 1``.
    ``k = 0`` therefore yields the event bar's own return
    (``prices[idx] / prices[idx - 1] - 1``), a value that *does* include
    information from the signal bar and is excluded from the default
    offsets for that reason.

    The ``sign`` column carries ``sign(factor)`` for every row, so a consumer
    that measures the signed return against an *unsigned* baseline (the
    panel's drift, in :func:`~factrix.metrics.event_horizon.event_around_return`)
    can sign the baseline the same way instead of subtracting ``+mu`` from a
    return that carries ``-mu``.
    """
    if offsets is None:
        offsets = [-6, -3, -1, 1, 6, 12, 24]

    date_dtype = data.schema["date"]
    empty_schema = {
        "offset": pl.Int32,
        "date": date_dtype,
        "asset_id": pl.String,
        "signed_return": pl.Float64,
        "sign": pl.Float64,
    }

    if price_col not in data.columns:
        return pl.DataFrame(schema=empty_schema)  # type: ignore[arg-type]

    sorted_df = data.sort(["asset_id", "date"])
    events = sorted_df.filter(pl.col(factor_col) != 0)

    if len(events) == 0:
        return pl.DataFrame(schema=empty_schema)  # type: ignore[arg-type]

    # Offsets are counted on the panel's distinct-date grid, not on the asset's
    # own rows: an asset missing periods that others have would otherwise step
    # over the hole as though it were one period, so its "k periods after the
    # event" price would sit further out on the grid than every other name's.
    # Each asset's prices are laid onto the full grid, absent periods carrying
    # NaN, and an offset that lands on one is skipped like a missing bar.
    grid_idx = {d: i for i, d in enumerate(sorted_df["date"].unique().sort().to_list())}
    n_grid = len(grid_idx)
    event_assets = set(events["asset_id"].unique().to_list())
    asset_prices: dict[str, np.ndarray] = {}
    for aid in event_assets:
        adf = sorted_df.filter(pl.col("asset_id") == aid)
        prices = np.full(n_grid, np.nan, dtype=np.float64)
        positions = np.fromiter(
            (grid_idx[d] for d in adf["date"].to_list()),
            dtype=np.int64,
            count=adf.height,
        )
        prices[positions] = adf[price_col].to_numpy().astype(np.float64)
        asset_prices[aid] = prices

    rows: list[dict] = []
    for row in events.iter_rows(named=True):
        aid = row["asset_id"]
        edate = row["date"]
        direction = np.sign(row[factor_col])

        prices = asset_prices[aid]
        idx = grid_idx.get(edate)
        if idx is None:
            continue

        for k in offsets:
            if k > 0:
                entry_idx = idx + 1
                exit_idx = idx + 1 + k
                if entry_idx >= n_grid or exit_idx >= n_grid:
                    continue
                entry_p = prices[entry_idx]
                exit_p = prices[exit_idx]
                if not np.isfinite(entry_p) or not np.isfinite(exit_p):
                    continue
                if entry_p < EPSILON:
                    continue
                raw_ret = exit_p / entry_p - 1
                signed_ret = float(direction * raw_ret)
            else:
                bar_idx = idx + k
                prev_idx = bar_idx - 1
                if bar_idx < 0 or prev_idx < 0 or bar_idx >= n_grid:
                    continue
                bar_p = prices[bar_idx]
                prev_p = prices[prev_idx]
                if not np.isfinite(bar_p) or not np.isfinite(prev_p):
                    continue
                if prev_p < EPSILON:
                    continue
                raw_ret = bar_p / prev_p - 1
                signed_ret = float(direction * raw_ret)

            rows.append(
                {
                    "offset": k,
                    "date": edate,
                    "asset_id": aid,
                    "signed_return": signed_ret,
                    "sign": float(direction),
                }
            )

    if not rows:
        return pl.DataFrame(schema=empty_schema)  # type: ignore[arg-type]

    return pl.DataFrame(rows).with_columns(
        pl.col("offset").cast(pl.Int32),
        pl.col("date").cast(date_dtype),
    )
