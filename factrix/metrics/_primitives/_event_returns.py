from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import TypedDict

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    InputShape,
    OutputShape,
    SpecRole,
)
from factrix._data_input import DataInput, _coerce_price_data
from factrix._metric_index import cell
from factrix._types import EPSILON
from factrix.metrics._decorators import metric


class EventOffsetAuditEntry(TypedDict):
    eligible: int
    computed: int
    censored: int
    censor_reasons: dict[str, int]


type EventOffsetAudit = dict[int, EventOffsetAuditEntry]


def _empty_event_returns(date_dtype: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "offset": pl.Int32,
            "date": date_dtype,
            "asset_id": pl.String,
            "signed_return": pl.Float64,
            "sign": pl.Float64,
        }
    )


def _record_censor(audit: EventOffsetAudit, offset: int, reason: str) -> None:
    entry = audit[offset]
    entry["censored"] += 1
    reasons = entry["censor_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _compute_event_returns_with_audit(
    data: pl.DataFrame,
    *,
    price_data: pl.DataFrame | None,
    offsets: list[int],
    factor_col: str,
    price_col: str,
) -> tuple[pl.DataFrame, EventOffsetAudit]:
    """Compute valid event-offset rows and account for every omitted row."""
    date_dtype = data.schema["date"]
    sorted_events = data.sort(["asset_id", "date"]).filter(pl.col(factor_col) != 0)
    audit: EventOffsetAudit = {
        offset: {
            "eligible": sorted_events.height,
            "computed": 0,
            "censored": 0,
            "censor_reasons": {},
        }
        for offset in offsets
    }
    if sorted_events.is_empty():
        return _empty_event_returns(date_dtype), audit

    paths = data if price_data is None else price_data
    if price_col not in paths.columns:
        for offset in offsets:
            audit[offset]["censored"] = sorted_events.height
            audit[offset]["censor_reasons"] = {
                "missing_price_column": sorted_events.height
            }
        return _empty_event_returns(date_dtype), audit

    sorted_paths = paths.sort(["asset_id", "date"])
    grid_idx = {
        date: index
        for index, date in enumerate(sorted_paths["date"].unique().sort().to_list())
    }
    n_grid = len(grid_idx)
    event_assets = set(sorted_events["asset_id"].unique().to_list())
    asset_prices: dict[object, npt.NDArray[np.float64]] = {}
    for asset_id, asset_frame in sorted_paths.partition_by(
        "asset_id", as_dict=True, maintain_order=True
    ).items():
        key = asset_id[0]
        if key not in event_assets:
            continue
        prices = np.full(n_grid, np.nan, dtype=np.float64)
        positions = np.fromiter(
            (grid_idx[date] for date in asset_frame["date"].to_list()),
            dtype=np.int64,
            count=asset_frame.height,
        )
        prices[positions] = asset_frame[price_col].to_numpy().astype(np.float64)
        asset_prices[key] = prices

    rows: list[dict[str, object]] = []
    for row in sorted_events.iter_rows(named=True):
        asset_id = row["asset_id"]
        event_date = row["date"]
        direction = np.sign(row[factor_col])
        idx = grid_idx.get(event_date)
        event_prices = asset_prices.get(asset_id)

        for offset in offsets:
            reason: str | None = None
            if idx is None:
                reason = "event_date_not_on_price_grid"
            elif event_prices is None:
                reason = "asset_not_in_price_data"
            elif offset > 0:
                entry_idx = idx + 1
                exit_idx = idx + 1 + offset
                if entry_idx >= n_grid or exit_idx >= n_grid:
                    reason = "offset_out_of_bounds"
                else:
                    entry_price = event_prices[entry_idx]
                    exit_price = event_prices[exit_idx]
                    if not np.isfinite(entry_price):
                        reason = "missing_entry_price"
                    elif entry_price < EPSILON:
                        reason = "invalid_entry_price"
                    elif not np.isfinite(exit_price):
                        reason = "missing_exit_price"
                    else:
                        signed_return = float(
                            direction * (exit_price / entry_price - 1)
                        )
            else:
                bar_idx = idx + offset
                previous_idx = bar_idx - 1
                if bar_idx < 0 or previous_idx < 0 or bar_idx >= n_grid:
                    reason = "offset_out_of_bounds"
                else:
                    bar_price = event_prices[bar_idx]
                    previous_price = event_prices[previous_idx]
                    if not np.isfinite(previous_price):
                        reason = "missing_previous_price"
                    elif previous_price < EPSILON:
                        reason = "invalid_previous_price"
                    elif not np.isfinite(bar_price):
                        reason = "missing_offset_price"
                    else:
                        signed_return = float(
                            direction * (bar_price / previous_price - 1)
                        )

            if reason is not None:
                _record_censor(audit, offset, reason)
                continue
            audit[offset]["computed"] += 1
            rows.append(
                {
                    "offset": offset,
                    "date": event_date,
                    "asset_id": asset_id,
                    "signed_return": signed_return,
                    "sign": float(direction),
                }
            )

    if not rows:
        return _empty_event_returns(date_dtype), audit
    return (
        pl.DataFrame(rows).with_columns(
            pl.col("offset").cast(pl.Int32),
            pl.col("date").cast(date_dtype),
        ),
        audit,
    )


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
    price_data: DataInput | None = None,
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

    Args:
        data: Evaluation panel owning event dates and factor values.
        price_data: Optional complete ``date, asset_id, price`` panel. When
            supplied, offsets walk this price grid while events still come
            only from ``data``. This preserves tail and between-evaluation-date
            prices removed by ``compute_forward_return``.
        offsets: Period-grid offsets to compute.
        factor_col: Event factor column on ``data``.
        price_col: Price column on the selected price panel.

    Returns:
        One row per computed event-offset pair. Censored pairs remain omitted
        from this low-level table; :func:`event_around_return` reports their
        eligible, computed, censored, and reason counts per offset.
    """
    resolved_offsets = [-6, -3, -1, 1, 6, 12, 24] if offsets is None else offsets
    resolved_prices = _coerce_price_data(
        price_data,
        data=data,
        func_name="compute_event_returns",
        price_col=price_col,
    )
    returns, _ = _compute_event_returns_with_audit(
        data,
        price_data=resolved_prices,
        offsets=resolved_offsets,
        factor_col=factor_col,
        price_col=price_col,
    )
    return returns
