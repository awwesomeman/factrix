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
    r"""Per-event return at multiple time offsets relative to event date."""
    if offsets is None:
        offsets = [-6, -3, -1, 1, 6, 12, 24]

    date_dtype = data.schema["date"]
    empty_schema = {
        "offset": pl.Int32,
        "date": date_dtype,
        "asset_id": pl.String,
        "signed_return": pl.Float64,
    }

    if price_col not in data.columns:
        return pl.DataFrame(schema=empty_schema)  # type: ignore[arg-type]

    sorted_df = data.sort(["asset_id", "date"])
    events = sorted_df.filter(pl.col(factor_col) != 0)

    if len(events) == 0:
        return pl.DataFrame(schema=empty_schema)  # type: ignore[arg-type]

    event_assets = set(events["asset_id"].unique().to_list())
    asset_data: dict[str, tuple[dict, np.ndarray]] = {}
    for aid in event_assets:
        adf = sorted_df.filter(pl.col("asset_id") == aid)
        date_idx = {d: i for i, d in enumerate(adf["date"].to_list())}
        prices = adf[price_col].to_numpy()
        asset_data[aid] = (date_idx, prices)

    rows: list[dict] = []
    for row in events.iter_rows(named=True):
        aid = row["asset_id"]
        edate = row["date"]
        direction = np.sign(row[factor_col])

        date_idx, prices = asset_data[aid]
        idx = date_idx.get(edate)
        if idx is None:
            continue

        for k in offsets:
            if k > 0:
                entry_idx = idx + 1
                exit_idx = idx + 1 + k
                if entry_idx >= len(prices) or exit_idx >= len(prices):
                    continue
                entry_p = prices[entry_idx]
                if entry_p < EPSILON:
                    continue
                raw_ret = prices[exit_idx] / entry_p - 1
                signed_ret = float(direction * raw_ret)
            else:
                bar_idx = idx + k
                prev_idx = bar_idx - 1
                if bar_idx < 0 or prev_idx < 0 or bar_idx >= len(prices):
                    continue
                if prices[prev_idx] < EPSILON:
                    continue
                raw_ret = prices[bar_idx] / prices[prev_idx] - 1
                signed_ret = float(direction * raw_ret)

            rows.append(
                {
                    "offset": k,
                    "date": edate,
                    "asset_id": aid,
                    "signed_return": signed_ret,
                }
            )

    if not rows:
        return pl.DataFrame(schema=empty_schema)  # type: ignore[arg-type]

    return pl.DataFrame(rows).with_columns(
        pl.col("offset").cast(pl.Int32),
        pl.col("date").cast(date_dtype),
    )
