from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    InputShape,
    OutputShape,
    SEMethod,
    SpecRole,
    TestMethod,
)
from factrix._metric_index import cell
from factrix._types import EPSILON
from factrix.metrics import metric

DEFAULT_MIN_ESTIMATION_SAMPLES: int = 20


def _empty_mfe_mae_schema(date_dtype: pl.DataType) -> dict[str, pl.DataType]:
    """Output schema with ``date`` dtype mirroring the caller's panel.

    Users with Datetime('us') or TZ-aware inputs get a joinable result.
    """
    return {
        "date": date_dtype,
        "asset_id": pl.String(),
        "mfe": pl.Float64(),
        "mae": pl.Float64(),
        "mfe_z": pl.Float64(),
        "mae_z": pl.Float64(),
        "est_sigma": pl.Float64(),
        "bars_to_mfe": pl.Int32(),
        "bars_to_mae": pl.Int32(),
    }


@metric(
    cell=cell(
        None, FactorDensity.SPARSE, DataStructure.PANEL, raw="(*, SPARSE, PANEL)"
    ),
    aggregation=Aggregation.EVENT_TIME,
    test_method=TestMethod.T,
    se_method=SEMethod.HAC,
    input_shape=InputShape.PANEL,
    output_shape=OutputShape.SERIES,
    role=SpecRole.PIPELINE,
)
def compute_mfe_mae(
    df: pl.DataFrame,
    *,
    window: int = 20,
    estimation_window: int = 60,
    min_estimation_samples: int = DEFAULT_MIN_ESTIMATION_SAMPLES,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    r"""Per-event Maximum Favorable/Adverse Excursion.

    For each event ($\text{factor} \neq 0$), examines the ``window``
    subsequent bars to find the peak gain (MFE) and peak loss (MAE)
    relative to event entry price, adjusted for density direction.
    """
    if min_estimation_samples < 2:
        raise ValueError(
            f"min_estimation_samples must be >= 2 (std needs ddof=1 "
            f"and at least 2 observations), got {min_estimation_samples}"
        )

    date_dtype = df.schema["date"]
    empty_schema = _empty_mfe_mae_schema(date_dtype)

    if price_col not in df.columns:
        return pl.DataFrame(schema=empty_schema)

    sorted_df = df.sort(["asset_id", "date"])
    events = sorted_df.filter(pl.col(factor_col) != 0)

    if len(events) == 0:
        return pl.DataFrame(schema=empty_schema)

    event_assets = set(events["asset_id"].unique().to_list())
    asset_groups: dict[str, tuple[dict, np.ndarray]] = {}
    for asset_id in event_assets:
        asset_data = sorted_df.filter(pl.col("asset_id") == asset_id)
        date_to_idx = {d: i for i, d in enumerate(asset_data["date"].to_list())}
        prices = asset_data[price_col].to_numpy()
        asset_groups[asset_id] = (date_to_idx, prices)

    rows: list[dict] = []
    for row in events.iter_rows(named=True):
        asset_id = row["asset_id"]
        event_date = row["date"]
        direction = 1.0 if row[factor_col] > 0 else -1.0

        date_to_idx, prices = asset_groups[asset_id]
        idx = date_to_idx.get(event_date)
        if idx is None:
            continue

        entry_price = prices[idx]
        if entry_price < EPSILON:
            continue

        end_idx = min(idx + window + 1, len(prices))
        if idx + 1 >= end_idx:
            continue

        future_prices = prices[idx + 1 : end_idx]
        signed_returns = direction * (future_prices / entry_price - 1)

        mfe = float(np.max(signed_returns))
        mae = float(np.min(signed_returns))
        bars_to_mfe = int(np.argmax(signed_returns)) + 1
        bars_to_mae = int(np.argmin(signed_returns)) + 1

        est_start = max(0, idx - estimation_window)
        est_prices = prices[est_start:idx]
        est_sigma = float("nan")
        if len(est_prices) > min_estimation_samples:
            prior = est_prices[:-1]
            safe = prior > EPSILON
            if safe.sum() >= min_estimation_samples:
                daily_rets = (est_prices[1:][safe] / prior[safe]) - 1.0
                if len(daily_rets) >= 2:
                    est_sigma = float(np.std(daily_rets, ddof=1))
        window_scale = (
            est_sigma * np.sqrt(window)
            if est_sigma > 0 and np.isfinite(est_sigma)
            else float("nan")
        )
        if np.isfinite(window_scale) and window_scale > EPSILON:
            mfe_z = float(mfe / window_scale)
            mae_z = float(mae / window_scale)
        else:
            mfe_z = float("nan")
            mae_z = float("nan")

        rows.append(
            {
                "date": event_date,
                "asset_id": asset_id,
                "mfe": mfe,
                "mae": mae,
                "mfe_z": mfe_z,
                "mae_z": mae_z,
                "est_sigma": est_sigma,
                "bars_to_mfe": bars_to_mfe,
                "bars_to_mae": bars_to_mae,
            }
        )

    if not rows:
        return pl.DataFrame(schema=empty_schema)

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(date_dtype),
        pl.col("bars_to_mfe").cast(pl.Int32),
        pl.col("bars_to_mae").cast(pl.Int32),
    )
