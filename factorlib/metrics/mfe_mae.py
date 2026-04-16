"""MFE/MAE and per-event path quality metrics for event signals.

Per-event path analysis — requires bar-by-bar ``price`` data within the
event window. If ``price`` is not available, ``compute_mfe_mae`` returns
an empty DataFrame and downstream metrics return None gracefully.

Metrics:
    compute_mfe_mae   — per-event MFE/MAE/Bars_to_MFE/Bars_to_MAE
    mfe_mae_summary   — aggregate summary (p50, p75, ratio)
    profit_factor     — sum(gains) / sum(losses) per event
    event_skewness    — skewness of signed_car distribution
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factorlib._types import EPSILON, MIN_EVENTS, MetricOutput
from factorlib._stats import _significance_marker
from factorlib.metrics._helpers import _signed_car

_EMPTY_MFE_MAE_SCHEMA = {
    "date": pl.Datetime("ms"), "asset_id": pl.String,
    "mfe": pl.Float64, "mae": pl.Float64,
    "bars_to_mfe": pl.Int32, "bars_to_mae": pl.Int32,
}


def compute_mfe_mae(
    df: pl.DataFrame,
    *,
    window: int = 20,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    """Per-event Maximum Favorable/Adverse Excursion.

    For each event (factor ≠ 0), examines the ``window`` subsequent bars
    to find the peak gain (MFE) and peak loss (MAE) relative to event
    entry price, adjusted for signal direction.

    Args:
        df: Panel with ``date, asset_id, factor, price``.
        window: Number of bars after event to examine. Maps to
            ``EventConfig.event_window_post``.
        factor_col: Event signal column.
        price_col: Price column for bar-by-bar path.

    Returns:
        DataFrame with ``date, asset_id, mfe, mae, bars_to_mfe, bars_to_mae``.
        Empty DataFrame if ``price_col`` not present.
    """
    if price_col not in df.columns:
        return pl.DataFrame(schema=_EMPTY_MFE_MAE_SCHEMA)

    sorted_df = df.sort(["asset_id", "date"])
    events = sorted_df.filter(pl.col(factor_col) != 0)

    if len(events) == 0:
        return pl.DataFrame(schema=_EMPTY_MFE_MAE_SCHEMA)

    # Build per-asset price arrays and date→index lookup for event assets only
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

        rows.append({
            "date": event_date,
            "asset_id": asset_id,
            "mfe": mfe,
            "mae": mae,
            "bars_to_mfe": bars_to_mfe,
            "bars_to_mae": bars_to_mae,
        })

    if not rows:
        return pl.DataFrame(schema=_EMPTY_MFE_MAE_SCHEMA)

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
        pl.col("bars_to_mfe").cast(pl.Int32),
        pl.col("bars_to_mae").cast(pl.Int32),
    )


def mfe_mae_summary(mfe_mae_df: pl.DataFrame) -> MetricOutput | None:
    """Aggregate MFE/MAE statistics.

    Reports MFE/MAE ratio as the primary value — higher is better
    (favorable excursion exceeds adverse excursion).

    Args:
        mfe_mae_df: Output of ``compute_mfe_mae()``.

    Returns:
        MetricOutput with value=MFE_p50/|MAE_p75| ratio, or None if
        no MFE/MAE data available.
    """
    if mfe_mae_df.is_empty():
        return None

    n = len(mfe_mae_df)
    if n < MIN_EVENTS:
        return None

    mfe_p50 = float(mfe_mae_df["mfe"].quantile(0.50))
    mae_p75 = float(mfe_mae_df["mae"].quantile(0.75))
    mae_p95 = float(mfe_mae_df["mae"].quantile(0.95))

    ratio = mfe_p50 / abs(mae_p75) if abs(mae_p75) > EPSILON else 0.0

    bars_to_mfe_mean = float(mfe_mae_df["bars_to_mfe"].mean())
    bars_to_mae_mean = float(mfe_mae_df["bars_to_mae"].mean())

    return MetricOutput(
        name="mfe_mae",
        value=ratio,
        metadata={
            "mfe_p50": mfe_p50,
            "mae_p75": mae_p75,
            "mae_p95": mae_p95,
            "mfe_mae_ratio": ratio,
            "bars_to_mfe_mean": bars_to_mfe_mean,
            "bars_to_mae_mean": bars_to_mae_mean,
            "n_events": n,
        },
    )


def profit_factor(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """sum(positive signed_car) / sum(negative signed_car).

    Per-event aggregate — no strategy assumptions. A profit factor > 1
    means gross gains exceed gross losses across all events.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=profit_factor.
    """
    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return MetricOutput(name="profit_factor", value=0.0)

    signed = _signed_car(events, factor_col, return_col)

    gains = float(np.sum(signed[signed > 0]))
    losses = float(np.abs(np.sum(signed[signed < 0])))

    pf = gains / losses if losses > EPSILON else 0.0

    return MetricOutput(
        name="profit_factor",
        value=pf,
        metadata={
            "total_gains": gains,
            "total_losses": losses,
            "n_events": n,
            "n_wins": int(np.sum(signed > 0)),
            "n_losses": int(np.sum(signed < 0)),
        },
    )


def event_skewness(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Skewness of signed_car distribution.

    Positive skew = occasional large gains, frequent small losses
    (desirable for event strategies). Uses scipy's Fisher skewness
    (bias-corrected).

    Also tests H₀: skewness = 0 via D'Agostino's skew test.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=skewness, stat=z from D'Agostino test.
    """
    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return MetricOutput(name="event_skewness", value=0.0)

    signed = _signed_car(events, factor_col, return_col)

    skew = float(sp_stats.skew(signed, bias=False))

    if n >= 20:
        z, p = sp_stats.skewtest(signed)
        z = float(z)
        p = float(p)
    else:
        z = None
        p = None

    return MetricOutput(
        name="event_skewness",
        value=skew,
        stat=z,
        significance=_significance_marker(p) if p is not None else None,
        metadata={
            "n_events": n,
            **({"p_value": p, "stat_type": "z", "h0": "skew=0",
                "method": "D'Agostino skew test"} if p is not None else {}),
        },
    )
