"""MFE/MAE — per-event price path excursion analysis.

Aggregation: per-event MFE / MAE excursion over a fixed window
(per-event step), then cross-event quantile / ratio summary;
descriptive (no formal H₀).

Answers: "what does the price path look like after events?"

Requires bar-by-bar ``price`` data within the event window.
If ``price`` is not available, ``compute_mfe_mae`` returns an empty
DataFrame and ``mfe_mae_summary`` returns a short-circuit ``MetricOutput``
(``value=NaN``, ``metadata["reason"]``) — never ``None``.

Metrics:
    compute_mfe_mae   — per-event MFE/MAE/Bars_to_MFE/Bars_to_MAE
    mfe_mae_summary   — aggregate summary (p50, p75, ratio)

Matrix-row: compute_mfe_mae, mfe_mae_summary | (*, SPARSE, *, PANEL) | per-event | no formal H₀ | _short_circuit_output
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import EPSILON, MIN_EVENTS, MetricOutput
from factrix.metrics._helpers import _short_circuit_output

DEFAULT_MIN_ESTIMATION_SAMPLES: int = 20


def _empty_mfe_mae_schema(date_dtype: pl.DataType) -> dict[str, pl.DataType]:
    """Output schema with ``date`` dtype mirroring the caller's panel so
    users with Datetime('us') or TZ-aware inputs get a joinable result."""
    return {
        "date": date_dtype, "asset_id": pl.String,
        "mfe": pl.Float64, "mae": pl.Float64,
        "mfe_z": pl.Float64, "mae_z": pl.Float64,
        "est_sigma": pl.Float64,
        "bars_to_mfe": pl.Int32, "bars_to_mae": pl.Int32,
    }


def compute_mfe_mae(
    df: pl.DataFrame,
    *,
    window: int = 20,
    estimation_window: int = 60,
    min_estimation_samples: int = DEFAULT_MIN_ESTIMATION_SAMPLES,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    """Per-event Maximum Favorable/Adverse Excursion.

    For each event (factor != 0), examines the ``window`` subsequent bars
    to find the peak gain (MFE) and peak loss (MAE) relative to event
    entry price, adjusted for signal direction.

    Also reports an estimation-window-normalised z-score per event:

        ``mfe_z = mfe / (est_sigma · √window)``
        ``mae_z = mae / (est_sigma · √window)``

    where ``est_sigma`` is the daily-return std over the ``estimation_window``
    bars preceding the event. MFE/MAE are order statistics whose expected
    magnitude grows as ``√(window · σ²)``; comparing raw MFE across
    horizons or vol regimes conflates time-scale with signal strength.
    The z-scored versions are the apples-to-apples quantity for
    cross-setup comparisons (Campbell-Lo-MacKinlay 1997 Ch 4 on horizon
    scaling of order statistics).

    Args:
        df: Panel with ``date, asset_id, factor, price``.
        window: Number of bars after event to examine. Maps to
            ``EventConfig.event_window_post``.
        estimation_window: Look-back window for per-event daily-return
            σ (default 60).
        min_estimation_samples: Minimum non-degenerate prior bars
            required to produce a finite ``est_sigma``. Default 20
            mirrors the BMP (Boehmer-Musumeci-Poulsen 1991) daily-σ
            convention; weekly panels can drop to ~8-10. Below the
            threshold, ``mfe_z`` / ``mae_z`` report ``NaN``. Must be
            ≥2 (the std degrees-of-freedom floor).
        factor_col: Event signal column.
        price_col: Price column for bar-by-bar path.

    Returns:
        DataFrame with ``date, asset_id, mfe, mae, mfe_z, mae_z,
        est_sigma, bars_to_mfe, bars_to_mae``. Empty DataFrame if
        ``price_col`` not present.
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

        # σ̂ comes from strictly pre-event bars — excluding prices[idx]
        # avoids feeding the event-day move (the signal itself) back
        # into the volatility denominator, which would suppress z-scores
        # precisely when the signal is strong.
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
            if est_sigma > 0 and np.isfinite(est_sigma) else float("nan")
        )
        if np.isfinite(window_scale) and window_scale > EPSILON:
            mfe_z = float(mfe / window_scale)
            mae_z = float(mae / window_scale)
        else:
            mfe_z = float("nan")
            mae_z = float("nan")

        rows.append({
            "date": event_date,
            "asset_id": asset_id,
            "mfe": mfe,
            "mae": mae,
            "mfe_z": mfe_z,
            "mae_z": mae_z,
            "est_sigma": est_sigma,
            "bars_to_mfe": bars_to_mfe,
            "bars_to_mae": bars_to_mae,
        })

    if not rows:
        return pl.DataFrame(schema=empty_schema)

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(date_dtype),
        pl.col("bars_to_mfe").cast(pl.Int32),
        pl.col("bars_to_mae").cast(pl.Int32),
    )


def mfe_mae_summary(mfe_mae_df: pl.DataFrame) -> MetricOutput:
    """Aggregate MFE/MAE statistics.

    Reports MFE/MAE ratio as the primary value — higher is better
    (favorable excursion exceeds adverse excursion).

    Args:
        mfe_mae_df: Output of ``compute_mfe_mae()``.

    Returns:
        MetricOutput with value=MFE_p50/|MAE_p75| ratio. On insufficient
        data (empty input or fewer than ``MIN_EVENTS`` rows), returns a
        short-circuit MetricOutput (``value=NaN``, ``metadata["reason"]``
        set) so all metrics share a single return contract.
    """
    if mfe_mae_df.is_empty():
        return _short_circuit_output(
            "mfe_mae_summary", "no_price_data", n_events=0,
        )

    n = len(mfe_mae_df)
    if n < MIN_EVENTS:
        return _short_circuit_output(
            "mfe_mae_summary", "insufficient_events",
            n_events=n, min_required=MIN_EVENTS,
        )

    mfe_p50 = float(mfe_mae_df["mfe"].quantile(0.50))
    mae_p75 = float(mfe_mae_df["mae"].quantile(0.75))
    mae_p95 = float(mfe_mae_df["mae"].quantile(0.95))

    ratio = mfe_p50 / abs(mae_p75) if abs(mae_p75) > EPSILON else 0.0

    bars_to_mfe_mean = float(mfe_mae_df["bars_to_mfe"].mean())
    bars_to_mae_mean = float(mfe_mae_df["bars_to_mae"].mean())

    metadata: dict = {
        "mfe_p50": mfe_p50,
        "mae_p75": mae_p75,
        "mae_p95": mae_p95,
        "mfe_mae_ratio": ratio,
        "bars_to_mfe_mean": bars_to_mfe_mean,
        "bars_to_mae_mean": bars_to_mae_mean,
        "n_events": n,
        "p_value": 1.0,
    }

    # Normalized quantiles (apples-to-apples across horizons / vol regimes).
    if "mfe_z" in mfe_mae_df.columns:
        mfe_z = mfe_mae_df["mfe_z"].drop_nulls().drop_nans()
        mae_z = mfe_mae_df["mae_z"].drop_nulls().drop_nans()
        if len(mfe_z) >= MIN_EVENTS and len(mae_z) >= MIN_EVENTS:
            mfe_z_p50 = float(mfe_z.quantile(0.50))
            mae_z_p75 = float(mae_z.quantile(0.75))
            metadata["mfe_z_p50"] = mfe_z_p50
            metadata["mae_z_p75"] = mae_z_p75
            metadata["mfe_mae_ratio_z"] = (
                mfe_z_p50 / abs(mae_z_p75) if abs(mae_z_p75) > EPSILON else 0.0
            )
            metadata["n_events_z"] = int(min(len(mfe_z), len(mae_z)))

    return MetricOutput(
        name="mfe_mae_summary",
        value=ratio,
        metadata=metadata,
    )
