"""Multi-horizon event analysis — how does the signal behave across time?

Answers:
    - Is there pre-event leakage? (T-6..T-1 should be ~0)
    - At which horizon is the signal strongest?
    - Does the alpha persist or decay quickly?

Metrics:
    compute_event_returns — per-event, per-offset raw return data
    event_around_return   — return profile summary at each offset
    multi_horizon_hit_rate — win rate at multiple holding periods
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import EPSILON, MIN_EVENTS, MetricOutput
from factorlib._stats import _p_value_from_z, _significance_marker
from factorlib.metrics._helpers import _short_circuit_output


def compute_event_returns(
    df: pl.DataFrame,
    *,
    offsets: list[int] | None = None,
    factor_col: str = "factor",
    price_col: str = "price",
) -> pl.DataFrame:
    """Per-event return at multiple time offsets relative to event date.

    For each event (factor != 0) and each offset k:
        - Post-event (k > 0): cumulative return from t+1 entry.
          ``signed_return = sign(factor) × (price[t+1+k] / price[t+1] - 1)``
        - Pre-event (k < 0): single-bar return at that offset.
          ``return = price[t+k] / price[t+k-1] - 1`` (unsigned, for leakage check)

    Args:
        df: Panel with ``date, asset_id, factor, price``.
        offsets: Time offsets relative to event date.
            Defaults to ``[-6, -3, -1, 1, 6, 12, 24]``.

    Returns:
        DataFrame with ``offset, date, asset_id, signed_return, abs_return``.
        Empty if ``price`` column not present.
    """
    if offsets is None:
        offsets = [-6, -3, -1, 1, 6, 12, 24]

    # Mirror input's date dtype in the output so TZ / precision are
    # preserved — users with Datetime('us') or TZ-aware panels can join
    # this result back without another cast.
    date_dtype = df.schema["date"]
    empty_schema = {
        "offset": pl.Int32, "date": date_dtype,
        "asset_id": pl.String, "signed_return": pl.Float64,
    }

    if price_col not in df.columns:
        return pl.DataFrame(schema=empty_schema)

    sorted_df = df.sort(["asset_id", "date"])
    events = sorted_df.filter(pl.col(factor_col) != 0)

    if len(events) == 0:
        return pl.DataFrame(schema=empty_schema)

    # Build per-asset price lookup
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
                # Post-event: cumulative from t+1 entry
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
                # Pre-event: single-bar return (unsigned for leakage detection)
                bar_idx = idx + k
                prev_idx = bar_idx - 1
                if bar_idx < 0 or prev_idx < 0 or bar_idx >= len(prices):
                    continue
                if prices[prev_idx] < EPSILON:
                    continue
                signed_ret = float(prices[bar_idx] / prices[prev_idx] - 1)

            rows.append({
                "offset": k,
                "date": edate,
                "asset_id": aid,
                "signed_return": signed_ret,
            })

    if not rows:
        return pl.DataFrame(schema=empty_schema)

    return pl.DataFrame(rows).with_columns(
        pl.col("offset").cast(pl.Int32),
        pl.col("date").cast(date_dtype),
    )


def event_around_return(
    df: pl.DataFrame,
    *,
    offsets: list[int] | None = None,
    factor_col: str = "factor",
    price_col: str = "price",
) -> MetricOutput:
    """Return profile at multiple offsets around event date.

    Summarizes per-offset: mean, median, p25, p75, hit_rate, n.

    The primary value is the pre-event leakage score:
    mean absolute return at pre-event offsets (should be ~0).
    High leakage → signal may be reactive, not predictive.

    Args:
        df: Panel with ``date, asset_id, factor, price``.
        offsets: Defaults to ``[-6, -3, -1, 1, 6, 12, 24]``.

    Returns:
        MetricOutput with per-offset stats in metadata. When price data is
        unavailable, returns a short-circuit MetricOutput (``value=0.0``,
        ``metadata["reason"]="no_price_data"``) so all metrics share a
        single return contract.
    """
    if offsets is None:
        offsets = [-6, -3, -1, 1, 6, 12, 24]

    event_rets = compute_event_returns(
        df, offsets=offsets, factor_col=factor_col, price_col=price_col,
    )

    if event_rets.is_empty():
        return _short_circuit_output(
            "event_around_return", "no_price_data", per_offset={},
        )

    per_offset: dict[int, dict] = {}
    pre_leakage_vals: list[float] = []

    for k in offsets:
        subset = event_rets.filter(pl.col("offset") == k)["signed_return"]
        n = len(subset)
        if n < 5:
            per_offset[k] = {"mean": None, "n": n}
            continue

        arr = subset.to_numpy()
        mean_v = float(np.mean(arr))
        per_offset[k] = {
            "mean": mean_v,
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "hit_rate": float(np.mean(arr > 0)),
            "n": n,
        }

        if k < 0:
            pre_leakage_vals.append(abs(mean_v))

    # Primary value: pre-event leakage (mean of |pre-event returns|)
    leakage = float(np.mean(pre_leakage_vals)) if pre_leakage_vals else 0.0

    return MetricOutput(
        name="event_around_return",
        value=leakage,
        metadata={
            "per_offset": per_offset,
            "interpretation": "value = mean |pre-event return|; high = potential leakage",
            "p_value": 1.0,
        },
    )


def multi_horizon_hit_rate(
    df: pl.DataFrame,
    *,
    horizons: list[int] | None = None,
    factor_col: str = "factor",
    price_col: str = "price",
) -> MetricOutput:
    """Win rate at multiple holding periods.

    Answers: "how long do you need to hold for the signal to work?"

    For each horizon, computes the fraction of events where the
    directional return is positive (signed_return > 0).

    Args:
        df: Panel with ``date, asset_id, factor, price``.
        horizons: Holding periods to test. Defaults to ``[1, 6, 12, 24]``.

    Returns:
        MetricOutput with value = hit rate at longest horizon, per-horizon
        details in metadata. When price data is unavailable, returns a
        short-circuit MetricOutput (``value=0.0``,
        ``metadata["reason"]="no_price_data"``) so all metrics share a
        single return contract.
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    event_rets = compute_event_returns(
        df, offsets=horizons, factor_col=factor_col, price_col=price_col,
    )

    if event_rets.is_empty():
        return _short_circuit_output(
            "multi_horizon_hit_rate", "no_price_data",
            per_horizon={}, horizons=horizons,
        )

    per_horizon: dict[int, dict] = {}
    last_valid_rate = 0.0

    for h in horizons:
        subset = event_rets.filter(pl.col("offset") == h)["signed_return"]
        n = len(subset)
        if n < MIN_EVENTS:
            per_horizon[h] = {"hit_rate": None, "n": n}
            continue

        arr = subset.to_numpy()
        hits = int(np.sum(arr > 0))
        rate = hits / n

        z = (hits - n * 0.5) / (np.sqrt(n) * 0.5)
        p = float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(z))))

        per_horizon[h] = {
            "hit_rate": rate,
            "n": n,
            "z_stat": float(z),
            "p_value": p,
            "significance": _significance_marker(p),
        }
        last_valid_rate = rate

    return MetricOutput(
        name="multi_horizon_hit_rate",
        value=last_valid_rate,
        metadata={
            "per_horizon": per_horizon,
            "horizons": horizons,
            "p_value": 1.0,
        },
    )
