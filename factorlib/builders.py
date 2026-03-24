"""Stateless artifact builders — pure Polars transforms for IC, NAV, and event matrices."""

import numpy as np
import polars as pl
from scipy import stats as sp_stats


def build_signal_feature_matrix(
    raw_data: pl.DataFrame,
    date_col: str = "datetime",
    asset_col: str = "ticker",
    price_col: str = "close",
    factor_col: str = "factor",
    horizons: list[int] | None = None,
    continuous: bool = False,
) -> pl.DataFrame:
    """Build multi-horizon Signal Feature Matrix for event signals.

    Generates absolute directional returns, win rates, and K-S test results for Buy and Sell signals.
    By default, computes discrete points for consistency and performance.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]
    
    # WHY: Discrete calculation is 5x faster than continuous for 20D window.
    calc_h = list(range(1, max(horizons) + 1)) if continuous else sorted(list(set(horizons)))
    
    df = raw_data.sort([asset_col, date_col])

    # Batch compute forward returns only for needed horizons
    fwd_exprs = [
        (pl.col(price_col).shift(-h).over(asset_col) / pl.col(price_col) - 1).alias(f"fwd_ret_{h}")
        for h in calc_h
    ]
    df = df.with_columns(fwd_exprs)

    # WHY: Winsorize forward returns at 1st/99th percentile per date, consistent
    # with engine.py prepare_factor_data() so artifact CAAR matches scored CAAR.
    for h in calc_h:
        col_name = f"fwd_ret_{h}"
        lb = pl.col(col_name).quantile(0.01).over(date_col)
        ub = pl.col(col_name).quantile(0.99).over(date_col)
        df = df.with_columns(pl.col(col_name).clip(lb, ub).alias(col_name))

    events = df.filter((pl.col(factor_col).is_not_null()) & (pl.col(factor_col) != 0))
    non_events = df.filter((pl.col(factor_col).is_null()) | (pl.col(factor_col) == 0))

    results = []

    for signal_type, sign in [("Buy", 1), ("Sell", -1)]:
        signal_events = events.filter(pl.col(factor_col).sign() == sign)
        if signal_events.is_empty():
            continue

        # Paths for charting (only for calc_h)
        car_path = []
        car_std_path = []
        hit_rate_path = []
        
        n_path = []
        for h in calc_h:
            col = f"fwd_ret_{h}"
            rets = signal_events[col].drop_nulls()
            n_h = len(rets)
            if n_h > 0:
                mean_car = rets.mean()
                std_car = rets.std()
                hr = (rets > 0).sum() / n_h
                car_path.append(float(mean_car) if mean_car is not None else 0.0)
                car_std_path.append(float(std_car) if std_car is not None else 0.0)
                hit_rate_path.append(float(hr))
                n_path.append(n_h)
            else:
                car_path.append(0.0)
                car_std_path.append(0.0)
                hit_rate_path.append(0.5)
                n_path.append(0)

        # Detailed metrics only for the main horizons requested
        for h in sorted(horizons):
            col = f"fwd_ret_{h}"
            raw_rets = signal_events[col].drop_nulls()

            if len(raw_rets) == 0:
                continue

            hit_rate = float((raw_rets > 0).sum() / len(raw_rets))

            event_rets = raw_rets.to_numpy()
            non_event_rets = non_events[col].drop_nulls().to_numpy()

            ks_stat, p_val = None, None
            if len(event_rets) >= 5 and len(non_event_rets) >= 5:
                ks_stat, p_val = sp_stats.ks_2samp(event_rets, non_event_rets)

            results.append({
                "signal": signal_type,
                "horizon": h,
                "hit_rate": hit_rate,
                "ks_stat": float(ks_stat) if ks_stat is not None else None,
                "p_value": float(p_val) if p_val is not None else None,
                "car_path": car_path,
                "car_std_path": car_std_path,
                "hit_rate_path": hit_rate_path,
                "n_path": n_path,               # event count per horizon for CI computation
                "calc_horizons": calc_h,
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def build_ic_artifact(
    prepared_data: pl.DataFrame,
    min_assets: int = 10,  # WHY: mirrors MIN_IC_PERIODS in _utils.py; update together
    rolling_window: int = 63,
) -> pl.DataFrame:
    """Build IC time series artifact from prepared factor data.

    Uses method='average' for Spearman rank and filters dates with fewer than
    min_assets stocks, consistent with the scoring IC computation in _utils.py.

    Args:
        rolling_window: Window size for rolling mean IC (default 63 ≈ 3 months).
            WHY: 滾動均值 IC 直接呈現近期因子有效性趨勢，比累積 IC 更能反映 regime shift。
    """
    ranked = prepared_data.with_columns(
        pl.col("factor").rank(method="average").over("date").alias("rank_factor"),
        pl.col("forward_return").rank(method="average").over("date").alias("rank_return"),
    )
    ic_df = (
        ranked.group_by("date")
        .agg(
            pl.corr("rank_factor", "rank_return").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_assets)
        .sort("date")
        .select("date", "ic")
    )
    return ic_df.with_columns([
        pl.col("ic").cum_sum().alias("cum_ic"),
        pl.col("ic").rolling_mean(window_size=rolling_window).alias("rolling_ic"),
    ])


def build_event_temporal_artifact(
    prepared_data: pl.DataFrame,
    bucket: str = "3mo",
) -> pl.DataFrame:
    """Quarterly signed CAAR breakdown for event signal temporal stability.

    Analogous to rolling IC for continuous factors: shows whether the event
    signal's directional alpha is consistent across time periods, or
    concentrated in specific regimes.

    Args:
        prepared_data: Output of prepare_factor_data() — must contain
            ``forward_return`` and ``factor_raw`` columns.
        bucket: Polars truncate duration string (default "3mo" = quarterly).
            WHY: 季度粒度在 2–5 年資料下通常有 8–20 個觀察點，
            足以看出 regime shift 又不至於每期樣本過少。

    Returns:
        DataFrame with columns: period, mean_signed_car, std_signed_car,
        n_events, hit_rate, ci_95. Empty if required columns are missing.
    """
    if "forward_return" not in prepared_data.columns or "factor_raw" not in prepared_data.columns:
        return pl.DataFrame()

    events = prepared_data.filter(pl.col("factor_raw") != 0)
    if events.is_empty():
        return pl.DataFrame()

    events = (
        events
        .with_columns([
            (pl.col("forward_return") * pl.col("factor_raw").sign()).alias("signed_car"),
            pl.col("date").dt.truncate(bucket).alias("period"),
        ])
        .drop_nulls(["signed_car"])
    )

    quarterly = (
        events.group_by("period")
        .agg([
            pl.col("signed_car").mean().alias("mean_signed_car"),
            pl.col("signed_car").std(ddof=1).alias("std_signed_car"),
            pl.len().alias("n_events"),
            (pl.col("signed_car") > 0).sum().cast(pl.Float64).alias("n_wins"),
        ])
        .with_columns([
            (pl.col("n_wins") / pl.col("n_events")).alias("hit_rate"),
            # WHY: 95% CI of the mean = 1.96 × SEM = 1.96 × std / √n
            (1.96 * pl.col("std_signed_car") / pl.col("n_events").sqrt()).alias("ci_95"),
        ])
        .sort("period")
    )
    return quarterly


def build_nav_artifact(
    prepared_data: pl.DataFrame,
    step: int = 5,
) -> pl.DataFrame:
    """Build Long-Only Q1 vs Universe NAV curves (non-overlapping).

    Uses method='average' for rank consistency with scoring metrics.
    Drops nulls before computing NAV to guarantee aligned lengths.
    """
    dates = prepared_data["date"].unique().sort()
    sampled = dates.gather_every(step)
    filtered = prepared_data.filter(pl.col("date").is_in(sampled.implode()))

    returns_df = (
        filtered.with_columns(
            (pl.col("factor").rank(method="average").over("date") / pl.len().over("date"))
            .alias("pct_rank")
        )
        .group_by("date")
        .agg(
            pl.col("forward_return")
            .filter(pl.col("pct_rank") >= 0.8)
            .mean()  # TODO: replace with inverse-vol weighted mean
            .alias("q1_return"),
            pl.col("forward_return").mean().alias("universe_return"),
        )
        .sort("date")
    )

    # Drop nulls FIRST, then compute NAV from the same clean DataFrame
    valid = returns_df.drop_nulls()

    q1_nav = np.cumprod(1 + valid["q1_return"].to_numpy())
    univ_nav = np.cumprod(1 + valid["universe_return"].to_numpy())

    return pl.DataFrame({
        "date": valid["date"],
        "Q1_NAV": q1_nav,
        "Universe_NAV": univ_nav,
        "Excess_NAV": q1_nav / univ_nav,
    })
