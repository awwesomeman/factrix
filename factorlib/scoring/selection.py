"""IC-based and cross-sectional metrics (computation substrate: rank correlations)."""

import logging

import numpy as np
import polars as pl

from factorlib.scoring.registry import register, map_linear
from factorlib.scoring._utils import (
    MetricResult,
    calc_t_stat,
    _ic_series,
    _non_overlapping_ic_tstat,
    _non_overlapping_dates,
    _rolling_windows,
    EPSILON,
    DDOF,
    CALENDAR_DAYS_PER_YEAR,
    MIN_IC_PERIODS,
    MIN_STABILITY_PERIODS,
    MIN_OOS_PERIODS,
    MIN_PORTFOLIO_PERIODS,
    MIN_EVENT_DATES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IC-based Metrics
# ---------------------------------------------------------------------------

@register("Rank_IC")
def calc_rank_ic(
    df: pl.DataFrame, _ic_cache: pl.DataFrame | None = None, _forward_periods: int = 5, **kwargs
) -> MetricResult:
    """Mean |Rank IC|. Good factor: 0.03~0.10.

    Score uses all IC periods for precision; t-stat uses non-overlapping
    samples (every forward_periods-th date) to avoid autocorrelation inflation.
    """
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    if n < MIN_IC_PERIODS:
        logger.warning("Rank_IC: only %d IC periods (need >=%d), skipping", n, MIN_IC_PERIODS)
        return MetricResult(0.0, 0.0)
    mean_ic = ic_vals.mean()
    if mean_ic is None:
        return MetricResult(0.0, 0.0)

    score = map_linear(abs(mean_ic), min_val=0.01, max_val=0.08)
    t_stat = _non_overlapping_ic_tstat(ic_df, _forward_periods)

    return MetricResult(score, t_stat, float(mean_ic))


@register("IC_IR")
def calc_ic_ir(
    df: pl.DataFrame, _ic_cache: pl.DataFrame | None = None, _forward_periods: int = 5, **kwargs
) -> MetricResult:
    """|mean(IC)| / std(IC). Good: 0.3~0.8.

    IC_IR score uses full IC series; t-stat uses non-overlapping samples.
    """
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    if n < MIN_IC_PERIODS:
        logger.warning("IC_IR: only %d IC periods (need >=%d), skipping", n, MIN_IC_PERIODS)
        return MetricResult(0.0, 0.0)
    mean_ic = ic_vals.mean()
    std_ic = ic_vals.std()
    if std_ic is None or std_ic < EPSILON or mean_ic is None:
        return MetricResult(0.0, 0.0)

    ic_ir = abs(mean_ic) / std_ic
    score = map_linear(ic_ir, min_val=0.1, max_val=0.6)
    t_stat = _non_overlapping_ic_tstat(ic_df, _forward_periods)

    return MetricResult(score, t_stat, ic_ir)


@register("Long_Alpha")
def calc_long_alpha(
    df: pl.DataFrame, q_top: float = 0.2, _forward_periods: int = 5, **kwargs
) -> MetricResult:
    """Q1 excess return vs universe mean (non-overlapping).

    Annualized using actual date range for precision.
    t_stat is from the per-period excess series, enabling adaptive weighting.
    """
    sampled = _non_overlapping_dates(df, step=_forward_periods)
    filtered = df.filter(pl.col("date").is_in(sampled.implode()))

    q1_excess = (
        filtered.with_columns(
            (pl.col("factor").rank(method="average").over("date") / pl.len().over("date"))
            .alias("pct_rank")
        )
        .group_by("date")
        .agg(
            pl.col("forward_return")
            .filter(pl.col("pct_rank") >= (1 - q_top))
            .mean()  # TODO: replace with inverse-vol weighted mean
            .alias("q1_return"),
            pl.col("forward_return").mean().alias("univ_return"),
        )
        .with_columns(
            (pl.col("q1_return") - pl.col("univ_return")).alias("excess")
        )
        .sort("date")
    )

    excess_vals = q1_excess["excess"].drop_nulls()
    n = len(excess_vals)
    if n < MIN_PORTFOLIO_PERIODS:
        logger.warning("Long_Alpha: only %d non-overlapping periods (need >=%d)", n, MIN_PORTFOLIO_PERIODS)
        return MetricResult(0.0, 0.0)

    # WHY: t-stat on non-overlapping excess returns — already sampled every
    # forward_periods steps, so no autocorrelation concern
    excess_arr = excess_vals.to_numpy()
    mean_excess = float(np.mean(excess_arr))
    std_excess = float(np.std(excess_arr, ddof=DDOF))
    t_stat = calc_t_stat(mean_excess, std_excess, n)

    nav = np.cumprod(1 + excess_arr)
    total_excess = nav[-1] - 1

    date_range = q1_excess["date"].drop_nulls()
    n_years = (date_range.max() - date_range.min()).days / CALENDAR_DAYS_PER_YEAR
    if n_years < 0.1:
        logger.warning("Long_Alpha: date range < 0.1 years, cannot annualize")
        return MetricResult(0.0, 0.0)
    ann_excess = (1 + total_excess) ** (1 / n_years) - 1

    score = map_linear(ann_excess, min_val=0.0, max_val=0.20)
    return MetricResult(score, float(t_stat), ann_excess)


@register("Monotonicity")
def calc_monotonicity(
    df: pl.DataFrame, _forward_periods: int = 5, n_groups: int = 5, **kwargs
) -> MetricResult | None:
    """Quintile returns Spearman correlation.

    Per non-overlapping date: split into n_groups by factor rank,
    compute mean return per group, Spearman corr between group index and return.
    """
    sampled = _non_overlapping_dates(df, step=_forward_periods)
    filtered = df.filter(pl.col("date").is_in(sampled.implode()))

    # Assign quintile groups per date
    grouped = filtered.with_columns(
        pl.col("factor").rank(method="average").over("date").alias("_rank"),
        pl.len().over("date").alias("_n"),
    ).with_columns(
        ((pl.col("_rank") - 1) * n_groups / pl.col("_n")).cast(pl.Int32).clip(0, n_groups - 1).alias("group")
    )

    # Mean return per group per date
    group_returns = (
        grouped.group_by(["date", "group"])
        .agg(pl.col("forward_return").mean().alias("group_ret"))  # TODO: replace with inverse-vol weighted mean
        .sort(["date", "group"])
    )

    # WHY: Spearman(group_index, returns) = Pearson(group_index, rank(returns))
    # group_index [0..n_groups-1] is ordinal, so ranking it is a linear transform
    # that doesn't affect Pearson; we only need to rank the group returns per date.
    mono_df = (
        group_returns
        .filter(pl.col("group_ret").is_not_null() & pl.col("group_ret").is_not_nan())
        .with_columns(
            pl.col("group_ret").rank(method="average").over("date").alias("ret_rank")
        )
        .group_by("date")
        .agg(
            pl.corr("group", "ret_rank").alias("mono"),
            pl.len().alias("n"),
        )
        .filter((pl.col("n") == n_groups) & pl.col("mono").is_not_null() & pl.col("mono").is_not_nan())
        .sort("date")
    )

    if len(mono_df) < MIN_PORTFOLIO_PERIODS:
        logger.warning("Monotonicity: only %d valid periods (need >=%d)", len(mono_df), MIN_PORTFOLIO_PERIODS)
        return None

    mono_arr = mono_df["mono"].to_numpy()
    avg_mono = np.mean(np.abs(mono_arr))
    std_mono = np.std(mono_arr, ddof=DDOF)
    t_stat = calc_t_stat(float(np.mean(mono_arr)), float(std_mono), len(mono_arr))

    score = map_linear(avg_mono, 0.3, 0.9)
    return MetricResult(score, t_stat, float(np.mean(mono_arr)))


# ---------------------------------------------------------------------------
# IC-based Metrics (continued) — stability and decay
# ---------------------------------------------------------------------------

@register("OOS_Decay")
def calc_oos_decay(
    df: pl.DataFrame,
    oos_ratio: float = 0.2,
    _ic_cache: pl.DataFrame | None = None,
    **kwargs,
) -> MetricResult:
    """IS vs OOS |IC| comparison with sign-flip penalty."""
    # WHY: _ic_series already returns sorted-by-date with unique dates (group_by + sort)
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    dates = ic_df["date"]
    split_idx = int(len(dates) * (1 - oos_ratio))

    n_oos = len(dates) - split_idx
    if n_oos < MIN_OOS_PERIODS:
        logger.warning("OOS_Decay: only %d OOS periods (need >=%d), score unreliable", n_oos, MIN_OOS_PERIODS)
    if split_idx < MIN_OOS_PERIODS:
        logger.warning("OOS_Decay: only %d IS periods (need >=%d), score unreliable", split_idx, MIN_OOS_PERIODS)

    split_date = dates[split_idx]
    mean_is = ic_df.filter(pl.col("date") < split_date)["ic"].mean()
    mean_oos = ic_df.filter(pl.col("date") >= split_date)["ic"].mean()

    if mean_is is None or mean_oos is None or abs(mean_is) < EPSILON:
        return MetricResult(0.0)

    sign_consistent = (mean_is > 0 and mean_oos > 0) or (mean_is < 0 and mean_oos < 0)
    decay = abs(mean_oos) / abs(mean_is)
    if not sign_consistent:
        decay *= 0.5

    oos_ic = ic_df.filter(pl.col("date") >= split_date)["ic"].drop_nulls()
    n_oos_ic = len(oos_ic)
    std_oos_ic = float(oos_ic.std(ddof=DDOF)) if n_oos_ic > 1 else None
    # WHY: OOS t-stat 衡量 OOS IC 均值的統計顯著性，讓 adaptive_weight 在 OOS
    # 樣本不足時自動降權此指標；定義與 Event_Decay 一致
    t_stat = calc_t_stat(float(mean_oos), std_oos_ic, n_oos_ic) if std_oos_ic else 0.0

    return MetricResult(map_linear(decay, min_val=0.3, max_val=1.0), t_stat, decay)


@register("IC_Stability")
def calc_ic_stability(
    df: pl.DataFrame,
    _ic_cache: pl.DataFrame | None = None,
    _forward_periods: int = 5,
    **kwargs,
) -> MetricResult:
    """Rolling-window IC_IR consistency (vectorized via stride tricks).
    Window adapts to sample size: min(12, n//3), minimum 4 periods.

    # WHY: IC series from overlapping forward returns has strong positive
    # autocorrelation (consecutive ICs share forward_periods-1 return days),
    # making within-window IC_IR artificially high and non-discriminating.
    # Sampling every forward_periods-th IC removes this autocorrelation,
    # yielding a meaningful stability score.
    """
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    ic_vals = ic_df.gather_every(_forward_periods)["ic"].drop_nulls().to_numpy().copy()

    if len(ic_vals) < MIN_STABILITY_PERIODS:
        logger.warning("IC_Stability: only %d IC periods (need >=%d), skipping", len(ic_vals), MIN_STABILITY_PERIODS)
        return MetricResult(0.0)

    window = max(4, min(12, len(ic_vals) // 3))
    windows = _rolling_windows(ic_vals, window)

    means = np.abs(windows.mean(axis=1))
    # WHY: ddof=1 for sample std, consistent with Polars .std() default;
    # rolling windows are small (4~12), so ddof matters
    stds = windows.std(axis=1, ddof=DDOF)
    valid = stds > EPSILON
    if not valid.any():
        return MetricResult(0.0)

    rolling_irs = means[valid] / stds[valid]
    mean_ir = float(rolling_irs.mean())
    return MetricResult(map_linear(mean_ir, min_val=0.1, max_val=0.5), None, mean_ir)


# ---------------------------------------------------------------------------
# Placeholders — registered for config references, pending implementation
# ---------------------------------------------------------------------------

@register("Orthogonality")
def calc_orthogonality(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Placeholder: requires factor zoo. Returns None to signal 'skip'."""
    return None


@register("Cross_Consistency")
def calc_cross_consistency(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Placeholder: group/region sub-sample IC stability."""
    return None


@register("Sensitivity_Dispersion")
def calc_sensitivity_dispersion(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Placeholder: global macro beta dispersion across assets."""
    return None


@register("Capacity")
def calc_capacity(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Placeholder: portfolio capacity estimation."""
    return None


@register("Slippage")
def calc_slippage(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Placeholder: slippage impact on alpha."""
    return None


# ---------------------------------------------------------------------------
# Cross-sectional Metrics
# ---------------------------------------------------------------------------

@register("Event_CAR_Dispersion")
def calc_event_car_dispersion(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """事件日橫截面 forward return 離散度：衡量訊號在不同資產間的區分能力。

    對每個事件日，計算該日所有觸發事件的 forward_return 橫截面標準差，
    再取多個事件日的均值。離散度越高，訊號越能分辨強弱資產。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（資料不足時）。

    單資產：
        每個事件日只有 1 row → std(1 值) = NaN → 全部被過濾 → 永遠回傳 None。
        此為設計決策：橫截面離散度在單資產下無意義。

    多資產範例（某事件日 5 支股票同時觸發）：
        date D: A=+3%, B=-2%, C=+5%, D=+1%, E=-1%
        car_std = std([3, -2, 5, 1, -1]%) ≈ 2.9%

        date E: A=+4%, B=+1%, C=-3%, D=+2%
        car_std = std([4, 1, -3, 2]%) ≈ 2.9%

        mean_dispersion = (2.9 + 2.9) / 2 = 2.9%
        score = map_linear(2.9%, 1%, 10%) ≈ 21
        t_stat = mean / SEM（跨事件日）

    統計假設：
        - WHY: 使用 forward_return 而非 abnormal_return，確保單標的也能呼叫
          而不會因 abnormal_return 恆 0 導致誤導性的 std=0 分數。
        - 每個事件日需至少 MIN_EVENT_DATES 支資產觸發才計入橫截面 std。
        - 評分範圍：mean_dispersion 從 1%（0分）到 10%（100分）。
    """
    events = df.filter(pl.col("factor_raw") != 0)
    if events.is_empty():
        return None

    dispersion = (
        events.group_by("date")
        .agg(
            pl.col("forward_return").std().alias("car_std"),
            pl.len().alias("n")
        )
        .filter((pl.col("n") >= MIN_EVENT_DATES) & pl.col("car_std").is_not_null())
    )

    if dispersion.is_empty() or len(dispersion) < MIN_EVENT_DATES:
        return None

    mean_disp = dispersion["car_std"].mean()
    std_disp = dispersion["car_std"].std(ddof=DDOF)
    if mean_disp is None or std_disp is None:
        return None

    n = len(dispersion)
    t_stat = calc_t_stat(float(mean_disp), float(std_disp), n)

    score = map_linear(mean_disp, min_val=0.01, max_val=0.10)
    return MetricResult(score, float(t_stat), float(mean_disp))
