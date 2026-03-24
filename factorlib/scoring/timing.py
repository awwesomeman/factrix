"""Portfolio-based and event metrics (computation substrate: return series)."""

import logging

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factorlib.scoring.registry import register, map_linear
from factorlib.scoring._utils import (
    MetricResult,
    calc_t_stat,
    _ic_series,
    _non_overlapping_dates,
    _signed_event_cars,
    EPSILON,
    DDOF,
    MIN_IC_PERIODS,
    MIN_PORTFOLIO_PERIODS,
    MIN_EVENT_SAMPLE,
    MIN_EVENT_DECAY_PERIODS,
    MIN_EVENT_MONTHS,
    MIN_SKEW_PERIODS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio-based Metrics
# ---------------------------------------------------------------------------

# WHY: rank correlation > 0.99 表示因子幾乎未變（可能資料延遲或因子本身極低頻），
# 給固定 80 分表示「低換手成本」但不滿分（避免獎勵停滯因子）
_TURNOVER_RC_CAP: float = 0.99
_TURNOVER_STATIC_SCORE: float = 80.0


@register("Turnover")
def calc_turnover(df: pl.DataFrame, **kwargs) -> MetricResult:
    """Rank autocorrelation (vectorized). High = low turnover = lower costs."""
    dates = df["date"].unique().sort()
    if len(dates) < 2:
        logger.warning("Turnover: need at least 2 dates")
        return MetricResult(50.0)

    date_map = pl.DataFrame({
        "date": dates[1:],
        "prev_date": dates[:-1],
    })

    # WHY: rank once and reuse for both curr/prev to avoid duplicate .rank().over()
    ranked = df.select(
        "date", "asset_id",
        pl.col("factor").rank(method="average").over("date").alias("factor_rank"),
    )

    paired = (
        ranked.rename({"factor_rank": "rank_curr"})
        .join(date_map, on="date")
        .join(
            ranked.rename({"date": "prev_date", "factor_rank": "rank_prev"}),
            on=["prev_date", "asset_id"],
        )
    )

    rc_df = (
        paired.group_by("date")
        .agg(
            pl.corr("rank_curr", "rank_prev").alias("rc"),
            pl.len().alias("n"),
        )
        .filter((pl.col("n") >= 5) & pl.col("rc").is_not_null() & pl.col("rc").is_not_nan())
    )

    if rc_df.is_empty():
        return MetricResult(50.0)

    avg_rc = rc_df["rc"].mean()
    if avg_rc is None:
        return MetricResult(50.0)
    if avg_rc > _TURNOVER_RC_CAP:
        return MetricResult(_TURNOVER_STATIC_SCORE, None, avg_rc)
    return MetricResult(map_linear(avg_rc, min_val=0.5, max_val=0.95), None, avg_rc)


@register("MDD")
def calc_mdd(df: pl.DataFrame, _forward_periods: int = 5, **kwargs) -> MetricResult:
    """L/S portfolio max drawdown (non-overlapping periods)."""
    sampled = _non_overlapping_dates(df, step=_forward_periods)
    filtered = df.filter(pl.col("date").is_in(sampled.implode()))

    ls_df = (
        filtered.with_columns(
            (pl.col("factor").rank(method="average").over("date") / pl.len().over("date"))
            .alias("pct_rank")
        )
        .group_by("date")
        .agg(
            pl.col("forward_return")
            .filter(pl.col("pct_rank") >= 0.8)
            .mean()  # TODO: replace with inverse-vol weighted mean
            .alias("long_ret"),
            pl.col("forward_return")
            .filter(pl.col("pct_rank") <= 0.2)
            .mean()  # TODO: replace with inverse-vol weighted mean
            .alias("short_ret"),
        )
        .with_columns((pl.col("long_ret") - pl.col("short_ret")).alias("ls_ret"))
        .sort("date")
    )

    ls_rets = ls_df["ls_ret"].drop_nulls().to_numpy()
    if len(ls_rets) < MIN_PORTFOLIO_PERIODS:
        logger.warning("MDD: only %d non-overlapping periods (need >=%d)", len(ls_rets), MIN_PORTFOLIO_PERIODS)
        return MetricResult(50.0)

    nav = np.cumprod(1 + ls_rets)
    running_max = np.maximum.accumulate(nav)
    mdd = abs((nav / running_max - 1).min())

    return MetricResult(map_linear(1 - mdd, min_val=0.5, max_val=0.9), None, mdd)


# ---------------------------------------------------------------------------
# IC-derived Metrics (uses IC series, but returns scalar directional stat)
# ---------------------------------------------------------------------------

@register("Hit_Rate")
def calc_hit_rate(
    df: pl.DataFrame,
    _ic_cache: pl.DataFrame | None = None,
    _forward_periods: int = 5,
    **kwargs,
) -> MetricResult | None:
    """Fraction of non-overlapping periods where IC > 0.

    t_stat is from a binomial test (H0: p=0.5).
    """
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)

    all_dates = ic_df["date"].sort()
    sampled = all_dates.gather_every(_forward_periods)
    ic_sampled = ic_df.filter(pl.col("date").is_in(sampled.implode()))

    ic_vals = ic_sampled["ic"].drop_nulls()
    n = len(ic_vals)
    if n < MIN_IC_PERIODS:
        logger.warning("Hit_Rate: only %d non-overlapping IC periods (need >=%d)", n, MIN_IC_PERIODS)
        return None

    hits = (ic_vals > 0).sum()
    hit_rate = hits / n

    # Binomial t-stat: (p - 0.5) / sqrt(0.25 / n)
    t_stat = (hit_rate - 0.5) / np.sqrt(0.25 / n)

    score = map_linear(hit_rate, 0.45, 0.65)
    return MetricResult(score, float(t_stat), float(hit_rate))


# ---------------------------------------------------------------------------
# Event Metrics
# ---------------------------------------------------------------------------

@register("Event_CAAR")
def calc_event_caar(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Signed CAAR — 衡量事件訊號的方向性 alpha 均值。

    signed_car = forward_return × sign(factor_raw)：Buy(+1) 漲、Sell(-1) 跌皆為正
    貢獻；方向錯誤為負貢獻。先對每個事件日取橫截面均值，再對時序均值與 t_stat。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（資料不足時）。

    統計假設：
        - t_stat = mean(daily_caar) / SEM，SEM = std(daily_caar) / √n_dates。
        - n_dates（而非 n_events）作為獨立觀測數，避免同日多資產膨脹 t_stat。
        - 負均值 → score=0，懲罰方向錯誤，但不回傳 None 以確保懲罰計入總分。
        - 評分範圍：mean_signed_car 從 0.5%（0分）到 5%（100分）。
    """
    if "factor_raw" not in df.columns or "forward_return" not in df.columns:
        return None

    event_dates = (
        df.filter(pl.col("factor_raw") != 0)
        .with_columns(
            (pl.col("forward_return") * pl.col("factor_raw").sign()).alias("signed_car")
        )
        .group_by("date")
        .agg(pl.col("signed_car").mean().alias("daily_caar"))
        .drop_nulls("daily_caar")
        .sort("date")
    )

    n_dates = len(event_dates)
    if n_dates < MIN_EVENT_SAMPLE:
        logger.warning("Event_CAAR: only %d event dates (need >=%d)", n_dates, MIN_EVENT_SAMPLE)
        return None

    daily = event_dates["daily_caar"].to_numpy()
    mean_s = float(np.mean(daily))
    std_s = float(np.std(daily, ddof=DDOF))

    t_stat = calc_t_stat(mean_s, std_s, n_dates)
    # WHY: 使用 signed mean（非 abs），負值→0 分，懲罰方向錯誤的訊號
    score = map_linear(mean_s, min_val=0.005, max_val=0.05)
    return MetricResult(float(score), float(t_stat), float(mean_s))


@register("Event_KS")
def calc_event_ks(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """兩樣本 K-S 檢定：事件觀測 vs 非事件觀測的 forward return 分布差異。

    以所有 factor_raw≠0 的 rows 為事件組、factor_raw=0 的 rows 為對照組，
    檢驗兩組 forward_return 分布是否顯著不同，並以 signed_car 均值過濾方向。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（樣本不足時）。

    統計假設：
        - WHY: 方向過濾用 signed_car 均值而非 forward_return 均值；
          Buy+Sell 混合時 event_rets 雙峰均值 ≈ 0，直接比較均值會誤判方向。
        - t_stat = D × √(n1×n2/(n1+n2))，KS 統計量的漸近常態化，近似 N(0,1)。
        - 評分範圍：1-p_value 從 0.90（0分）到 0.99（100分）。
    """
    if "forward_return" not in df.columns:
        return None
    events = df.filter(pl.col("factor_raw") != 0)
    non_events = df.filter(pl.col("factor_raw") == 0)

    if events.is_empty() or non_events.is_empty():
        return None

    event_rets = events["forward_return"].drop_nulls().to_numpy()
    non_event_rets = non_events["forward_return"].drop_nulls().to_numpy()

    if len(event_rets) < MIN_EVENT_SAMPLE or len(non_event_rets) < MIN_EVENT_SAMPLE:
        logger.warning("Event_KS: insufficient samples (events=%d, non-events=%d, need >=%d)",
                       len(event_rets), len(non_event_rets), MIN_EVENT_SAMPLE)
        return None

    stat, p_value = sp_stats.ks_2samp(event_rets, non_event_rets)

    # WHY: KS z-stat = D × √(n_eff)，n_eff = n1×n2/(n1+n2)，為業界標準常態化方式，
    # 讓 adaptive_weight 能依統計顯著性衰減此指標的貢獻
    n1, n2 = len(event_rets), len(non_event_rets)
    ks_z = float(stat * np.sqrt(n1 * n2 / (n1 + n2)))

    # WHY: 方向過濾用 signed_car 均值（非 raw forward_return）：
    # Buy+Sell 混合訊號時，event_rets 可能雙峰且均值 ≈ 0，直接比較均值會誤判；
    # signed_car = forward_return × sign(factor_raw) 才能正確衡量方向性 alpha
    mean_signed = float((events["forward_return"] * events["factor_raw"].sign()).drop_nulls().mean() or 0.0)
    if mean_signed < 0:
        return MetricResult(0.0, ks_z, float(p_value))

    score = map_linear(1 - p_value, min_val=0.90, max_val=0.99)
    return MetricResult(float(score), ks_z, float(p_value))


# ---------------------------------------------------------------------------
# Event Metrics (continued) — decay and stability
# ---------------------------------------------------------------------------

@register("Event_Decay")
def calc_event_decay(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """IS vs OOS signed CAAR 比較：衡量事件訊號的樣本外持續性。

    以唯一事件日期排序後 80/20 切割：IS 段估計基準 CAAR，OOS 段盲測。
    decay = |mean_oos| / |mean_is|，反映訊號強度保留比例。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（資料不足時）。

    統計假設：
        - WHY: 80/20 split 同 OOS_Decay（selection 指標），IS 更多資料讓基準更穩定。
        - 異號（OOS 方向反轉）：decay ×0.5 懲罰，表示訊號在 OOS 翻轉。
        - t_stat = mean_oos / SEM_oos；OOS 樣本稀疏時 t_stat→0，adaptive_weight 自動降權。
        - 評分範圍：decay 從 0.3（0分）到 1.0（100分）。
    """
    if "factor_raw" not in df.columns:
        return None
    events = df.filter(pl.col("factor_raw") != 0).sort("date")
    if events.is_empty():
        return None

    dates = events["date"].unique().sort()
    n_dates = len(dates)
    if n_dates < MIN_EVENT_DECAY_PERIODS:
        logger.warning("Event_Decay: only %d event dates (need >=%d)", n_dates, MIN_EVENT_DECAY_PERIODS)
        return None

    # WHY: 業界標準 IS/OOS split；更多 IS 資料讓基準 CAAR 估計更穩定，
    # OOS 段作為嚴格的盲測；與 OOS_Decay（selection 指標）保持一致
    split_date = dates[int(n_dates * 0.8)]
    is_events = events.filter(pl.col("date") < split_date)
    oos_events = events.filter(pl.col("date") >= split_date)

    is_signed = _signed_event_cars(is_events)
    oos_signed = _signed_event_cars(oos_events)
    mean_is = is_signed.mean() if is_signed is not None and len(is_signed) > 0 else None
    mean_oos = oos_signed.mean() if oos_signed is not None and len(oos_signed) > 0 else None

    if mean_is is None or mean_oos is None or abs(mean_is) < EPSILON:
        return None

    decay = abs(mean_oos) / abs(mean_is)
    # WHY: 異號表示 OOS 方向反轉，給予 0.5× 懲罰
    if (mean_is > 0) != (mean_oos > 0):
        decay *= 0.5

    # WHY: OOS t-stat 衡量 OOS alpha 的統計顯著性；定義與 Event_CAAR 一致，
    # 讓 adaptive_weight 在 OOS 樣本不足時自動降權此指標
    n_oos = len(oos_signed) if oos_signed is not None else 0
    if n_oos < 2:
        # WHY: std_oos 需要 n>=2（ddof=1）；MIN_EVENT_DECAY_PERIODS=15 在多資產下通常
        # 保證 ≥3 OOS rows，單資產 80/20 亦保證恰好 3 rows；n<2 代表資料有缺漏
        logger.warning("Event_Decay: only %d OOS samples (need >=2)", n_oos)
        return None
    std_oos = float(oos_signed.std(ddof=DDOF)) if n_oos > 1 else None
    t_stat = calc_t_stat(float(mean_oos), std_oos, n_oos) if std_oos else 0.0

    return MetricResult(map_linear(decay, min_val=0.3, max_val=1.0), t_stat, decay)


@register("Event_Stability")
def calc_event_stability(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """月度 signed CAAR 一致性：1 - CV(monthly_signed_CAAR)。

    將事件按月分組，每月取 signed_car 均值得到 monthly_CAAR 序列，
    再以變異係數（CV）衡量跨月的穩定性。CV 小 → stability 高 → 訊號在不同 regime 穩定。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（資料不足時）。

    統計假設：
        - stability = max(0, 1 - |CV|)，CV = std / |mean|（跨月）。
        - t_stat = mean(monthly_caar) / SEM，adaptive_weight 在月份數少時降權。
        - WHY: 原版衡量事件密度均勻性（CV of counts），與報酬無關；
          此版本對應 IC_Stability，衡量跨 regime 的 alpha 穩定性。
        - 評分範圍：stability 從 0.2（0分）到 0.8（100分）。
    """
    events = df.filter(pl.col("factor_raw") != 0)
    if events.is_empty():
        return None

    monthly = (
        events
        .with_columns([
            (pl.col("forward_return") * pl.col("factor_raw").sign()).alias("signed_car"),
            pl.col("date").dt.truncate("1mo").alias("month"),
        ])
        .group_by("month")
        .agg(pl.col("signed_car").mean().alias("monthly_caar"))
        .drop_nulls("monthly_caar")
    )

    if len(monthly) < MIN_EVENT_MONTHS:
        logger.warning("Event_Stability: only %d months (need >=%d)", len(monthly), MIN_EVENT_MONTHS)
        return None

    vals = monthly["monthly_caar"].to_numpy()
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=DDOF))

    if abs(mean_val) < EPSILON:
        # WHY: mean ≈ 0 → CV 分母趨零，穩定性無意義；訊號無方向性時 persistence 無從評估
        return None

    cv = std_val / abs(mean_val)
    stability = max(0.0, 1.0 - cv)
    t_stat = calc_t_stat(mean_val, std_val, len(vals))

    return MetricResult(map_linear(stability, min_val=0.2, max_val=0.8), float(t_stat), float(stability))


@register("Event_Hit_Rate")
def calc_event_hit_rate(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """事件方向正確率：signed_car > 0 的事件佔比。

    所有事件的 signed_car 不論來自哪支股票均池化計算。
    Buy 漲、Sell 跌皆計為命中（signed_car>0）。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（樣本不足時）。

    統計假設：
        - t_stat 使用二項分布 H0: p=0.5 的標準化統計量，近似 N(0,1)。
        - 事件視為獨立觀測；若 forward_return 窗口重疊（高頻事件），
          t_stat 可能因序列相關而偏大，適合月頻或更稀疏的事件訊號。
        - 評分範圍：hit_rate 從 0.45（0分）到 0.65（100分）。
    """
    signed = _signed_event_cars(df)
    if signed is None:
        return None

    n = len(signed)
    if n < MIN_EVENT_SAMPLE:
        logger.warning("Event_Hit_Rate: only %d event samples (need >=%d)", n, MIN_EVENT_SAMPLE)
        return None

    hit_rate = (signed > 0).sum() / n
    # WHY: 與 IC-based Hit_Rate 同模式；二項分布 p=0.5 下的 t-stat
    t_stat = (hit_rate - 0.5) / np.sqrt(0.25 / n)
    score = map_linear(hit_rate, 0.45, 0.65)
    return MetricResult(float(score), float(t_stat), float(hit_rate))


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

@register("Profit_Factor")
def calc_profit_factor(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """P&L 結構比：sum(signed_car > 0) / |sum(signed_car < 0)|。

    池化所有事件的 signed_car，計算總獲利相對於總虧損的倍數。
    衡量「贏的時候贏多少、輸的時候輸多少」，補充 Hit_Rate 的頻率資訊。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（樣本不足時）。

    統計假設：
        - Profit Factor 本身無標準誤公式；t_stat 借用 mean(signed_car)/SEM
          作為統計顯著性代理，與 Event_CAAR 定義一致。
        - 零虧損時 pf 設為 10.0（cap 避免 MLflow 序列化 inf）。
        - 評分範圍：pf 從 0.8（0分）到 2.0（100分）。
    """
    signed = _signed_event_cars(df)
    if signed is None:
        return None

    n = len(signed)
    if n < MIN_EVENT_SAMPLE:
        logger.warning("Profit_Factor: only %d event samples (need >=%d)", n, MIN_EVENT_SAMPLE)
        return None

    # WHY: signed_car 均值 t-stat 衡量整體 P&L 方向顯著性；
    # Profit Factor 本身是比值無標準誤，借用均值 t-stat 代理統計可靠度
    _mean_s = signed.mean()
    _std_s = signed.std(ddof=DDOF)
    mean_s = float(_mean_s) if _mean_s is not None else 0.0
    std_s = float(_std_s) if _std_s is not None else 0.0
    t_stat = calc_t_stat(mean_s, std_s, n)

    gains = signed.filter(signed > 0)
    losses = signed.filter(signed < 0)

    total_gains = gains.sum() if len(gains) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0

    if total_gains < EPSILON and total_losses < EPSILON:
        # WHY: 所有 signed_car 為 0，無效資料（例如：使用錯誤的 return 欄位）
        return None
    if total_losses < EPSILON:
        # WHY: 有獲利但零虧損，cap 於有限值避免序列化問題（MLflow parquet）
        return MetricResult(100.0, t_stat, 10.0)

    pf = total_gains / total_losses
    return MetricResult(map_linear(pf, min_val=0.8, max_val=2.0), t_stat, pf)


@register("Event_Skewness")
def calc_event_skewness(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """signed CAR 分布的偏態：衡量 P&L 尾部不對稱性。

    正偏（右尾肥）= 偶有大賺、多為小虧，比負偏（大虧偶有小賺）更理想。
    池化所有事件的 signed_car 計算樣本偏態。

    Args:
        df: 含 ``factor_raw``、``forward_return`` 的 prepared DataFrame。

    Returns:
        MetricResult 或 None（樣本不足時）。

    統計假設：
        - t_stat = skewness / √(6/n)，偏態的漸近標準誤，H0 下近似 N(0,1)。
        - WHY: MIN_SKEW_PERIODS=30 因小樣本偏態估計不穩（SE≈√(6/n)，n=10 時
          SE≈0.77，偏態可達 ±3），t-stat 保護可防止任意分數注入。
        - 評分範圍：skewness 從 -0.5（0分）到 +1.5（100分）。
    """
    signed = _signed_event_cars(df)
    if signed is None:
        return None

    n = len(signed)
    if n < MIN_SKEW_PERIODS:
        logger.warning("Event_Skewness: only %d event samples (need >=%d)", n, MIN_SKEW_PERIODS)
        return None

    skew = float(sp_stats.skew(signed.to_numpy(), bias=False))
    # WHY: SE of skewness ≈ sqrt(6/n) under normality；t = skew / SE
    t_stat = skew / np.sqrt(6.0 / n)
    score = map_linear(skew, min_val=-0.5, max_val=1.5)
    return MetricResult(score, float(t_stat), skew)


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------

@register("LS_Vol")
def calc_ls_vol(df: pl.DataFrame, **kwargs) -> MetricResult | None:
    """Placeholder: annualized L/S portfolio return volatility."""
    return None
