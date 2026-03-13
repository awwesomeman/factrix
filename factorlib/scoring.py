"""
Layer 3: Scoring Core — Registry, metrics, and FactorScorer.
All metric functions are Pure Polars. Config uses string keys via registry.
Frequency-agnostic: forward_periods flows through FactorScorer to all metrics.
"""

import logging
import polars as pl
import numpy as np
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def map_linear(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100
    return float(np.clip(score, 0, 100))


def _ic_series(df: pl.DataFrame) -> pl.DataFrame:
    """Per-date Spearman Rank IC (Pure Polars).
    Uses method='average' for correct Spearman with tied values.
    """
    ranked = df.with_columns(
        pl.col("factor").rank(method="average").over("date").alias("rank_factor"),
        pl.col("forward_return").rank(method="average").over("date").alias("rank_return"),
    )
    return (
        ranked.group_by("date")
        .agg(
            pl.corr("rank_factor", "rank_return").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 10)
        .sort("date")
        .select("date", "ic")
    )


def _non_overlapping_dates(df: pl.DataFrame, step: int) -> pl.Series:
    """Sample every N-th date to avoid overlapping forward returns."""
    return df["date"].unique().sort().gather_every(step)


def _rolling_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """Create rolling windows via numpy stride tricks (zero-copy)."""
    shape = (arr.shape[0] - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


# Minimum sample thresholds (in number of IC periods)
MIN_IC_PERIODS = 10          # Rank_IC, IC_IR
MIN_STABILITY_PERIODS = 12   # IC_Stability (adaptive window = n//3, so need >=12 for window>=4)
MIN_OOS_PERIODS = 5          # Internal_OOS_Decay (OOS partition)
MIN_PORTFOLIO_PERIODS = 5    # Long_Only_Alpha, MDD


# ---------------------------------------------------------------------------
# Alpha Metrics
# ---------------------------------------------------------------------------

@register("Rank_IC")
def calc_rank_ic(df: pl.DataFrame, _ic_cache: pl.DataFrame | None = None, **kwargs) -> float:
    """Mean |Rank IC|. Good factor: 0.03~0.10."""
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    n = len(ic_df)
    if n < MIN_IC_PERIODS:
        logger.warning("Rank_IC: only %d IC periods (need >=%d), score unreliable", n, MIN_IC_PERIODS)
    mean_ic = ic_df["ic"].mean()
    if mean_ic is None:
        return 0.0
    return map_linear(abs(mean_ic), min_val=0.01, max_val=0.08)


@register("IC_IR")
def calc_ic_ir(df: pl.DataFrame, _ic_cache: pl.DataFrame | None = None, **kwargs) -> float:
    """|mean(IC)| / std(IC). Good: 0.3~0.8."""
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    ic_vals = ic_df["ic"].drop_nulls()
    if len(ic_vals) < MIN_IC_PERIODS:
        logger.warning("IC_IR: only %d IC periods (need >=%d), skipping", len(ic_vals), MIN_IC_PERIODS)
        return 0.0
    mean_ic = ic_vals.mean()
    std_ic = ic_vals.std()
    if std_ic is None or std_ic == 0 or mean_ic is None:
        return 0.0
    return map_linear(abs(mean_ic) / std_ic, min_val=0.1, max_val=0.6)


@register("Long_Only_Alpha")
def calc_long_only_alpha(df: pl.DataFrame, q_top: float = 0.2, _forward_periods: int = 5, **kwargs) -> float:
    """
    Q1 excess return vs universe mean (non-overlapping).
    Annualized using actual date range for precision.
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
            .mean()
            .alias("q1_return"),
            pl.col("forward_return").mean().alias("univ_return"),
        )
        .with_columns(
            (pl.col("q1_return") - pl.col("univ_return")).alias("excess")
        )
        .sort("date")
    )

    excess_vals = q1_excess["excess"].drop_nulls()
    if len(excess_vals) < MIN_PORTFOLIO_PERIODS:
        logger.warning("Long_Only_Alpha: only %d non-overlapping periods (need >=%d)", len(excess_vals), MIN_PORTFOLIO_PERIODS)
        return 0.0

    nav = np.cumprod(1 + excess_vals.to_numpy())
    total_excess = nav[-1] - 1

    date_range = q1_excess["date"].drop_nulls()
    n_years = (date_range.max() - date_range.min()).days / 365.25
    if n_years < 0.1:
        logger.warning("Long_Only_Alpha: date range < 0.1 years, cannot annualize")
        return 0.0
    ann_excess = (1 + total_excess) ** (1 / n_years) - 1

    return map_linear(ann_excess, min_val=0.0, max_val=0.20)


# ---------------------------------------------------------------------------
# Robustness Metrics
# ---------------------------------------------------------------------------

@register("Internal_OOS_Decay")
def calc_internal_oos_decay(df: pl.DataFrame, oos_ratio: float = 0.2, _ic_cache: pl.DataFrame | None = None, **kwargs) -> float:
    """IS vs OOS |IC| comparison with sign-flip penalty."""
    ic_df = (_ic_cache if _ic_cache is not None else _ic_series(df)).sort("date")
    dates = ic_df["date"].unique().sort()
    split_idx = int(len(dates) * (1 - oos_ratio))

    n_oos = len(dates) - split_idx
    if n_oos < MIN_OOS_PERIODS:
        logger.warning("Internal_OOS_Decay: only %d OOS periods (need >=%d), score unreliable", n_oos, MIN_OOS_PERIODS)
    if split_idx < MIN_OOS_PERIODS:
        logger.warning("Internal_OOS_Decay: only %d IS periods (need >=%d), score unreliable", split_idx, MIN_OOS_PERIODS)

    split_date = dates[split_idx]
    mean_is = ic_df.filter(pl.col("date") < split_date)["ic"].mean()
    mean_oos = ic_df.filter(pl.col("date") >= split_date)["ic"].mean()

    if mean_is is None or mean_oos is None or abs(mean_is) < 1e-6:
        return 0.0

    sign_consistent = (mean_is > 0 and mean_oos > 0) or (mean_is < 0 and mean_oos < 0)
    decay = abs(mean_oos) / abs(mean_is)
    if not sign_consistent:
        decay *= 0.5

    return map_linear(decay, min_val=0.3, max_val=1.0)


@register("IC_Stability")
def calc_ic_stability(df: pl.DataFrame, _ic_cache: pl.DataFrame | None = None, **kwargs) -> float:
    """Rolling-window IC_IR consistency (vectorized via stride tricks).
    Window adapts to sample size: min(12, n//3), minimum 4 periods.
    """
    ic_df = _ic_cache if _ic_cache is not None else _ic_series(df)
    ic_vals = ic_df["ic"].drop_nulls().to_numpy().copy()

    if len(ic_vals) < MIN_STABILITY_PERIODS:
        logger.warning("IC_Stability: only %d IC periods (need >=%d), skipping", len(ic_vals), MIN_STABILITY_PERIODS)
        return 0.0

    window = max(4, min(12, len(ic_vals) // 3))
    windows = _rolling_windows(ic_vals, window)

    means = np.abs(windows.mean(axis=1))
    stds = windows.std(axis=1)
    valid = stds > 1e-8
    if not valid.any():
        return 0.0

    rolling_irs = means[valid] / stds[valid]
    return map_linear(float(rolling_irs.mean()), min_val=0.1, max_val=0.5)


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

@register("Turnover")
def calc_turnover(df: pl.DataFrame, **kwargs) -> float:
    """Rank autocorrelation (vectorized). High = low turnover = lower costs."""
    dates = df["date"].unique().sort()
    if len(dates) < 2:
        logger.warning("Turnover: need at least 2 dates")
        return 50.0

    date_map = pl.DataFrame({
        "date": dates[1:],
        "prev_date": dates[:-1],
    })

    curr = df.select(
        "date", "asset_id",
        pl.col("factor").rank(method="average").over("date").alias("rank_curr"),
    )
    prev = df.select(
        pl.col("date").alias("prev_date"),
        "asset_id",
        pl.col("factor").rank(method="average").over("date").alias("rank_prev"),
    )

    paired = (
        curr.join(date_map, on="date")
        .join(prev, on=["prev_date", "asset_id"])
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
        return 50.0

    avg_rc = rc_df["rc"].mean()
    if avg_rc is None:
        return 50.0
    if avg_rc > 0.99:
        return 80.0
    return map_linear(avg_rc, min_val=0.5, max_val=0.95)


@register("MDD")
def calc_mdd(df: pl.DataFrame, _forward_periods: int = 5, **kwargs) -> float:
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
            pl.col("forward_return").filter(pl.col("pct_rank") >= 0.8).mean().alias("long_ret"),
            pl.col("forward_return").filter(pl.col("pct_rank") <= 0.2).mean().alias("short_ret"),
        )
        .with_columns((pl.col("long_ret") - pl.col("short_ret")).alias("ls_ret"))
        .sort("date")
    )

    ls_rets = ls_df["ls_ret"].drop_nulls().to_numpy()
    if len(ls_rets) < MIN_PORTFOLIO_PERIODS:
        logger.warning("MDD: only %d non-overlapping periods (need >=%d)", len(ls_rets), MIN_PORTFOLIO_PERIODS)
        return 50.0

    nav = np.cumprod(1 + ls_rets)
    running_max = np.maximum.accumulate(nav)
    mdd = abs((nav / running_max - 1).min())

    return map_linear(1 - mdd, min_val=0.5, max_val=0.9)


@register("Orthogonality")
def calc_orthogonality(df: pl.DataFrame, **kwargs) -> float:
    """Placeholder: requires factor zoo. Returns None to signal 'skip'."""
    return None


# ---------------------------------------------------------------------------
# Scoring Config (serializable — string keys only)
# ---------------------------------------------------------------------------

SCORING_CONFIG = {
    "Alpha": {
        "weight": 0.30,
        "metrics": {
            "Rank_IC": {"weight": 0.4},
            "IC_IR": {"weight": 0.2},
            "Long_Only_Alpha": {"weight": 0.4, "q_top": 0.2},
        },
    },
    "Robustness": {
        "weight": 0.35,
        "metrics": {
            "Internal_OOS_Decay": {"weight": 0.6, "oos_ratio": 0.2},
            "IC_Stability": {"weight": 0.4},
        },
    },
    "Risk": {
        "weight": 0.25,
        "metrics": {
            "Turnover": {"weight": 0.5, "min_threshold": 20},
            "MDD": {"weight": 0.5},
        },
    },
    "Novelty": {
        "weight": 0.10,
        "metrics": {
            "Orthogonality": {"weight": 1.0},
        },
    },
}


# ---------------------------------------------------------------------------
# FactorScorer
# ---------------------------------------------------------------------------

class FactorScorer:
    def __init__(
        self,
        prepared_data: pl.DataFrame,
        config: dict | None = None,
        forward_periods: int = 5,
    ):
        self.data = prepared_data
        self.config = config or SCORING_CONFIG
        self.forward_periods = forward_periods
        self._ic_cache: pl.DataFrame | None = None

    @property
    def ic_series(self) -> pl.DataFrame:
        """Cached IC series — computed once, reused across metrics."""
        if self._ic_cache is None:
            self._ic_cache = _ic_series(self.data)
        return self._ic_cache

    def compute(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "total": 0.0,
            "dimensions": {},
            "penalties": [],
        }
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_name, dim_cfg in self.config.items():
            dim_w_sum = 0.0
            active_w = 0.0
            metric_scores = {}

            for m_name, m_cfg in dim_cfg["metrics"].items():
                func = METRIC_REGISTRY.get(m_name)
                if func is None:
                    raise ValueError(f"Unknown metric: {m_name}")

                extra = {k: v for k, v in m_cfg.items() if k not in ("weight", "min_threshold")}
                score = func(
                    self.data,
                    _ic_cache=self.ic_series,
                    _forward_periods=self.forward_periods,
                    **extra,
                )

                if score is None:
                    continue

                metric_scores[m_name] = round(score, 2)

                threshold = m_cfg.get("min_threshold")
                if threshold is not None and score < threshold:
                    results["penalties"].append(
                        f"VETO: {m_name} score {score:.1f} below {threshold}"
                    )

                dim_w_sum += score * m_cfg["weight"]
                active_w += m_cfg["weight"]

            if active_w == 0:
                continue

            dim_final = dim_w_sum / active_w
            results["dimensions"][dim_name] = {
                "score": round(dim_final, 2),
                "metrics": metric_scores,
            }
            weighted_sum += dim_final * dim_cfg["weight"]
            total_weight += dim_cfg["weight"]

        raw_total = weighted_sum / total_weight if total_weight > 0 else 0
        results["total"] = round(raw_total * 0.2, 2) if results["penalties"] else round(raw_total, 2)

        return results
