"""Shared types and constants for all factorlib tools.

Every tool returns ``MetricOutput``; every tool shares the same
numerical constants. This module is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# WHY: 統一除零保護門檻，避免 std ≈ 0 時 t-stat 膨脹至天文數字
EPSILON: float = 1e-9

# WHY: ddof=1 為樣本標準差（業界主流），全專案統一避免跨指標比較產生系統性偏差
# Polars .std() 預設 ddof=1，NumPy 需顯式傳入
DDOF: int = 1

# WHY: 日曆天年化因子，集中管理避免魔術數字散落各指標
CALENDAR_DAYS_PER_YEAR: float = 365.25

# WHY: 1.4826 = 1/Φ⁻¹(0.75)，使 MAD 成為常態分佈下 σ 的無偏估計量
MAD_CONSISTENCY_CONSTANT: float = 1.4826


# ---------------------------------------------------------------------------
# Minimum sample thresholds
# ---------------------------------------------------------------------------

MIN_IC_PERIODS: int = 10
MIN_OOS_PERIODS: int = 5
MIN_PORTFOLIO_PERIODS: int = 5
MIN_MONOTONICITY_PERIODS: int = 5


# ---------------------------------------------------------------------------
# Unified output type
# ---------------------------------------------------------------------------

@dataclass
class MetricOutput:
    """Unified return type for all tools.

    Every tool function returns ``MetricOutput``.  The ``name`` and ``value``
    fields are mandatory; everything else is optional context.

    Args:
        name: Metric identifier (e.g. "IC_IR", "OOS_Decay").
        value: Raw metric value — never mapped to 0-100.
        t_stat: t-statistic, when applicable.
        significance: "***" (|t| >= 3.0) / "**" (|t| >= 2.0) / "*" (|t| >= 1.65) / "" (ns).
        metadata: Tool-specific context (e.g. per-split details, via path).
    """

    name: str
    value: float
    t_stat: float | None = None
    significance: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
