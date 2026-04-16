"""Shared types and constants for all factorlib tools.

Every tool returns ``MetricOutput``; every tool shares the same
numerical constants. This module is the single source of truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Factor type taxonomy
# ---------------------------------------------------------------------------

class FactorType(enum.StrEnum):
    """Supported factor type categories.

    Determines which metrics, gates, and profile logic apply.
    """

    CROSS_SECTIONAL = "cross_sectional"
    EVENT_SIGNAL = "event_signal"
    MACRO_PANEL = "macro_panel"
    MACRO_COMMON = "macro_common"


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
        name: Metric identifier (e.g. "ic_ir", "oos_decay").
        value: Raw metric value — never mapped to 0-100.
        stat: Test statistic (t, z, W, chi2, ...), when applicable.
        significance: "***" (p < 0.01) / "**" (p < 0.05) / "*" (p < 0.10) / "" (ns).
            Derived from ``metadata["p_value"]`` when available.
        metadata: Tool-specific context. Standard keys:
            - ``p_value`` (float): p-value for significance determination.
            - ``stat_type`` (str): "t" | "z" | "wilcoxon" | "bootstrap" | ...
            - ``h0`` (str): null hypothesis, e.g. "mu=0", "p=0.5".
            - ``method`` (str): e.g. "non-overlapping t-test", "binomial score".
    """

    name: str
    value: float
    stat: float | None = None
    significance: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
