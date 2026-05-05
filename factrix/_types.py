"""Shared types and constants for the v0.4 metric primitives.

The v0.5 axis enums (``FactorScope`` / ``Signal`` / ``Metric`` / ``Mode``)
live in :mod:`factrix._axis`; the v0.5 result type
(``FactorProfile``) lives in :mod:`factrix._profile`. This module
keeps only the numerical constants and ``MetricOutput`` shared by the
``factrix.metrics.*`` primitives that v0.5 procedures wrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NewType

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# WHY: 統一除零保護門檻，避免 std ≈ 0 時 t-stat 膨脹至天文數字
EPSILON: float = 1e-9

# WHY: ddof=1 為樣本標準差（業界主流），全專案統一避免跨指標比較產生系統性偏差
# Polars .std() 預設 ddof=1，NumPy 需顯式傳入
DDOF: int = 1

# WHY: 1.4826 = 1/Φ⁻¹(0.75)，使 MAD 成為常態分佈下 σ 的無偏估計量
MAD_CONSISTENCY_CONSTANT: float = 1.4826


# ---------------------------------------------------------------------------
# Minimum sample thresholds (used by metrics primitives)
# ---------------------------------------------------------------------------

# Per-date minimum asset count below which ``compute_ic`` drops the date.
# Renamed from MIN_IC_PERIODS in #18 — the "PERIODS" suffix was misleading;
# the value has always been checked against per-date asset counts.
MIN_ASSETS_PER_DATE_IC: int = 10
MIN_EVENTS: int = 10
MIN_OOS_PERIODS: int = 5
MIN_PORTFOLIO_PERIODS: int = 5
MIN_MONOTONICITY_PERIODS: int = 5


# ---------------------------------------------------------------------------
# Unified output type for metric primitives
# ---------------------------------------------------------------------------


@dataclass
class MetricOutput:
    """Return type for ``factrix.metrics.*`` primitives.

    Args:
        name: Metric identifier (e.g. "ic_ir", "oos_decay").
        value: Raw metric value.
        stat: Test statistic (t, z, W, chi2, ...), when applicable.
        significance: ``***`` / ``**`` / ``*`` / ``""`` derived from
            ``metadata["p_value"]`` when available.
        metadata: Tool-specific context (``p_value``, ``stat_type``,
            ``h0``, ``method`` are the standard keys).
    """

    name: str
    value: float
    stat: float | None = None
    significance: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"{self.name}={self.value:.4f}"]
        if self.stat is not None:
            parts.append(f"stat={self.stat:.2f}")
        if self.significance:
            parts.append(f"sig={self.significance}")
        return f"MetricOutput({', '.join(parts)})"


# Structural alias used by metric internals to mark "this float is a
# p-value, not an effect-size". Carried for the metrics surface; the
# v0.5 profile schema uses ``primary_p: float`` directly.
PValue = NewType("PValue", float)


# ---------------------------------------------------------------------------
# Metric-option Literal aliases
# ---------------------------------------------------------------------------

# Kolari-Pynnönen (2010) clustering correction for CAAR — which intra-day
# correlation source the Z is built from.
KPSource = Literal["icc", "no_multi_event_dates"]

# Shanken (1992) errors-in-variables correction — which σ²(λ) input feeds
# the second-stage variance inflation.
ShankenVarSource = Literal["user_supplied", "betas_timeseries_proxy"]

# Top-bucket concentration weight basis — pure factor magnitude vs.
# realised contribution to the long-leg's α.
ConcentrationWeight = Literal["abs_factor", "alpha_contribution"]
