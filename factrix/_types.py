"""Shared types and constants for all factrix tools.

Every tool returns ``MetricOutput``; every tool shares the same
numerical constants. This module is the single source of truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Literal, NewType


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


def coerce_factor_type(factor_type: "FactorType | str") -> "FactorType":
    """Accept a FactorType enum or its string value and return the enum.

    Centralizes the string→enum coercion used by public API entry points
    (``evaluate``, ``evaluate_batch``, ``describe_profile``,
    ``register_rule``). Surfaces the same error message regardless of
    call site so users can pattern-match it reliably.
    """
    if isinstance(factor_type, FactorType):
        return factor_type
    try:
        return FactorType(factor_type)
    except ValueError:
        valid = ", ".join(ft.value for ft in FactorType)
        raise ValueError(
            f"Unknown factor_type {factor_type!r}. "
            f"Supported: {valid}. "
            f"Use fl.describe_factor_types() for details."
        ) from None


# ---------------------------------------------------------------------------
# Missing-argument sentinel
# ---------------------------------------------------------------------------

class _Unset:
    """Sentinel marking "caller supplied no value".

    Distinct from ``None`` (which is a legitimate user-passable value) and
    from the function's default, which is what a reader of the signature
    sees. Used by ``_api.py`` / ``preprocess/pipeline.py`` to detect
    ``config=`` + ``**overrides`` double-pass without forcing callers to
    switch to ``**kwargs`` + ``pop`` gymnastics.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


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
# Minimum sample thresholds
# ---------------------------------------------------------------------------

MIN_IC_PERIODS: int = 10
MIN_EVENTS: int = 10
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

    def __repr__(self) -> str:
        parts = [f"{self.name}={self.value:.4f}"]
        if self.stat is not None:
            parts.append(f"stat={self.stat:.2f}")
        if self.significance:
            parts.append(f"sig={self.significance}")
        return f"MetricOutput({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Profile architecture types (new in v4 — see docs/gate_redesign_v2.md)
# ---------------------------------------------------------------------------

# WHY: PValue is structurally a float, but the NewType makes intent explicit
# and lets the whitelist in multiple_testing_correct() reject non-p fields
# (e.g. ic_ir, oos_decay) programmatically via field annotation inspection.
PValue = NewType("PValue", float)

# WHY: "CAUTION" / "SIGNIFICANT" shades still belong in Diagnostic;
# PASS_WITH_WARNINGS is strictly a UX signal that diagnose() is worth
# reading, not a severity level.
Verdict = Literal["PASS", "PASS_WITH_WARNINGS", "FAILED"]

# WHY: Severity is three-valued for practical triage. "info" is observation,
# "warn" suggests caution without blocking, "veto" is a deal-breaker that
# should override a PASS verdict in downstream filtering if the user opts in.
DiagnosticSeverity = Literal["info", "warn", "veto"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Contextual hint produced by ``Profile.diagnose()``.

    ``severity`` drives filtering; ``code`` is a stable machine-readable
    identifier for programmatic handling (e.g. AI agent triage).
    """

    severity: DiagnosticSeverity
    message: str
    code: str | None = None
    # WHY: when a rule fires because a risk diagnostic (clustering,
    # persistence, overlapping returns) suggests the canonical p-value
    # is under-stating uncertainty, ``recommended_p_source`` names the
    # whitelisted P_VALUE_FIELDS entry the user should consider for BHY
    # / verdict. None means the rule is purely informational with no
    # alternative test to recommend.
    recommended_p_source: str | None = None

    def __repr__(self) -> str:
        tag = f"[{self.severity}]"
        code_part = f" ({self.code})" if self.code else ""
        return f"Diagnostic{tag}{code_part} {self.message}"


# ---------------------------------------------------------------------------
# Metric-option Literal aliases
# ---------------------------------------------------------------------------
# Centralised so the public surface of ``compute_caar`` / ``fama_macbeth`` /
# ``top_concentration`` shares one canonical spelling, and so type checkers
# can flag a misspelt option from a single source of truth.

# Kolari-Pynnönen (2010) clustering correction for CAAR — which intra-day
# correlation source the Z is built from.
KPSource = Literal["icc", "no_multi_event_dates"]

# Shanken (1992) errors-in-variables correction — which σ²(λ) input feeds
# the second-stage variance inflation.
ShankenVarSource = Literal["user_supplied", "betas_timeseries_proxy"]

# Top-bucket concentration weight basis — pure factor magnitude vs.
# realised contribution to the long-leg's α.
ConcentrationWeight = Literal["abs_factor", "alpha_contribution"]
