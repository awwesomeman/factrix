"""Core types for the Gate Pipeline.

GateFn = Callable[[Artifacts], GateResult]

Each gate is a plain function. Use ``functools.partial`` to bind
custom thresholds — no Protocol classes needed.

Artifacts is a typed, read-only container of pre-computed intermediate
results. All gates share the same Artifacts instance; none may mutate it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from factorlib.gates.config import PipelineConfig
from factorlib.tools._typing import MetricOutput

GateStatus = Literal["PASS", "FAILED", "VETOED"]
EvaluationStatus = Literal["PASS", "CAUTION", "VETOED", "FAILED"]


@dataclass
class GateResult:
    """Outcome of a single gate evaluation."""

    name: str
    status: GateStatus
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass
class Artifacts:
    """Pre-computed intermediate results shared across all gates.

    Built once by ``_build_artifacts`` before any gate runs.
    Gates read from this; none may write to it.

    Attributes:
        prepared: Preprocessed panel (date, asset_id, factor, forward_return, ...).
        ic_series: Output of ``compute_ic()`` — columns ``date, ic``.
        ic_values: IC series with ``ic`` renamed to ``value`` for series tools.
        spread_series: Output of ``compute_spread_series()``
            — columns ``date, spread, q1_return, q5_return, universe_return``.
        config: Pipeline configuration.
    """

    prepared: pl.DataFrame
    ic_series: pl.DataFrame
    ic_values: pl.DataFrame
    spread_series: pl.DataFrame
    config: PipelineConfig


@dataclass
class FactorProfile:
    """Factor profile containing raw metric outputs.

    Attributes:
        reliability: Metrics measuring signal quality
            (IC, IC_IR, Hit_Rate, IC_Trend, Monotonicity, OOS_Decay).
        profitability: Metrics measuring economic significance
            (Q1-Q5 Spread, Long/Short Alpha, Turnover, Breakeven Cost,
            Net Spread, Q1 Concentration).
    """

    reliability: list[MetricOutput] = field(default_factory=list)
    profitability: list[MetricOutput] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation output for a single factor.

    Attributes:
        factor_name: Factor identifier.
        status: Overall status.
        gate_results: Per-gate results (in evaluation order).
        profile: Factor profile (None if a gate failed/vetoed before profiling).
        caution_reasons: Human-readable reasons for CAUTION status.
    """

    factor_name: str
    status: EvaluationStatus
    gate_results: list[GateResult] = field(default_factory=list)
    profile: FactorProfile | None = None
    caution_reasons: list[str] = field(default_factory=list)


GateFn = Callable[[Artifacts], GateResult]
