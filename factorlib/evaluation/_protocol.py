"""Core types for the evaluation pipeline.

GateFn = Callable[[Artifacts], GateResult]

Each gate is a plain function. Use ``functools.partial`` to bind
custom thresholds — no Protocol classes needed.

Artifacts is a read-only container of pre-computed intermediate
results. All gates share the same Artifacts instance; none may mutate it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Literal

import polars as pl

from factorlib.config import BaseConfig
from factorlib._types import MetricOutput

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


class _CompactedPrepared:
    """Placeholder substituted for ``Artifacts.prepared`` in compact mode.

    Any access -- attribute, indexing, truth-testing, iteration -- raises
    a targeted ``RuntimeError``. Slot-based dunders bypass ``__getattr__``
    so each must be listed explicitly; otherwise ``bool(prepared)`` would
    silently return True and ``if art.prepared:`` would give the wrong
    answer.
    """

    __slots__ = ()

    _MSG = (
        "Cannot use 'Artifacts.prepared': Artifacts is in compact mode "
        "(prepared DataFrame was dropped to save memory). Rebuild the "
        "Artifacts without compact=True if you need the prepared panel."
    )

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(
            f"Cannot access 'prepared.{name}': Artifacts is in compact mode "
            f"(prepared DataFrame was dropped to save memory). Rebuild the "
            f"Artifacts without compact=True if you need the prepared panel."
        )

    def __bool__(self) -> bool:
        raise RuntimeError(self._MSG)

    def __len__(self) -> int:
        raise RuntimeError(self._MSG)

    def __iter__(self):
        raise RuntimeError(self._MSG)

    def __getitem__(self, key):
        raise RuntimeError(self._MSG)

    def __contains__(self, item) -> bool:
        raise RuntimeError(self._MSG)

    def __repr__(self) -> str:
        return "<CompactedPrepared: prepared DataFrame dropped>"


_COMPACTED_PREPARED = _CompactedPrepared()


@dataclass
class Artifacts:
    """Pre-computed intermediate results shared across all gates.

    Built once by ``build_artifacts`` before any gate runs.
    Gates read from this; none may write to it.

    ``intermediates`` holds type-specific DataFrames (e.g. ic_series,
    spread_series for cross-sectional). Use ``.get(key)`` for access
    with a helpful KeyError on missing keys.

    ``factor_name`` identifies which factor this instance represents —
    consumed by per-type Profile ``from_artifacts`` classmethods and by
    downstream plotting / reporting that needs to label outputs.

    ``compact`` toggles memory-saving mode: when True, ``prepared`` is
    replaced with a sentinel that raises on any attribute access. Use
    for 1000-factor batches where the prepared panel (~MB per factor)
    would exhaust memory. Intermediates (small DataFrames) are kept
    because metrics and diagnose() need them.
    """

    prepared: pl.DataFrame
    config: BaseConfig
    intermediates: dict[str, pl.DataFrame] = field(default_factory=dict)
    factor_name: str = ""
    compact: bool = False

    def __post_init__(self) -> None:
        if self.compact:
            # WHY: object.__setattr__ bypasses any dataclass frozen-machinery
            # (we're not frozen today, but keeps the intent explicit).
            object.__setattr__(self, "prepared", _COMPACTED_PREPARED)

    def get(self, key: str) -> pl.DataFrame:
        if key not in self.intermediates:
            ft = type(self.config).factor_type
            raise KeyError(
                f"Artifacts has no '{key}'. "
                f"Available for {ft}: {list(self.intermediates.keys())}"
            )
        return self.intermediates[key]


@dataclass
class FactorProfile:
    """Factor profile — flat list of metric outputs.

    ``metrics``: all reliability + profitability metrics.
    ``attribution``: spanning alpha and related metrics (optional).
    """

    metrics: list[MetricOutput] = field(default_factory=list)
    attribution: list[MetricOutput] = field(default_factory=list)

    def get(self, name: str) -> MetricOutput | None:
        for m in chain(self.metrics, self.attribution):
            if m.name == name:
                return m
        return None


@dataclass
class EvaluationResult:
    """Complete evaluation output for a single factor."""

    factor_name: str
    status: EvaluationStatus
    gate_results: list[GateResult] = field(default_factory=list)
    profile: FactorProfile | None = None
    artifacts: Artifacts | None = None
    caution_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "factor_name": self.factor_name,
            "status": self.status,
            "caution_reasons": self.caution_reasons,
            "gate_results": [
                {"name": g.name, "status": g.status, "detail": g.detail}
                for g in self.gate_results
            ],
        }
        if self.profile:
            d["metrics"] = [
                {
                    "name": m.name, "value": m.value,
                    "stat": m.stat, "significance": m.significance,
                }
                for m in self.profile.metrics
            ]
            d["attribution"] = [
                {
                    "name": m.name, "value": m.value,
                    "stat": m.stat, "significance": m.significance,
                }
                for m in self.profile.attribution
            ]
        return d

    def to_dataframe(self) -> pl.DataFrame:
        if not self.profile:
            return pl.DataFrame()
        rows = []
        for m in self.profile.metrics + self.profile.attribution:
            rows.append({
                "factor": self.factor_name,
                "metric": m.name,
                "value": m.value,
                "stat": m.stat,
                "significance": m.significance or "",
            })
        return pl.DataFrame(rows)

    def __repr__(self) -> str:
        return _format_result(self)


GateFn = Callable[[Artifacts], GateResult]


# ---------------------------------------------------------------------------
# __repr__ helper
# ---------------------------------------------------------------------------

def _format_result(r: EvaluationResult) -> str:
    lines = [f"Factor: {r.factor_name} | Status: {r.status}"]

    if r.gate_results:
        lines.append("")
        lines.append("Gates:")
        for g in r.gate_results:
            via = g.detail.get("via")
            via_str = f" (via {', '.join(via)})" if via else ""
            lines.append(f"  {g.name}: {g.status}{via_str}")

    if r.profile:
        if r.profile.metrics:
            lines.append("")
            lines.append(_format_metric_table(r.profile.metrics))
        if r.profile.attribution:
            lines.append("")
            lines.append("Attribution:")
            lines.append(_format_metric_table(r.profile.attribution))

    if r.caution_reasons:
        lines.append("")
        lines.append("Caution:")
        for reason in r.caution_reasons:
            lines.append(f"  - {reason}")

    return "\n".join(lines)


def _format_metric_table(metrics: list[MetricOutput]) -> str:
    col_m = max(len(m.name) for m in metrics)
    col_m = max(col_m, 6)  # minimum width for "metric"
    col_v = 8
    col_t = 8
    col_s = 5

    def row(m_val: str, v_val: str, t_val: str, s_val: str) -> str:
        return (
            f"│ {m_val:<{col_m}} "
            f"│ {v_val:>{col_v}} "
            f"│ {t_val:>{col_t}} "
            f"│ {s_val:>{col_s}} │"
        )

    sep_top = f"┌{'─' * (col_m + 2)}┬{'─' * (col_v + 2)}┬{'─' * (col_t + 2)}┬{'─' * (col_s + 2)}┐"
    sep_mid = f"├{'─' * (col_m + 2)}┼{'─' * (col_v + 2)}┼{'─' * (col_t + 2)}┼{'─' * (col_s + 2)}┤"
    sep_bot = f"└{'─' * (col_m + 2)}┴{'─' * (col_v + 2)}┴{'─' * (col_t + 2)}┴{'─' * (col_s + 2)}┘"

    lines = [sep_top, row("metric", "value", "stat", "sig"), sep_mid]
    for m in metrics:
        v_str = f"{m.value:.4f}" if abs(m.value) < 100 else f"{m.value:.2f}"
        t_str = f"{m.stat:.2f}" if m.stat is not None else ""
        s_str = m.significance or ""
        lines.append(row(m.name, v_str, t_str, s_str))
    lines.append(sep_bot)
    return "\n".join(lines)
