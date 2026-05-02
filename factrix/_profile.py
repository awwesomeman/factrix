"""v0.5 ``FactorProfile`` — single unified result type (§4.4.2 B3).

Replaces the per-cell ``CrossSectionalProfile`` / ``EventProfile`` /
``MacroPanelProfile`` / ``MacroCommonProfile`` proliferation: every cell
produces an instance of this dataclass, with cell-specific scalars
keyed in the ``stats`` mapping. Adding a new metric does not grow the
schema (§7.5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from factrix._axis import Mode
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig


@dataclass(frozen=True, slots=True)
class FactorProfile:
    """Procedure-canonical analysis result for one factor.

    ``n_obs`` is the cell-canonical effective sample size (varies by
    procedure: T for IC/FM/TS, event count for CAAR, asset count for
    COMMON×* PANEL). ``n_assets`` is the cross-section width of the
    raw panel (always ``panel["asset_id"].n_unique()``); reading both
    side by side disambiguates whether a small ``n_obs`` came from a
    short series or a thin cross-section.
    """

    config: "AnalysisConfig"
    mode: Mode
    primary_p: float
    n_obs: int
    n_assets: int
    warnings: frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode] = frozenset()
    stats: Mapping[StatCode, float] = field(default_factory=dict)

    def verdict(
        self,
        *,
        threshold: float = 0.05,
        gate: StatCode | None = None,
    ) -> Verdict:
        """Pass/fail at ``threshold`` against ``primary_p`` (or ``gate``).

        ``threshold`` is a generic cutoff — not tied to Type-I-error
        semantics, since ``gate`` may name a non-p stat (t-stat, HHI,
        etc.). ``gate=None`` uses the procedure-canonical ``primary_p``;
        supplying a ``StatCode`` swaps the gate for user policy and the
        comparison ``value < threshold`` is interpreted by the caller.
        Raises ``KeyError`` if the requested gate is not populated.
        """
        p = self.primary_p if gate is None else self.stats[gate]
        return Verdict.PASS if p < threshold else Verdict.FAIL

    def diagnose(self) -> dict[str, Any]:
        """Secondary stats + flag sets for human / AI agent triage."""
        return {
            "mode": self.mode.value,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "primary_p": self.primary_p,
            "warnings": sorted(w.value for w in self.warnings),
            "info_notes": sorted(i.value for i in self.info_notes),
            "stats": {k.value: v for k, v in self.stats.items()},
        }
