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

    Reading ``n_obs`` and ``n_assets`` side by side disambiguates
    whether a small ``n_obs`` came from a short series or a thin
    cross-section.

    Attributes:
        config: The ``AnalysisConfig`` that produced this profile.
        mode: Evaluation mode derived from raw data; ``PANEL`` when
            ``n_assets > 1``, ``TIMESERIES`` at ``N == 1``.
        primary_p: Procedure-canonical p-value used by ``verdict()``
            and ``multi_factor.bhy``.
        n_obs: Cell-canonical effective sample size (T for IC/FM/TS,
            event count for CAAR, asset count for ``COMMON × *``
            PANEL).
        n_assets: Cross-section width of the raw panel
            (``panel["asset_id"].n_unique()``).
        warnings: ``WarningCode`` flags emitted by the procedure.
        info_notes: ``InfoCode`` annotations (e.g. axis collapses).
        stats: Cell-specific scalars keyed by ``StatCode`` (t-stats,
            secondary p-values, HHI, etc.).
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
        semantics, because ``gate`` may name a non-p stat (t-stat,
        HHI, etc.). The comparison ``value < threshold`` is
        interpreted by the caller.

        Args:
            threshold: Cutoff applied to the gated value. Default
                ``0.05``.
            gate: ``StatCode`` whose value is read from ``stats``;
                ``None`` uses the procedure-canonical ``primary_p``.

        Returns:
            ``Verdict.PASS`` if the gated value is below ``threshold``,
            otherwise ``Verdict.FAIL``.

        Raises:
            KeyError: If ``gate`` is not populated in ``stats``.
        """
        p = self.primary_p if gate is None else self.stats[gate]
        return Verdict.PASS if p < threshold else Verdict.FAIL

    def diagnose(self) -> dict[str, Any]:
        """Secondary stats + flag sets for human / AI agent triage.

        Returns:
            A plain-Python dict with mode, sample sizes, primary p,
            warning / info code names sorted alphabetically, and the
            full ``stats`` mapping with enum keys converted to their
            string values.
        """
        return {
            "mode": self.mode.value,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "primary_p": self.primary_p,
            "warnings": sorted(w.value for w in self.warnings),
            "info_notes": sorted(i.value for i in self.info_notes),
            "stats": {k.value: v for k, v in self.stats.items()},
        }
