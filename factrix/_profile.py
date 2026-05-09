"""v0.5 ``FactorProfile`` — single unified result type (§4.4.2 B3).

Replaces the per-cell ``CrossSectionalProfile`` / ``EventProfile`` /
``MacroPanelProfile`` / ``MacroCommonProfile`` proliferation: every cell
produces an instance of this dataclass, with cell-specific scalars
keyed in the ``stats`` mapping. Adding a new metric does not grow the
schema (§7.5).
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from factrix._axis import Mode
from factrix._codes import InfoCode, StatCode, Verdict, WarningCode

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig


@dataclass(frozen=True, slots=True, repr=False)
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
            densified panel-period count for CAAR, asset count for
            ``COMMON × *`` PANEL).
        n_assets: Cross-section width of the raw panel
            (``panel["asset_id"].n_unique()``).
        identity: ``(factor_id, forward_periods)`` hypothesis tuple.
            Defines "what hypothesis this profile tests"; consumed by
            ``factrix.multi_factor.bhy`` for family partitioning.
            Stamped by ``_evaluate`` from ``factor_col`` and
            ``config.forward_periods``.
        context: Sample-restriction / conditioning dimensions
            (``universe_id``, ``regime_id``, future axes). Empty by
            default; populated by higher-level verbs (``by_slice`` /
            ``by_regime`` consumers, future ``run_metrics``). The
            split from ``identity`` is the v1 anti-shopping defense:
            multi-horizon factor research's family forms naturally
            from ``identity``, while sample restrictions stay
            queryable via ``profile.context[key]``.
        warnings: ``WarningCode`` flags emitted by the procedure.
        info_notes: ``InfoCode`` annotations (e.g. axis collapses).
        stats: Cell-specific scalars keyed by ``StatCode`` (t-stats,
            secondary p-values, HHI, etc.).
    """

    config: AnalysisConfig
    mode: Mode
    primary_p: float
    n_obs: int
    n_assets: int
    identity: tuple[str, int]
    context: Mapping[str, Any] = field(default_factory=dict)
    warnings: frozenset[WarningCode] = frozenset()
    info_notes: frozenset[InfoCode] = frozenset()
    stats: Mapping[StatCode, float] = field(default_factory=dict)

    @property
    def factor_id(self) -> str:
        return self.identity[0]

    @property
    def forward_periods(self) -> int:
        return self.identity[1]

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

    def __repr__(self) -> str:
        parts = [
            f"factor_id={self.factor_id!r}",
            f"forward_periods={self.forward_periods}",
            f"mode={self.mode.value}",
            f"primary_p={self.primary_p:.4g}",
            f"n_obs={self.n_obs}",
            f"n_assets={self.n_assets}",
        ]
        if self.context:
            parts.append(f"context={dict(self.context)!r}")
        if self.warnings:
            parts.append(f"warnings={sorted(w.value for w in self.warnings)!r}")
        return f"FactorProfile({', '.join(parts)})"

    def _repr_html_(self) -> str:
        rows: list[tuple[str, Any]] = [
            ("factor_id", self.factor_id),
            ("forward_periods", self.forward_periods),
            ("mode", self.mode.value),
            ("primary_p", f"{self.primary_p:.4g}"),
            ("n_obs", self.n_obs),
            ("n_assets", self.n_assets),
        ]
        for k in sorted(self.context):
            rows.append((f"context.{k}", self.context[k]))
        if self.warnings:
            rows.append(("warnings", ", ".join(sorted(w.value for w in self.warnings))))
        body = "".join(
            f"<tr><th style='text-align:left'>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        return (
            "<table class='factrix-factor-profile'>"
            "<caption>FactorProfile</caption>"
            f"{body}</table>"
        )

    def diagnose(self) -> dict[str, Any]:
        """Secondary stats + flag sets for human / AI agent triage.

        Returns:
            A plain-Python dict with mode, sample sizes, primary p,
            warning / info code names sorted alphabetically, and the
            full ``stats`` mapping with enum keys converted to their
            string values.
        """
        return {
            "identity": {
                "factor_id": self.factor_id,
                "forward_periods": self.forward_periods,
            },
            "context": dict(self.context),
            "mode": self.mode.value,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "primary_p": self.primary_p,
            "warnings": sorted(w.value for w in self.warnings),
            "info_notes": sorted(i.value for i in self.info_notes),
            "stats": {k.value: v for k, v in self.stats.items()},
        }
